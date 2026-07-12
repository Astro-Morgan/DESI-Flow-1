"""Training helpers for Plato: DDP utilities, the CCF ground-truth similarity,
the triplet loss, and the hard-triplet-mining HDF5 dataset.
"""

import os
import sys
import json
import random

import numpy as np
import h5py
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset

# Repo root on sys.path so the shared preprocessing module resolves no matter
# where the script is launched from (this repo is a code reference, not a
# pip-installable package).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import preprocess_for_model  # noqa: E402


# ----------------------------------------------------------------------------
# 1. DDP & LOGGING UTILITIES
# ----------------------------------------------------------------------------

def setup_ddp():
    """Initialize the DDP process group (SLURM-aware).

    On SLURM clusters (e.g. TACC/Vista) the scheduler's env vars are
    translated to the PyTorch standard; MASTER_ADDR/MASTER_PORT must be set by
    the submit script. Without any DDP env vars, falls back to a single
    process so the script also runs on one GPU for debugging.
    """
    if 'SLURM_PROCID' in os.environ:
        os.environ['RANK'] = os.environ['SLURM_PROCID']
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
        os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
    elif 'RANK' not in os.environ:
        print("Warning: DDP environment variables not found. Running as single process.")
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'

    dist.init_process_group(backend='nccl')

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)  # pin this process to one GPU

    print(f"[Rank {rank}] DDP setup complete. WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
    return rank, world_size, local_rank


def cleanup_ddp():
    """Destroy the DDP process group."""
    dist.destroy_process_group()


def log_message(rank, message):
    """Print (flushed) on the main process only."""
    if rank == 0:
        print(message, flush=True)


def save_log_entry(rank, log_file, entry):
    """Append a JSON line to the log file, main process only."""
    if rank == 0:
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')


# ----------------------------------------------------------------------------
# 2. LOSS & SIMILARITY FUNCTIONS
# ----------------------------------------------------------------------------

def compute_ccf_similarity_torch(flux1, flux2, norm=True, epsilon=1e-8):
    """Normalized cross-correlation peak between two preprocessed spectra.

    This is the *ground-truth* similarity used to mine and weight triplets:
    the CCF is computed via FFT (zero-padded to 2L-1), optionally normalized
    by sqrt(sum(f1^2) * sum(f2^2)) so the peak lies in [-1, 1], and the peak
    height is returned as a Python float.

    Args:
        flux1, flux2: 1D tensors of whitened, standardized flux
                      (the output channel of ``preprocess_for_model``).
        norm: Normalize by the auto-correlation energies.
        epsilon: Numerical-stability floor for the denominator.
    """
    try:
        flux1 = flux1.float()
        flux2 = flux2.float()

        L = flux1.shape[0]
        n_pad = 2 * L - 1

        # Cross-correlation via the frequency domain.
        fft1 = torch.fft.rfft(flux1, n=n_pad)
        fft2_conj = torch.conj(torch.fft.rfft(flux2, n=n_pad))
        ccf = torch.fft.irfft(fft1 * fft2_conj, n=n_pad).real

        if norm:
            denominator = torch.sqrt(torch.sum(flux1 ** 2) * torch.sum(flux2 ** 2))
            if denominator > epsilon:
                peak_height = torch.max(ccf / denominator)
            else:
                return 0.0  # one of the signals is all zeros
        else:
            peak_height = torch.max(ccf)

        return peak_height.item()

    except Exception as e:
        print(f"WARNING: torch CCF computation failed: {e}")
        return 0.0


def triplet_loss(anchor, positive, negative, sim_ap, sim_an):
    """Similarity-calibrated triplet loss on 32-dim embeddings.

    Rather than a fixed margin, the CCF ground truth sets *target distances*:
        * margin_loss: squared-distance(A, P) should equal (1 - sim_ap),
        * ratio_loss:  squared-distance(A, N) should equal
                       (1 - sim_ap) * sim_ap / sim_an,
    so pairs that are more similar in CCF space are pulled proportionally
    closer in embedding space.
    """
    dist_positive = F.mse_loss(anchor, positive, reduction='none').sum(dim=-1)
    dist_negative = F.mse_loss(anchor, negative, reduction='none').sum(dim=-1)

    sim_ratio = torch.clamp(sim_ap, min=0.0) / torch.clamp(sim_an, min=1e-2)
    target_dist_one = torch.ones_like(dist_positive)
    margin_loss = F.mse_loss(dist_positive, target_dist_one - sim_ap, reduction='none')
    ratio_loss = F.mse_loss(dist_negative, (target_dist_one - sim_ap) * sim_ratio,
                            reduction='none')

    return (ratio_loss + margin_loss).mean()


# ----------------------------------------------------------------------------
# 3. HDF5 DATASET WITH HARD-TRIPLET MINING
# ----------------------------------------------------------------------------

class SpectrumTripletDataset(Dataset):
    """Loads spectra + AION embeddings from HDF5 and mines hard triplets.

    Each ``__getitem__`` samples three spectra, computes all three pairwise
    CCF similarities on the preprocessed flux, and assigns (A, P, N) as the
    valid triplet (sim_ap > sim_an) with the *smallest* similarity gap --
    i.e. the hardest distinction the model must learn. This prevents the
    embedding space from collapsing neighboring objects together.

    Modes:
        'hard'    -- always pick the hardest valid triplet.
        'relaxed' -- hardest 75% of the time, a random valid triplet otherwise.
    """

    def __init__(self, h5_path, aion_path, indices, hdf5_keys, mode='hard'):
        """
        Args:
            h5_path: HDF5 file with the raw spectra.
            aion_path: HDF5 file with the precomputed AION-1 embeddings.
            indices: Indices this split is allowed to sample from.
            hdf5_keys: Key map, e.g.
                {"flux": "FLUX", "ivar": "IVAR", "mask": "MASK",
                 "aion": "raw_aion"}
            mode: Triplet mining mode ('hard' or 'relaxed').
        """
        self.h5_path = h5_path
        self.aion_path = aion_path
        self.indices = indices
        self.num_samples = len(self.indices)
        self.keys = hdf5_keys
        self.mode = mode

        # File handles are opened lazily *per DataLoader worker* -- h5py
        # handles cannot be shared across forked processes.
        self.h5_file = None
        self.aion_file = None

        with h5py.File(self.h5_path, 'r') as f:
            self.total_dataset_size = len(f[self.keys['flux']])

    def _open_h5(self):
        """Open HDF5 handles for the current worker (with a larger chunk cache)."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', libver='latest',
                                     rdcc_nbytes=64 * 1024 ** 2, rdcc_nslots=250)
        if self.aion_file is None:
            self.aion_file = h5py.File(self.aion_path, 'r', libver='latest',
                                       rdcc_nbytes=64 * 1024 ** 2, rdcc_nslots=250)

    def _load_spectrum(self, idx):
        """Load one object: (AION embedding sequence, preprocessed spectrum)."""
        self._open_h5()

        aion_embed = self.aion_file[self.keys['aion']][idx]
        aion_embed = np.nan_to_num(aion_embed, nan=0.0, posinf=0.0, neginf=0.0)

        # Gaussian whitening + standardization (shared repo-wide pipeline).
        context = preprocess_for_model(
            flux=self.h5_file[self.keys['flux']][idx],
            ivar=self.h5_file[self.keys['ivar']][idx],
            mask=self.h5_file[self.keys['mask']][idx],
        )  # (1, L)

        return aion_embed, context

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # One index from the sampler, two more sampled from this split.
        idx_1 = self.indices[idx]
        idx_2, idx_3 = np.random.choice(self.indices, 2, replace=False)
        while idx_2 == idx_1:
            idx_2 = np.random.choice(self.indices)
        while idx_3 == idx_1 or idx_3 == idx_2:
            idx_3 = np.random.choice(self.indices)

        (aion_1, ctx_1) = self._load_spectrum(idx_1)
        (aion_2, ctx_2) = self._load_spectrum(idx_2)
        (aion_3, ctx_3) = self._load_spectrum(idx_3)

        # Ground-truth pairwise similarities on the whitened flux channel.
        sim_12 = compute_ccf_similarity_torch(torch.from_numpy(ctx_1[0]), torch.from_numpy(ctx_2[0]))
        sim_13 = compute_ccf_similarity_torch(torch.from_numpy(ctx_1[0]), torch.from_numpy(ctx_3[0]))
        sim_23 = compute_ccf_similarity_torch(torch.from_numpy(ctx_2[0]), torch.from_numpy(ctx_3[0]))

        specs = {
            1: (aion_1, ctx_1),
            2: (aion_2, ctx_2),
            3: (aion_3, ctx_3),
        }
        similarities = {
            (1, 2): sim_12, (2, 1): sim_12,
            (1, 3): sim_13, (3, 1): sim_13,
            (2, 3): sim_23, (3, 2): sim_23,
        }

        # Every (A, P, N) permutation scored by its similarity gap
        # sim(A,P) - sim(A,N); valid triplets have a positive gap, and the
        # hardest is the one with the smallest positive gap.
        scores = {
            (1, 2, 3): sim_12 - sim_13,
            (1, 3, 2): sim_13 - sim_12,
            (2, 1, 3): sim_12 - sim_23,
            (2, 3, 1): sim_23 - sim_12,
            (3, 1, 2): sim_13 - sim_23,
            (3, 2, 1): sim_23 - sim_13,
        }
        positive_scores = {t: s for t, s in scores.items() if s > 0}

        if self.mode == 'hard':
            a_idx, p_idx, n_idx = min(positive_scores, key=positive_scores.get)
        elif self.mode == 'relaxed':
            if random.random() < 0.75:
                a_idx, p_idx, n_idx = min(positive_scores, key=positive_scores.get)
            else:
                a_idx, p_idx, n_idx = random.choice(list(positive_scores.keys()))

        sim_ap = similarities[(a_idx, p_idx)]
        sim_an = similarities[(a_idx, n_idx)]

        qry_a, ctx_a = specs[a_idx]
        qry_p, ctx_p = specs[p_idx]
        qry_n, ctx_n = specs[n_idx]

        qry_batch = torch.from_numpy(np.stack([qry_a, qry_p, qry_n])).float()
        context_batch = torch.from_numpy(np.stack([ctx_a, ctx_p, ctx_n])).float()

        return qry_batch, context_batch, (sim_ap, sim_an)
