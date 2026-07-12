"""Training helpers for Aristotle distillation: DDP utilities and the
raw-spectrum + Plato-embedding dataset.

Note that unlike Plato's dataset, ``DistillationDataset`` returns *raw*
(flux, ivar, valid_flag) channels: the Gaussian whitening runs batched on the
GPU (``preprocessing.GPUPreprocessor``) because CPU-side preprocessing was
the throughput bottleneck at distillation batch sizes.
"""

import os
import json

import numpy as np
import h5py
import torch
import torch.distributed as dist
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------
# DDP & LOGGING UTILITIES
# ----------------------------------------------------------------------------

def setup_ddp():
    """Initialize the DDP process group (SLURM-aware).

    Translates SLURM env vars to the PyTorch standard where present
    (MASTER_ADDR/MASTER_PORT must come from the submit script); otherwise
    falls back to a single process for one-GPU debug runs.
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
# DISTILLATION DATASET
# ----------------------------------------------------------------------------

class DistillationDataset(Dataset):
    """Pairs raw spectra with precomputed Plato embeddings for distillation.

    Returns per item:
        plato_embed: (32,) the teacher's embedding (the regression target).
        context:     (3, L) raw channels [flux, ivar, valid_flag], where
                     valid_flag = (mask == 0). Preprocessing (Gaussian
                     whitening + standardization) is deliberately NOT done
                     here -- it happens on the GPU in the training loop via
                     ``preprocessing.GPUPreprocessor``.
    """

    def __init__(self, h5_path, plato_path, indices, hdf5_keys):
        """
        Args:
            h5_path: HDF5 file with the raw spectra.
            plato_path: HDF5 file with Plato's precomputed embeddings.
            indices: Indices this split is allowed to sample from.
            hdf5_keys: Key map, e.g.
                {"flux": "FLUX", "ivar": "IVAR", "mask": "MASK",
                 "plato": "plato_deep_1c"}
        """
        self.h5_path = h5_path
        self.plato_path = plato_path
        self.indices = indices
        self.num_samples = len(self.indices)
        self.keys = hdf5_keys

        # File handles are opened lazily per DataLoader worker.
        self.h5_file = None
        self.plato_file = None

        with h5py.File(self.h5_path, 'r') as f:
            self.total_dataset_size = len(f[self.keys['flux']])

    def _open_h5(self):
        """Open HDF5 handles for the current worker (with a larger chunk cache)."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', libver='latest',
                                     rdcc_nbytes=64 * 1024 ** 2, rdcc_nslots=250)
        if self.plato_file is None:
            self.plato_file = h5py.File(self.plato_path, 'r', libver='latest',
                                        rdcc_nbytes=64 * 1024 ** 2, rdcc_nslots=250)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = self.indices[idx]
        self._open_h5()

        # Teacher embedding (regression target).
        plato_embed = self.plato_file[self.keys['plato']][idx]
        plato_embed = np.nan_to_num(plato_embed, nan=0.0, posinf=0.0, neginf=0.0)

        # Raw spectrum channels; whitening happens later on the GPU.
        flux = np.nan_to_num(self.h5_file[self.keys['flux']][idx],
                             nan=0.0, posinf=0.0, neginf=0.0)
        ivar = np.nan_to_num(self.h5_file[self.keys['ivar']][idx],
                             nan=0.0, posinf=0.0, neginf=0.0)
        mask = self.h5_file[self.keys['mask']][idx]
        mask_bool = (mask == 0)  # convert to a validity flag (1 = good pixel)

        context = np.stack([flux, ivar, mask_bool], axis=0)  # (3, L)

        return plato_embed, context
