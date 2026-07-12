"""Distributed (DDP) distillation script: Plato -> Aristotle.

Trains the spectrum-only Aristotle student to reproduce Plato's 32-dim
embeddings via MSE. Raw (flux, ivar, valid_flag) batches are shipped to the
GPU and whitened there with ``preprocessing.GPUPreprocessor``.

Prerequisites:
    * a Plato checkpoint named 'plato.pth' somewhere under the working
      directory (Aristotle auto-loads and freezes its projection head), and
    * an HDF5 file of Plato embeddings for the training catalog.

Usage: edit the Config below, then launch, e.g.
    srun python aristotle/Aristotle_train.py   (SLURM)
    python aristotle/Aristotle_train.py        (single-GPU debug run)
"""

import os
import sys
import time
import random

import numpy as np
import h5py
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.amp as amp
import torch._dynamo
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from Aristotle import Aristotle, create_aristotle_optimizer
from Aristotle_training_helpers import (
    setup_ddp, cleanup_ddp, log_message, save_log_entry, DistillationDataset,
)

# Repo root on sys.path so the shared preprocessing module resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import GPUPreprocessor  # noqa: E402

torch._dynamo.disable()


# ----------------------------------------------------------------------------
# CONFIGURATION -- edit these values directly before running
# ----------------------------------------------------------------------------

class Config:
    torch.autograd.set_detect_anomaly(False)  # enable only when debugging NaNs

    # --- Data ---
    h5_path = "merged_filtered_sample.hdf5"   # <-- UPDATE: spectra HDF5
    hdf5_flux_key = "FLUX"
    hdf5_ivar_key = "IVAR"
    hdf5_mask_key = "MASK"
    plato_path = "embeddings.h5"              # <-- UPDATE: Plato embeddings HDF5
    plato_key = "plato_deep_1c"

    # --- Training ---
    epochs = 200
    batch_size = 256   # per GPU (larger than Plato's -- no triplets, no CCF)
    num_workers = 8
    seed = 32

    # --- Optimizer / scheduler ---
    lr_muon = 1e-4
    lr_adam = 1e-4
    wd = 0.0
    use_scheduler = True
    scheduler_type = 'ReduceLROnPlateau'
    scheduler_patience = 7
    scheduler_factor = 0.1
    scheduler_min_lr = 1e-6

    # --- Checkpointing ---
    checkpoint_dir = "./aristotle_checkpoints"
    resume = False
    use_amp = False


# ----------------------------------------------------------------------------
# SEEDING & SANITY HELPERS
# ----------------------------------------------------------------------------

def set_seed(seed):
    """Seed python/numpy/torch (all GPUs) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def worker_init_seed(worker_id):
    """Give each DataLoader worker (per rank) a distinct, deterministic seed."""
    rank = int(os.environ.get('RANK', 0))
    seed = Config.seed + worker_id + rank * Config.num_workers
    np.random.seed(seed)
    random.seed(seed)


def check_for_nan(tensor, name, rank):
    """Hard-stop training on all ranks if a NaN sneaks into `tensor`."""
    if torch.isnan(tensor).any():
        message = f"!!! FATAL [Rank {rank}]: NaN detected in tensor '{name}' !!!"
        print(message, flush=True)
        raise ValueError(message)


# ----------------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# ----------------------------------------------------------------------------

def main(config):
    set_seed(config.seed)

    # 1. --- DDP setup ---
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    scaler = amp.GradScaler('cuda', enabled=config.use_amp)

    # 2. --- Paths & logging ---
    if rank == 0:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    log_file_path = os.path.join(config.checkpoint_dir, "training_log.jsonl")
    best_ckpt_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    latest_ckpt_path = os.path.join(config.checkpoint_dir, "latest_model.pth")

    log_message(rank, f"Loading data from {config.h5_path}")

    # 3. --- Data: 90/10 split under the fixed seed ---
    try:
        with h5py.File(config.h5_path, 'r') as f:
            total_size = len(f[config.hdf5_flux_key])
    except Exception as e:
        print(f"ERROR [Rank {rank}]: Could not open HDF5: {config.h5_path} ({e})")
        if dist.is_initialized():
            dist.barrier()
        exit(1)

    all_indices = np.arange(total_size)
    np.random.shuffle(all_indices)
    split_idx = int(total_size * 0.9)
    train_indices = all_indices[:split_idx]
    val_indices = all_indices[split_idx:]

    hdf5_keys = {
        "flux": config.hdf5_flux_key,
        "ivar": config.hdf5_ivar_key,
        "mask": config.hdf5_mask_key,
        "plato": config.plato_key,
    }

    train_dataset = DistillationDataset(config.h5_path, config.plato_path,
                                        train_indices, hdf5_keys)
    val_dataset = DistillationDataset(config.h5_path, config.plato_path,
                                      val_indices, hdf5_keys)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=rank, shuffle=True, seed=config.seed)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size,
                                     rank=rank, shuffle=False)

    loader_kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                         pin_memory=True, prefetch_factor=2,
                         worker_init_fn=worker_init_seed, persistent_workers=True)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

    log_message(rank, f"Data loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 4. --- Model & GPU preprocessor ---
    log_message(rank, "Initializing Aristotle")
    preprocessor = GPUPreprocessor(device=device).to(device)
    model = Aristotle().to(device)  # auto-loads & freezes the Plato head

    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    scheduler = None
    checkpoint = None

    # --- Resume: load model state BEFORE wrapping in DDP ---
    if config.resume and os.path.exists(latest_ckpt_path):
        log_message(rank, f"Resuming from {latest_ckpt_path}")
        map_location = {'cuda:0': f'cuda:{local_rank}'}
        checkpoint = torch.load(latest_ckpt_path, map_location=map_location,
                                weights_only=False)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            log_message(rank, "Successfully loaded model state")
        except Exception as e:
            log_message(rank, f"Failed to load model state: {e}")
            raise e

        start_epoch = checkpoint.get('epoch', -1) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        log_message(rank, f"Resumed from epoch {start_epoch}. Best val loss: {best_val_loss}")
    elif config.resume:
        log_message(rank, "Failed to resume (no checkpoint found). Restarting from epoch 0")

    model = DDP(model, device_ids=[local_rank])

    optimizer = create_aristotle_optimizer(
        model, lr_muon=config.lr_muon, lr_adam=config.lr_adam, wd=config.wd,
    )

    if config.scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
        )

    # --- Resume: optimizer / scheduler / scaler state ---
    if checkpoint is not None:
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception:
                log_message(rank, "Could not load scheduler state")
        if config.use_amp and checkpoint.get('scaler_state_dict') is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Re-apply LRs from config in case they were changed for this run.
            for param_group in optimizer.param_groups:
                if param_group.get('use_muon') is True:
                    param_group['lr'] = config.lr_muon
                else:
                    param_group['lr'] = config.lr_adam
        except Exception:
            log_message(rank, "Failed to load optimizer state")

    dist.barrier()

    # 5. --- Training loop ---
    for epoch in range(start_epoch, config.epochs):
        train_sampler.set_epoch(epoch)

        # --- Train epoch ---
        model.train()
        train_loss_accumulator = torch.zeros(1).to(device)
        train_start_time = time.time()

        for batch_idx, (plato_embed, raw_context) in enumerate(train_loader):
            plato_embed = plato_embed.to(device, non_blocking=True)
            raw_context = raw_context.to(device, non_blocking=True)  # (B, 3, L)

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast('cuda', enabled=config.use_amp):
                # Whitening is a fixed transform -- no gradients needed.
                with torch.no_grad():
                    clean_context = preprocessor(raw_context)  # (B, 1, L)

                aristotle_embed = model(clean_context)
                loss = F.mse_loss(aristotle_embed, plato_embed)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accumulator += loss.detach()

            if rank == 0 and (batch_idx + 1) % 100 == 0:
                current_avg_loss = (train_loss_accumulator / (batch_idx + 1)).item()
                print(f"Epoch {epoch} Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Train Loss: {current_avg_loss:0.5f}", end='\r', flush=True)

        # --- Validation epoch ---
        model.eval()
        val_loss_accumulator = torch.zeros(1).to(device)
        with torch.no_grad():
            for plato_embed, raw_context in val_loader:
                plato_embed = plato_embed.to(device, non_blocking=True)
                raw_context = raw_context.to(device, non_blocking=True)

                with amp.autocast('cuda', enabled=config.use_amp):
                    clean_context = preprocessor(raw_context)
                    aristotle_embed = model(clean_context)
                    loss = F.mse_loss(aristotle_embed, plato_embed)

                val_loss_accumulator += loss.detach()

        # --- Aggregate across ranks ---
        avg_train_loss_local = train_loss_accumulator / len(train_loader)
        avg_val_loss_local = val_loss_accumulator / len(val_loader)
        dist.all_reduce(avg_train_loss_local, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_val_loss_local, op=dist.ReduceOp.SUM)

        avg_train_loss = (avg_train_loss_local / world_size).item()
        avg_val_loss = (avg_val_loss_local / world_size).item()

        # Every rank holds the same reduced value, so all can step.
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        if rank == 0:
            train_time = time.time() - train_start_time
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            current_lr_muon = config.lr_muon
            current_lr_adam = config.lr_adam
            for pg in optimizer.param_groups:
                if pg.get('use_muon') is True:
                    current_lr_muon = pg['lr']
                else:
                    current_lr_adam = pg['lr']

            log_entry = {
                'epoch': epoch,
                'train_time_sec': round(train_time, 2),
                'train_loss_total': round(avg_train_loss, 6),
                'val_loss_total': round(avg_val_loss, 6),
                'lr_muon': current_lr_muon,
                'lr_adam': current_lr_adam,
            }
            save_log_entry(rank, log_file_path, log_entry)

            print()  # clear the \r progress line
            log_message(rank, f"Epoch {epoch} | Train Loss: {avg_train_loss:.5f} | "
                              f"Val Loss: {avg_val_loss:.5f} | Time: {train_time:.2f}s")

            # --- Checkpointing ---
            latest_state = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if config.use_amp else None,
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': vars(config),
            }
            torch.save(latest_state, latest_ckpt_path)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                log_message(rank, f"New best val loss: {best_val_loss:.6f}")
                torch.save(model.module.state_dict(), best_ckpt_path)

        dist.barrier()

    cleanup_ddp()


if __name__ == '__main__':
    config = Config()
    try:
        main(config)
    except Exception as e:
        rank = int(os.environ.get('RANK', 0))
        if rank == 0:
            print(f'FATAL ERROR during training: {e}')
            import traceback
            traceback.print_exc()
        if dist.is_initialized():
            cleanup_ddp()
        raise e  # re-raise so SLURM sees the failure
