"""Distributed (DDP) training script for the Plato teacher model.

Trains Plato with the CCF-supervised triplet loss on hard-mined triplets of
(AION embedding sequence, preprocessed spectrum) pairs. Designed for
SLURM + multi-GPU via torch.distributed, but also runs single-GPU.

Usage: edit the Config paths/hyperparameters below, then launch, e.g.
    srun python plato/Plato_train.py          (SLURM, env vars set by scheduler)
    python plato/Plato_train.py               (single-GPU debug run)
"""

import os
import time
import random

import numpy as np
import h5py
import torch
import torch.distributed as dist
import torch.amp as amp
import torch._dynamo
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from Plato import Plato, create_plato_optimizer
from Plato_training_helpers import (
    setup_ddp, cleanup_ddp, log_message, save_log_entry,
    SpectrumTripletDataset, triplet_loss,
)

# Dynamo/compile is disabled: the interleaved attention loop did not benefit
# and graph breaks slowed training on the target cluster.
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
    aion_path = "embeddings.h5"               # <-- UPDATE: AION embeddings HDF5
    aion_key = "raw_aion"

    # --- Training ---
    epochs = 100
    batch_size = 32        # per GPU
    num_workers = 8        # DataLoader workers per GPU
    seed = 32
    mining_mode = 'hard'   # 'hard' = hardest triplet always; 'relaxed' = 75% of the time

    # --- Optimizer / scheduler ---
    lr_muon = 1e-4
    lr_adam = 1e-4
    wd = 0.0
    use_scheduler = True
    scheduler_type = 'ReduceLROnPlateau'
    scheduler_patience = 7    # epochs without val improvement before LR drop
    scheduler_factor = 0.1    # LR multiplier on plateau (1e-4 -> 1e-5)
    scheduler_min_lr = 1e-5

    # --- Checkpointing ---
    checkpoint_dir = "./plato_checkpoints"
    resume = False            # True: resume from latest_checkpoint.pth
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

    if rank == 0:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    log_file_path = os.path.join(config.checkpoint_dir, "training_log.jsonl")
    best_ckpt_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    latest_ckpt_path = os.path.join(config.checkpoint_dir, "latest_checkpoint.pth")

    # 2. --- Data setup: 90/10 split under the fixed seed ---
    log_message(rank, f"Loading data from {config.h5_path}...")
    try:
        with h5py.File(config.h5_path, 'r') as f:
            total_size = len(f[config.hdf5_flux_key])
    except Exception as e:
        print(f"ERROR [Rank {rank}]: Could not open or read HDF5 file: {config.h5_path}")
        print(f"Original Error: {e}")
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
        "aion": config.aion_key,
    }

    train_dataset = SpectrumTripletDataset(config.h5_path, config.aion_path,
                                           train_indices, hdf5_keys,
                                           mode=config.mining_mode)
    val_dataset = SpectrumTripletDataset(config.h5_path, config.aion_path,
                                         val_indices, hdf5_keys, mode='hard')

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=rank, shuffle=True, seed=config.seed)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size,
                                     rank=rank, shuffle=False)

    loader_kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                         pin_memory=True, prefetch_factor=4,
                         worker_init_fn=worker_init_seed)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

    log_message(rank, f"Data loaded. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # 3. --- Model ---
    log_message(rank, "Initializing Plato model...")
    model = Plato(
        n_dim=768,
        n_layers=12,
        n_self_attn_heads=12,
        self_attn_hidden=3072,
        n_cross_attn_heads=12,
        cross_attn_hidden=3072,
        n_registers=4,
    ).to(device)

    # 4. --- Resume: load model state BEFORE wrapping in DDP ---
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    scheduler = None
    checkpoint = None

    if config.resume and os.path.exists(latest_ckpt_path):
        log_message(rank, f"Resuming training from {latest_ckpt_path}...")
        map_location = {'cuda:0': f'cuda:{local_rank}'}
        checkpoint = torch.load(latest_ckpt_path, map_location=map_location,
                                weights_only=False)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            log_message(rank, "Successfully loaded model state_dict.")
        except Exception as e:
            log_message(rank, f"CRITICAL: Failed to load model state_dict: {e}. "
                              "Checkpoint may be incompatible.")
            raise e

        start_epoch = checkpoint.get('epoch', -1) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        log_message(rank, f"Resumed from epoch {start_epoch - 1}. "
                          f"Best val loss so far: {best_val_loss:.4f}")

    # --- Wrap in DDP, then build the optimizer over the wrapped model ---
    model = DDP(model, device_ids=[local_rank])

    log_message(rank, "Initializing Muon optimizer...")
    optimizer = create_plato_optimizer(
        model, lr_muon=config.lr_muon, lr_adam=config.lr_adam, wd=config.wd,
    )
    if config.scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
        )

    # --- Resume: optimizer / scheduler / scaler state (checkpoint already loaded) ---
    if checkpoint is not None:
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                log_message(rank, "Loaded LR scheduler state.")
            except Exception as e:
                log_message(rank, f"Warning: Could not load scheduler state: {e}. "
                                  "Starting scheduler from scratch.")
        if config.use_amp and checkpoint.get('scaler_state_dict') is not None:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                log_message(rank, "Loaded GradScaler state.")
            except Exception as e:
                log_message(rank, f"Warning: Could not load GradScaler state: {e}. "
                                  "Initializing fresh scaler.")
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            log_message(rank, "Loaded optimizer state.")
            # Re-apply LRs from config in case they were changed for this run.
            for param_group in optimizer.param_groups:
                if param_group.get('use_muon') is True:
                    param_group['lr'] = config.lr_muon
                elif param_group.get('use_muon') is False:
                    param_group['lr'] = config.lr_adam
        except Exception as e:
            log_message(rank, f"Warning: Could not load optimizer state: {e}. "
                              "Starting optimizer from scratch.")

    dist.barrier()  # all ranks finish loading before training starts

    # 5. --- Training loop ---
    log_message(rank, f"Starting training from epoch {start_epoch}...")

    for epoch in range(start_epoch, config.epochs):
        train_sampler.set_epoch(epoch)  # reshuffle DDP shards each epoch

        # --- Train epoch ---
        model.train()
        train_loss_accumulator = torch.zeros(1).to(device)
        train_start_time = time.time()

        for batch_idx, (query, context, sim) in enumerate(train_loader):
            query = query.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            sim_ap = sim[0].to(device, non_blocking=True).float()
            sim_an = sim[1].to(device, non_blocking=True).float()

            # (B, 3, ...) triplets -> anchor / positive / negative
            qry_a, qry_p, qry_n = query.unbind(1)
            ctx_a, ctx_p, ctx_n = context.unbind(1)

            optimizer.zero_grad(set_to_none=True)

            # Single forward over the concatenated triplet (3B batch).
            all_queries = torch.cat([qry_a, qry_p, qry_n], dim=0)
            all_contexts = torch.cat([ctx_a, ctx_p, ctx_n], dim=0)
            with amp.autocast('cuda', enabled=config.use_amp):
                all_embeddings = model(all_queries, all_contexts)
                check_for_nan(all_embeddings, "train all_embeddings (forward pass)", rank)
                emb_a, emb_p, emb_n = torch.chunk(all_embeddings, 3, dim=0)
                loss = triplet_loss(emb_a, emb_p, emb_n, sim_ap, sim_an)
                check_for_nan(loss, "train loss", rank)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accumulator += loss.detach()

            if rank == 0 and (batch_idx + 1) % 150 == 0:
                current_avg_loss = (train_loss_accumulator / (batch_idx + 1)).item()
                print(f"  Epoch {epoch} Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Train Loss: {current_avg_loss:.4f}", end='\r', flush=True)

        # --- Validation epoch ---
        model.eval()
        val_loss_accumulator = torch.zeros(1).to(device)
        with torch.no_grad():
            for query, context, sim in val_loader:
                query = query.to(device, non_blocking=True)
                context = context.to(device, non_blocking=True)
                sim_ap = sim[0].to(device, non_blocking=True).float()
                sim_an = sim[1].to(device, non_blocking=True).float()

                qry_a, qry_p, qry_n = query.unbind(1)
                ctx_a, ctx_p, ctx_n = context.unbind(1)

                all_queries = torch.cat([qry_a, qry_p, qry_n], dim=0)
                all_contexts = torch.cat([ctx_a, ctx_p, ctx_n], dim=0)
                with amp.autocast('cuda', enabled=config.use_amp):
                    all_embeddings = model(all_queries, all_contexts)
                    check_for_nan(all_embeddings, "val all_embeddings (forward pass)", rank)
                    emb_a, emb_p, emb_n = torch.chunk(all_embeddings, 3, dim=0)
                    loss = triplet_loss(emb_a, emb_p, emb_n, sim_ap, sim_an)
                    check_for_nan(loss, "val loss", rank)

                val_loss_accumulator += loss.detach()

        # --- Aggregate across ranks, step scheduler, log, checkpoint ---
        train_loss_accumulator /= len(train_loader)
        val_loss_accumulator /= len(val_loader)
        dist.all_reduce(train_loss_accumulator, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_accumulator, op=dist.ReduceOp.SUM)

        avg_train_loss = (train_loss_accumulator / world_size).item()
        avg_val_loss = (val_loss_accumulator / world_size).item()

        # Every rank now holds the same reduced value, so all can step.
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
                elif pg.get('use_muon') is False:
                    current_lr_adam = pg['lr']

            log_entry = {
                "epoch": epoch,
                "train_time_sec": round(train_time, 2),
                "train_loss_total": round(avg_train_loss, 5),
                "val_loss_total": round(avg_val_loss, 5),
                "lr_muon": current_lr_muon,
                "lr_adam": current_lr_adam,
            }
            save_log_entry(rank, log_file_path, log_entry)

            print()  # clear the \r progress line
            log_message(rank,
                        f"Epoch {epoch} | Train Total Loss: {avg_train_loss:.6f} | "
                        f"Val Total Loss: {avg_val_loss:.6f} | Time: {train_time:.2f}s")

            # Weight-magnitude diagnostics: catches silent optimizer failures
            # (e.g. Muon leaving the ConvNet untouched).
            conv_weights = model.module.ConvNet.ConvNet[0].weight.abs().mean().item()
            cross_attn_weights = model.module.fusion_layers[0][
                'cross_attn'].cross_attn.in_proj_weight.abs().mean().item()
            log_message(rank, f"  -> DIAGNOSTIC: ConvNet W_mean: {conv_weights:.5e} | "
                              f"CrossAttn W_mean: {cross_attn_weights:.5e}")

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
                log_message(rank, f"  -> New best val loss: {best_val_loss:.4f}. "
                                  f"Saving best model to {best_ckpt_path}")
                torch.save(model.module.state_dict(), best_ckpt_path)

        dist.barrier()  # rank 0 finishes saving before the next epoch

    log_message(rank, "Training complete.")
    cleanup_ddp()


# ----------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    config = Config()
    try:
        main(config)
    except Exception as e:
        rank = int(os.environ.get('RANK', 0))
        if rank == 0:
            print(f"\nFATAL ERROR during training: {e}")
            import traceback
            traceback.print_exc()
        if dist.is_initialized():
            cleanup_ddp()
        raise e  # re-raise so the scheduler sees the failure
