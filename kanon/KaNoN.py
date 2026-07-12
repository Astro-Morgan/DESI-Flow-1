"""KaNoN: conditional normalizing flows over Aristotle/Plato embeddings.

KaNoN models two densities on the 32-dim embedding space (phi):

    * Manifold flow  p(phi)      -- an unconditional neural spline flow used
                                    as an out-of-distribution (OOD) score.
    * Physics flow   p(z | phi)  -- a conditional neural spline flow over the
                                    *residual* between the true redshift and a
                                    k-NN anchor prediction, scaled by a
                                    z-binned residual sigma. Gives full
                                    redshift posteriors, not point estimates.

The k-NN anchor (``KNNWrapper``) is a distance-weighted k-NN regressor
imported from a fitted sklearn model and stored as GPU-ready buffers, so
anchoring runs natively inside the model at inference time.
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import zuko
from zuko.transforms import MonotonicRQSTransform


class KNNWrapper(nn.Module):
    """Distance-weighted k-NN regression as a torch module.

    Training data are stored as buffers (saved in the .pth, untouched by the
    optimizer). ``forward`` is mathematically identical to sklearn's
    KNeighborsRegressor with ``weights='distance'``, but runs on GPU.
    """

    def __init__(self, k=10):
        super().__init__()
        self.k = k
        self.register_buffer('X_train', torch.empty(0))
        self.register_buffer('y_train', torch.empty(0))
        # k is also persisted as a buffer so checkpoints restore it
        # (torch.topk needs a Python int, hence the mirrored attribute).
        self.register_buffer('_k', torch.tensor(k))
        self.fitted = False

    def load_from_sklearn(self, sklearn_model):
        """Copy the anchor set (X, y) out of a fitted sklearn KNeighborsRegressor."""
        if not hasattr(sklearn_model, '_fit_X'):
            raise ValueError("Sklearn model is not fitted!")

        # sklearn stores training data in _fit_X and _y.
        X_np = sklearn_model._fit_X.astype(np.float32)
        y_np = sklearn_model._y.astype(np.float32)
        if y_np.ndim == 1:
            y_np = y_np[:, None]

        self.X_train = torch.from_numpy(X_np)
        self.y_train = torch.from_numpy(y_np)
        self.k = sklearn_model.n_neighbors
        self._k = torch.tensor(self.k)
        self.fitted = True

        print(f"   [KNN] Imported {len(X_np)} anchor points (k={self.k})")

    def forward(self, query_phi):
        """(B, D) query embeddings -> (B, 1) distance-weighted k-NN prediction."""
        if not self.fitted:
            # Before loading (e.g. during a bare init) return zeros to avoid a crash.
            return torch.zeros(query_phi.shape[0], 1, device=query_phi.device)

        dists = torch.cdist(query_phi, self.X_train)
        topk_dists, topk_idxs = dists.topk(self.k, dim=1, largest=False)

        # 1/d weighting, normalized (epsilon prevents division by zero).
        weights = 1.0 / (topk_dists + 1e-9)
        weights = weights / weights.sum(dim=1, keepdim=True)

        neighbor_y = self.y_train[topk_idxs]  # (B, K, 1)
        if neighbor_y.ndim == 2:
            neighbor_y = neighbor_y.unsqueeze(-1)

        return torch.sum(weights.unsqueeze(-1) * neighbor_y, dim=1)


class ResBlock(nn.Module):
    """Dimension-preserving residual MLP block used by the context encoder."""

    def __init__(self, dim: int = 96, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class ResNet(nn.Module):
    """Residual MLP: phi -> conditioning context for the physics flow.

    The head is initialized near zero so early conditioning is weak and the
    flow starts close to its unconditional behavior.
    """

    def __init__(self, feature_dim: int = 32, hidden_dim: int = 256,
                 out_dim: int = 256, n_blocks: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        self.body = nn.ModuleList([
            ResBlock(dim=feature_dim, hidden_dim=hidden_dim) for _ in range(n_blocks)
        ])
        self.head = nn.Linear(feature_dim, out_dim)
        torch.nn.init.normal_(self.head.weight, 0.0, 1e-4)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.stem(x)
        for block in self.body:
            x = block(x)
        return self.head(x)


class EmbeddingFlow(nn.Module):
    """Unconditional neural spline flow modeling p(phi) (the manifold density)."""

    def __init__(self, phi_dim=32, n_layers=5, hidden_dim=256, n_bins=16):
        super().__init__()
        self.flow = zuko.flows.NSF(
            features=phi_dim,
            context=0,
            transforms=n_layers,
            bins=n_bins,
            hidden_features=[hidden_dim, hidden_dim],
            randperm=True,
        )
        # Widen the spline bounding box from zuko's default to [-10, 10] so
        # standardized embeddings in the tails stay inside the spline support.
        for layer in self.flow.transform.transforms:
            layer.univariate = partial(MonotonicRQSTransform, slope=1e-3, bound=10.0)

    def log_prob(self, phi):
        return self.flow().log_prob(phi)

    def sample(self, num_samples=1):
        return self.flow().sample((num_samples,))


class RedshiftFlow(nn.Module):
    """Conditional flow p(scaled residual | phi): 1D NSF conditioned via ResNet(phi)."""

    def __init__(self, phi_dim=32, hidden_dim=256, n_bins=32, n_layers=5):
        super().__init__()
        self.encoder = ResNet(
            feature_dim=phi_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_blocks=4,
        )
        self.flow = zuko.flows.NSF(
            features=1,
            context=hidden_dim,
            transforms=n_layers,
            bins=n_bins,
            hidden_features=[hidden_dim, hidden_dim],
        )
        # Widen the spline bounding box to [-10, 10] (matches EmbeddingFlow).
        for layer in self.flow.transform.transforms:
            layer.univariate = partial(MonotonicRQSTransform, slope=1e-3, bound=10.0)

    def log_prob(self, z, phi):
        context = self.encoder(phi)
        if z.ndim == 1:
            z = z.unsqueeze(-1)
        return self.flow(context).log_prob(z)

    def sample(self, phi, num_samples=1):
        context = self.encoder(phi)
        samples = self.flow(context).sample((num_samples,))
        return samples.squeeze(-1).T


class KaNoN(nn.Module):
    """Joint density model over (phi, z) with a k-NN anchored residual parametrization.

    Rather than modeling p(z | phi) directly, the physics flow models the
    residual r = z - z_knn(phi), standardized by a redshift-binned sigma
    lookup computed at init from cross-validated anchor predictions. This
    keeps the flow's target roughly unit-scale everywhere along the redshift
    axis, which the spline flow handles far better than raw z.

    Args:
        init_data: (N, phi_dim + 1) array of [phi, z] used only to compute
            normalization statistics (phi mean/std, z min/max, residual bins).
        pretrained_knn: A *fitted* sklearn KNeighborsRegressor to import as
            the anchor (weights are copied; no inference happens here).
        precomputed_cv_anchors: Cross-validated anchor predictions for
            init_data, used to build the z-binned residual sigmas without
            train-set leakage. Falls back to dummy sigma=1 if omitted.
    """

    def __init__(self, init_data, phi_dim=32, hidden_dim=1024, n_bins=128,
                 n_layers_phi=10, n_layers_z=10, pretrained_knn=None,
                 k_neighbors=10, precomputed_cv_anchors=None):
        super().__init__()

        # --- Normalization statistics from the training data ---
        if isinstance(init_data, torch.Tensor):
            init_data = init_data.detach().cpu().numpy()

        phi_raw = init_data[:, :-1]
        z_raw = init_data[:, -1:]

        self.register_buffer('phi_mean', torch.from_numpy(phi_raw.mean(0)).float())
        self.register_buffer('phi_std', torch.from_numpy(phi_raw.std(0)).float() + 1e-6)
        self.register_buffer('z_min', torch.tensor(z_raw.min()).float())
        self.register_buffer('z_max', torch.tensor(z_raw.max()).float())

        # --- k-NN anchor ---
        self.anchor = KNNWrapper(k=k_neighbors)
        if pretrained_knn is not None:
            self.anchor.load_from_sklearn(pretrained_knn)

        # --- Z-binned residual sigma lookup ---
        print(">>> Computing Z-Binned Residual Statistics...")
        if precomputed_cv_anchors is not None:
            if isinstance(precomputed_cv_anchors, np.ndarray):
                cv_anchors = torch.from_numpy(precomputed_cv_anchors).float()
            else:
                cv_anchors = precomputed_cv_anchors.float()

            z_tensor = torch.from_numpy(z_raw).float()
            if cv_anchors.ndim == 1:
                cv_anchors = cv_anchors.unsqueeze(1)
            if z_tensor.ndim == 1:
                z_tensor = z_tensor.unsqueeze(1)

            # Residuals of the CV anchor (truth - out-of-fold prediction),
            # binned in z so sigma tracks how anchor quality varies with z.
            all_residuals = z_tensor - cv_anchors

            n_sigma_bins = 20
            z_bins = torch.linspace(z_raw.min(), z_raw.max(), steps=n_sigma_bins + 1)
            sigma_vals = []
            for i in range(n_sigma_bins):
                bin_mask = (z_tensor.squeeze() >= z_bins[i]) & (z_tensor.squeeze() < z_bins[i + 1])
                if bin_mask.sum() > 5:
                    sigma_vals.append(all_residuals[bin_mask].std())
                else:
                    sigma_vals.append(all_residuals.std())  # global fallback

            self.register_buffer('sigma_bins', torch.tensor(sigma_vals).float() + 1e-6)
            self.register_buffer('z_bin_edges', z_bins)
            print(f"    Computed {n_sigma_bins} residual bins using Precomputed CV.")
            print(f"    Min Sigma: {min(sigma_vals):.4f}, Max Sigma: {max(sigma_vals):.4f}")
        else:
            print(">>> WARNING: No precomputed CV anchors provided. Using dummy sigma=1.0.")
            self.register_buffer('sigma_bins', torch.ones(20))
            self.register_buffer('z_bin_edges', torch.linspace(0, 5, 21))

        # --- The two flows ---
        self.manifold_flow = EmbeddingFlow(phi_dim, n_layers_phi, hidden_dim, n_bins)
        self.physics = RedshiftFlow(phi_dim, hidden_dim, n_bins, n_layers_z)

        self._print_params()

    def _print_params(self):
        manifold_params = sum(p.numel() for p in self.manifold_flow.parameters() if p.requires_grad)
        physics_params = sum(p.numel() for p in self.physics.parameters() if p.requires_grad)
        print("-" * 40)
        print("KaNoN Model Initialized")
        print(f"   Manifold Flow (p(phi)):  {manifold_params:,} params")
        print(f"   Physics Flow (p(z|phi)): {physics_params:,} params")
        print(f"   TOTAL:                   {manifold_params + physics_params:,} params")
        print("-" * 40)

    # ------------------------------------------------------------------
    # Scaling utilities
    # ------------------------------------------------------------------

    def standardize_phi(self, phi):
        """Raw phi -> ~N(0, 1) per dimension (flow / encoder input space)."""
        return (phi - self.phi_mean) / self.phi_std

    def scale_z(self, z):
        """Raw z -> linear [-4.5, 4.5] (global bounds; residuals matter more)."""
        target = 4.5
        z_range = self.z_max - self.z_min + 1e-6
        z_01 = (z - self.z_min) / z_range
        return z_01 * (2 * target) - target

    def unscale_z(self, z_scaled):
        """Inverse of :meth:`scale_z`."""
        target = 4.5
        z_range = self.z_max - self.z_min
        z_01 = (z_scaled + target) / (2 * target)
        return z_01 * z_range + self.z_min

    def get_residual_std(self, z):
        """Look up the residual sigma for raw redshift(s) via the z-bin table."""
        # bucketize returns right-edge indices, hence the -1; clamp handles
        # values at/below the first edge and at/above the last.
        indices = torch.bucketize(z, self.z_bin_edges) - 1
        indices = torch.clamp(indices, 0, len(self.sigma_bins) - 1)

        if z.ndim == 1:
            return self.sigma_bins[indices]
        return self.sigma_bins[indices.squeeze()].view(-1, 1)

    def standardize_residual(self, raw_residual, z_context):
        """Scale a raw residual by the sigma of its anchor's z bin."""
        sigma = self.get_residual_std(z_context)
        return raw_residual / (sigma + 1e-6)

    def unstandardize_residual(self, scaled_residual, z_context):
        """Inverse of :meth:`standardize_residual`."""
        sigma = self.get_residual_std(z_context)
        return scaled_residual * (sigma + 1e-6)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, checkpoint_path, device='cpu'):
        """Rebuild a KaNoN from a bare state dict, inferring the architecture.

        Dimensions (phi_dim, hidden_dim, layer counts, spline bins) are
        inferred from tensor shapes/key names so the checkpoint alone is
        enough to reload the model. The k-NN anchor buffers are pre-allocated
        to the checkpointed shapes before ``load_state_dict``, since buffers
        cannot be resized during loading.
        """
        print(f"Loading KaNoN from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        # Strip a DDP 'module.' prefix if present.
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Infer embedding dimension.
        phi_dim = state_dict['phi_mean'].shape[0] if 'phi_mean' in state_dict else 32

        # Infer hidden dimension from the physics encoder's first ResBlock.
        hidden_dim = 256  # fallback
        for k in ['physics.encoder.body.0.net.0.weight_orig',
                  'physics.encoder.body.0.net.0.weight']:
            if k in state_dict:
                hidden_dim = state_dict[k].shape[0]
                break

        # Infer transform counts by scanning key prefixes.
        n_layers_phi = 0
        while any(k.startswith(f"manifold_flow.flow.transform.transforms.{n_layers_phi}.")
                  for k in state_dict.keys()):
            n_layers_phi += 1
        n_layers_z = 0
        while any(k.startswith(f"physics.flow.transform.transforms.{n_layers_z}.")
                  for k in state_dict.keys()):
            n_layers_z += 1

        # Infer spline bin count: zuko's 1D NSF hyper-net outputs 3*bins - 1.
        n_bins = 32
        prefix = "physics.flow.transform.transforms.0.hyper."
        for key, param in state_dict.items():
            if key.startswith(prefix) and (key.endswith(".weight") or key.endswith(".bias")):
                out_dim = param.shape[0]
                if (out_dim + 1) % 3 == 0:
                    candidate = (out_dim + 1) // 3
                    if candidate in [8, 16, 32, 64, 128, 256]:
                        n_bins = candidate
                        break

        print(f"   [Inferred] Phi: {phi_dim}, Hidden: {hidden_dim}, Bins: {n_bins}")
        print(f"   [Inferred] Layers: Manifold={n_layers_phi}, Physics={n_layers_z}")

        # Instantiate with dummy stats (all real stats live in the state dict).
        dummy_data = torch.zeros((10, phi_dim + 1))
        model = cls(
            init_data=dummy_data,
            phi_dim=phi_dim,
            hidden_dim=hidden_dim,
            n_bins=n_bins,
            n_layers_phi=n_layers_phi,
            n_layers_z=n_layers_z,
            k_neighbors=10,
        )

        # Pre-allocate the anchor buffers to the checkpointed shapes.
        if 'anchor.X_train' in state_dict:
            x_shape = state_dict['anchor.X_train'].shape
            y_shape = state_dict['anchor.y_train'].shape
            print(f"   [KNN] Pre-allocating buffers for {x_shape[0]} points.")
            model.anchor.X_train = torch.empty(x_shape)
            model.anchor.y_train = torch.empty(y_shape)
            model.anchor.fitted = True

        # Restore the anchor's k if the checkpoint carries it (older
        # checkpoints without 'anchor._k' fall back to the default of 10).
        if 'anchor._k' in state_dict:
            model.anchor.k = int(state_dict['anchor._k'])

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_z(self, phi, posterior=False, n_grid=4000):
        """Evaluate p(z | phi) on a dense grid and return posterior statistics.

        Args:
            phi: Raw embeddings (B, D).
            posterior: If True, also return the grid and normalized densities.
            n_grid: Grid resolution over the scaled residual range [-5, 5].

        Returns:
            z_mode: Posterior mode (B,).
            z_std: Posterior standard deviation (B,).
            (if posterior) z_grid_raw: Raw-z grid points (B, n_grid).
            (if posterior) probs: Normalized probabilities on the grid (B, n_grid).
        """
        self.eval()
        B = phi.shape[0]
        device = phi.device

        with torch.no_grad():
            # 1. Anchor + conditioning context.
            z_anchor_raw = self.anchor(phi)  # (B, 1)
            phi_norm = self.standardize_phi(phi)
            context = self.physics.encoder(phi_norm)

            # 2. Grid over the *scaled residual* axis.
            r_grid_scaled = torch.linspace(-5.0, 5.0, n_grid, device=device)
            r_input = r_grid_scaled.view(1, n_grid, 1).expand(B, -1, -1)
            flat_r = r_input.reshape(-1, 1)

            context_expanded = context.unsqueeze(1).expand(-1, n_grid, -1)
            flat_context = context_expanded.reshape(-1, context.shape[-1])

            # 3. Evaluate the flow in chunks (B * n_grid points -> OOM otherwise).
            chunk_size = 50000
            log_probs_list = []
            total_points = flat_r.shape[0]
            for k in range(0, total_points, chunk_size):
                end = min(k + chunk_size, total_points)
                chunk_lp = self.physics.flow(flat_context[k:end]).log_prob(flat_r[k:end])
                log_probs_list.append(chunk_lp)
            flat_log_prob = torch.cat(log_probs_list)

            log_probs = flat_log_prob.view(B, n_grid)
            probs = torch.exp(log_probs)
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)

            # 4. Map the scaled residual grid back to raw z around the anchor.
            r_grid_raw = self.unstandardize_residual(r_input.squeeze(-1), z_anchor_raw)
            z_grid_raw = z_anchor_raw + r_grid_raw

            # 5. Posterior statistics.
            peak_vals, peak_indices = torch.max(probs, dim=1)
            z_mode = torch.gather(z_grid_raw, 1, peak_indices.view(-1, 1)).squeeze()

            z_mean = torch.sum(probs * z_grid_raw, dim=1)
            z_var = torch.sum(probs * (z_grid_raw - z_mean.unsqueeze(1)) ** 2, dim=1)
            z_std = torch.sqrt(z_var + 1e-10)

            if posterior:
                return z_mode, z_std, z_grid_raw, probs
            return z_mode, z_std

    def spectral_logprob(self, phi):
        """Evaluate p(phi) for out-of-distribution detection.

        Returns:
            log_prob: log p(phi). Lower = more anomalous.
            latent_sq_norm: ||u||^2 of the latent coordinates (inverse
                transform of phi). Higher = more anomalous (Gaussian tails).
        """
        self.eval()
        with torch.no_grad():
            phi_norm = self.standardize_phi(phi)

            log_prob = self.manifold_flow.log_prob(phi_norm)

            # Data = Transform(Latent) in zuko, so invert to reach the latent.
            z_latent = self.manifold_flow.flow.transform().inv(phi_norm)
            latent_sq_norm = torch.sum(z_latent ** 2, dim=1)

            return log_prob, latent_sq_norm

    def redshift_logprob(self, phi, z):
        """Evaluate log p(z | phi) at hypothesis redshift(s) z.

        Useful for "is this redshift hypothesis consistent with the spectrum?"
        The Jacobian of the linear residual scaling (-log sigma) is applied so
        the returned density is with respect to raw z.
        """
        self.eval()
        with torch.no_grad():
            if z.ndim == 1:
                z = z.unsqueeze(-1)

            z_anchor = self.anchor(phi)
            phi_norm = self.standardize_phi(phi)
            context = self.physics.encoder(phi_norm)

            residual_raw = z - z_anchor
            residual_scaled = self.standardize_residual(residual_raw, z_anchor)

            log_prob_scaled = self.physics.flow(context).log_prob(residual_scaled)

            sigma = self.get_residual_std(z_anchor).squeeze()
            return log_prob_scaled - torch.log(sigma + 1e-6)

    def predict_full_inference(self, phi, n_grid=4000):
        """Full catalog-generation pass: posterior grid + all density scores.

        Combines :meth:`predict_z`, :meth:`spectral_logprob`, and the
        physics-flow likelihood at the posterior mode into one call so a
        catalog can be produced with a single forward per batch.

        Returns (per object): z_mode, z_std, z_grid_raw (B, n_grid),
        probs (B, n_grid), z_anchor, log p(phi), log p(z_mode | phi),
        and the joint score log p(phi) + log p(z_mode | phi).
        """
        self.eval()
        B = phi.shape[0]
        device = phi.device

        with torch.no_grad():
            # 1. Anchor.
            z_anchor_raw = self.anchor(phi)  # (B, 1)

            # 2. Manifold score p(phi).
            phi_norm = self.standardize_phi(phi)
            log_p_phi = self.manifold_flow.log_prob(phi_norm)  # (B,)

            # 3. Physics flow on the scaled-residual grid (chunked).
            context = self.physics.encoder(phi_norm)

            r_grid_scaled = torch.linspace(-5.0, 5.0, n_grid, device=device)
            r_input = r_grid_scaled.view(1, n_grid, 1).expand(B, -1, -1)
            flat_r = r_input.reshape(-1, 1)

            context_expanded = context.unsqueeze(1).expand(-1, n_grid, -1)
            flat_context = context_expanded.reshape(-1, context.shape[-1])

            chunk_size = 50000
            log_probs_list = []
            for k in range(0, flat_r.shape[0], chunk_size):
                end = min(k + chunk_size, flat_r.shape[0])
                lp = self.physics.flow(flat_context[k:end]).log_prob(flat_r[k:end])
                log_probs_list.append(lp)

            log_probs = torch.cat(log_probs_list).view(B, n_grid)
            probs = torch.exp(log_probs)
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)

            # Unscale the grid to raw z (sigma forced to (B, 1) for broadcasting).
            sigma_local = self.get_residual_std(z_anchor_raw.squeeze()).view(B, 1)
            r_grid_raw = r_input.squeeze(-1) * sigma_local
            z_grid_raw = z_anchor_raw + r_grid_raw

            # Posterior statistics.
            peak_indices = torch.argmax(probs, dim=1, keepdim=True)
            z_mode = torch.gather(z_grid_raw, 1, peak_indices).squeeze()
            z_mean = torch.sum(probs * z_grid_raw, dim=1)
            z_var = torch.sum(probs * (z_grid_raw - z_mean.unsqueeze(1)) ** 2, dim=1)
            z_std = torch.sqrt(z_var + 1e-10)

            # 4. Physics score at the mode, p(z_mode | phi).
            residual_raw = z_mode.unsqueeze(1) - z_anchor_raw
            residual_scaled = residual_raw / (sigma_local + 1e-6)
            log_prob_scaled = self.physics.flow(context).log_prob(residual_scaled)

            # Guard against a broadcasted (B, B) matrix from mismatched shapes.
            if log_prob_scaled.dim() > 1 and log_prob_scaled.shape[0] == log_prob_scaled.shape[1]:
                log_prob_scaled = log_prob_scaled.diag()

            log_p_z_given_phi = log_prob_scaled - torch.log(sigma_local.squeeze())

            # 5. Joint score.
            log_p_joint = log_p_phi + log_p_z_given_phi

        return (z_mode, z_std, z_grid_raw, probs, z_anchor_raw.squeeze(),
                log_p_phi, log_p_z_given_phi, log_p_joint)

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def forward_loss(self, phi_raw, z_raw, z_knn_anchor_raw, z_grad_raw, sig_phi_raw):
        """Joint NLL for both flows on a training batch.

        Args:
            phi_raw: (B, D) raw embeddings.
            z_raw: (B, 1) consensus redshifts.
            z_knn_anchor_raw: (B, 1) *precomputed cross-validated* anchor
                predictions (out-of-fold, to avoid leakage).
            z_grad_raw: (B, D) local dz/dphi gradients. Currently UNUSED --
                a tangent-slide augmentation was removed because it broke the
                phi <-> precomputed-anchor pairing -- but the argument is kept
                so precomputed loaders/dataset signatures stay valid.
            sig_phi_raw: (B, 1) local manifold scale (distance to the 50th
                neighbor), used to size the manifold-flow noise augmentation.

        Returns:
            (total_loss, {"nll_z": ..., "nll_phi": ...})
        """
        # 1. Scale the local manifold sigma into standardized-phi units.
        phi_std_scalar = self.phi_std.mean()
        sig_phi_norm = sig_phi_raw / phi_std_scalar

        # 2. Standardize phi.
        phi_norm = self.standardize_phi(phi_raw)

        # 3. Density-aware noise augmentation. Teaches the manifold flow that
        #    p(phi) has local volume rather than collapsing onto data points.
        noise_scale = 0.02
        eps_phi_norm = torch.randn_like(phi_norm) * sig_phi_norm * noise_scale
        phi_aug_norm = phi_norm + eps_phi_norm

        # --- Physics flow p(z | phi) ---
        # Target = (truth - precomputed CV anchor), scaled by the z-binned
        # sigma. Do NOT tangent-slide z here: z_knn_anchor_raw was precomputed
        # for phi_raw, and moving z would break that pairing.
        raw_residual = z_raw - z_knn_anchor_raw
        target_residual_scaled = self.standardize_residual(raw_residual, z_knn_anchor_raw)
        # Clamp inside the spline bound to prevent boundary collapse
        # (constant-bias failure mode of the RQS transform).
        target_residual_scaled = torch.clamp(target_residual_scaled, min=-9.99, max=9.99)

        if self.training:
            # Small jitter in the scaled space regularizes against overfitting
            # the exact anchor residuals; re-clamp in case it crossed a bound.
            jitter_scale = 0.05
            jitter = torch.randn_like(target_residual_scaled) * jitter_scale
            target_residual_scaled = torch.clamp(target_residual_scaled + jitter,
                                                 min=-9.99, max=9.99)

        context = self.physics.encoder(phi_aug_norm)
        loss_z = -self.physics.flow(context).log_prob(target_residual_scaled).mean()

        # --- Manifold flow p(phi) ---
        # Uses the AUGMENTED phi so the flow learns local manifold structure.
        loss_phi = -self.manifold_flow.log_prob(phi_aug_norm).mean()

        return loss_z + loss_phi, {"nll_z": loss_z.item(), "nll_phi": loss_phi.item()}
