"""Aristotle: the student model.

Aristotle is a spectrum-only encoder distilled from Plato. It replaces
Plato's AION-query cross-attention machinery with a plain 1D-CNN +
self-attention Transformer over the spectrum, but *re-uses Plato's trained
DeepProjectionHead frozen* so its 32-dim outputs live in the exact same
embedding space as the teacher's.

Distillation target: MSE against Plato's precomputed 32-dim embeddings
(see ``Aristotle_train.py``).
"""

import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from muon import MuonWithAuxAdam


class ConvNetTeacher(nn.Module):
    """1D-CNN spectrum encoder (~3M params with defaults).

    Identical construction to the block of the same name in
    ``plato/Plato.py`` (see that file for the full stem/body explanation);
    duplicated here so this model file reads standalone. Keeping the
    architecture identical to Plato's spectrum branch is intentional -- the
    frozen projection head expects features of the same character.
    """

    def __init__(self,
                 n_conv_layers: int = 5,
                 input_conv_channels: int = 1,
                 target_conv_dim: int = 768,
                 L_in: int = 7781,
                 L_target: int = 512,
                 default_kernel: int = 5,
                 n_narrow_layers: int = 2,
                 kernels: list = None,
                 strides: list = None,
                 paddings: list = None,
                 out_channels: list = None):
        super().__init__()
        if n_conv_layers < 1:
            n_conv_layers = 1
        self.n_conv_layers = n_conv_layers
        self.n_narrow_layers = n_narrow_layers
        self.input_conv_channels = input_conv_channels
        self.target_conv_dim = target_conv_dim

        if (kernels is None) or (strides is None) or (paddings is None) or (out_channels is None):
            # --- Automatic construction (narrow stem + downsampling body) ---
            n_total_layers = self.n_conv_layers + self.n_narrow_layers

            kernel_sizes = [3] * self.n_narrow_layers + [default_kernel] * self.n_conv_layers
            paddings = [1] * self.n_narrow_layers + [default_kernel // 2] * self.n_conv_layers
            strides_list = [1] * self.n_narrow_layers

            out_channels_list = []
            total_downsample = L_in / L_target
            total_downsample_done = 1

            stem_target_channels = 64
            narrow_ratio = (stem_target_channels / self.input_conv_channels) ** (1 / self.n_narrow_layers)
            for i in range(self.n_narrow_layers):
                out_channels_list.append(
                    np.ceil(self.input_conv_channels * (narrow_ratio ** (i + 1))).astype(int))

            conv_ratio = (self.target_conv_dim / stem_target_channels) ** (1 / self.n_conv_layers)
            for i in range(self.n_conv_layers):
                if total_downsample_done < total_downsample:
                    strides_list.append(2)
                    total_downsample_done *= 2
                else:
                    strides_list.append(1)
                out_channels_list.append(
                    np.ceil(stem_target_channels * (conv_ratio ** (i + 1))).astype(int))

            conv_layers_list = []
            for i in range(n_total_layers):
                in_c = self.input_conv_channels if i == 0 else out_channels_list[i - 1]
                out_c = self.target_conv_dim if i == n_total_layers - 1 else out_channels_list[i]
                conv_layers_list.append(nn.Conv1d(in_c, out_c,
                                                  kernel_size=kernel_sizes[i],
                                                  stride=strides_list[i],
                                                  padding=paddings[i]))
                conv_layers_list.append(nn.GELU())
                conv_layers_list.append(nn.GroupNorm(1, out_c))

            self.ConvNet = nn.Sequential(*conv_layers_list)
        else:
            # --- User-supplied layer specs ---
            conv_layers_list = []
            num_layers = np.max([len(kernels), len(strides), len(paddings), len(out_channels)])
            for lst in [kernels, strides, paddings, out_channels]:
                if len(lst) == 0:
                    raise ValueError("No kernels/strides/paddings/out_channels given")
                if len(lst) != num_layers:
                    lst += [lst[-1]] * (num_layers - len(lst))

            for i in range(num_layers):
                in_c = self.input_conv_channels if i == 0 else out_channels[i - 1]
                out_c = self.target_conv_dim if i == num_layers - 1 else out_channels[i]
                conv_layers_list.append(nn.Conv1d(in_c, out_c,
                                                  kernel_size=kernels[i],
                                                  stride=strides[i],
                                                  padding=paddings[i]))
                conv_layers_list.append(nn.GELU())
                conv_layers_list.append(nn.GroupNorm(1, out_c))

            self.ConvNet = nn.Sequential(*conv_layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L_in) preprocessed spectrum -> (B, target_conv_dim, ~L_target)
        return self.ConvNet(x)


class SelfAttentionLayer(nn.Module):
    """Standard pre-norm Transformer encoder layer (self-attention + FFN)."""

    def __init__(self, n_dim: int = 768, n_heads: int = 12, hidden_dim: int = 3072):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=n_dim, num_heads=n_heads,
                                               batch_first=True)
        self.norm2 = nn.LayerNorm(n_dim)
        self.ffn = nn.Sequential(
            nn.Linear(n_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_output
        x = x + self.ffn(self.norm2(x))
        return x


class ResidualBlock(nn.Module):
    """Linear residual block with a projection shortcut (matches Plato's)."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.main_path = nn.Linear(dim_in, dim_out)
        self.activation = nn.GELU()
        self.shortcut = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.activation(self.main_path(x)) + self.shortcut(x))


class DeepProjectionHead(nn.Module):
    """Residual projection head 768 -> 3072 -> 1536 -> 768 -> 32.

    Must match Plato's ``DeepProjectionHead`` exactly: Aristotle loads the
    teacher's trained weights into this module and freezes them.
    """

    def __init__(self,
                 n_dim_in: int = 768,
                 n_hidden: int = 3072,
                 n_dim_out_primary: int = 32):
        super().__init__()
        h1 = n_hidden
        h2 = n_hidden // 2
        h3 = n_hidden // 4

        self.projection_body = nn.Sequential(
            nn.Linear(n_dim_in, h1),
            nn.GELU(),
            nn.LayerNorm(h1),
            ResidualBlock(h1, h2),
            ResidualBlock(h2, h3),
        )
        self.proj32 = nn.Linear(h3, n_dim_out_primary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj32(self.projection_body(x))


class Aristotle(nn.Module):
    """Spectrum-only student: (B, 1, L_in) spectrum -> (B, 32) embedding.

    Pipeline: ConvNet encoder -> [CLS] + register tokens + learned positional
    embedding -> self-attention Transformer stack -> pool [CLS] -> frozen
    Plato projection head.

    On construction, this class searches the current directory (and
    subdirectories) for ``plato.pth`` and loads + freezes the teacher's
    DeepProjectionHead from it; construction fails loudly if the checkpoint
    cannot be found, since without it the output space is undefined.
    """

    # True output length of the ConvNet given L_in=7781, n_conv=5, L_target=512.
    # (Stride-2 layers halve with ceil-like rounding, landing on 487, not 512.)
    PLATO_CNN_OUT_LENGTH = 487
    PLATO_CHECKPOINT_NAME = 'plato.pth'

    def __init__(self,
                 # --- CNN feature extractor (matched to Plato) ---
                 L_in: int = 7781,
                 n_conv_layers: int = 5,
                 n_narrow_layers: int = 2,
                 input_conv_channels: int = 1,
                 # --- Core dimensions (matched to Plato) ---
                 n_dim: int = 768,
                 hidden_dim: int = 3072,
                 # --- Transformer encoder ---
                 n_layers: int = 4,
                 n_heads: int = 12,
                 n_registers: int = 4):
        super().__init__()
        print("Initializing 'Aristotle' Student Model...")
        self.n_dim = n_dim
        self.n_registers = int(n_registers)

        # --- Find and load the frozen Plato projection head ---
        try:
            plato_path = self._find_plato_checkpoint(self.PLATO_CHECKPOINT_NAME)
            print(f"Found Plato checkpoint at: {plato_path}")
            self.projection_head = self._load_frozen_plato_head(
                checkpoint_path=plato_path,
                n_dim_in=n_dim,
                n_hidden=hidden_dim,
                n_dim_out_primary=32,
            )
            print("Successfully loaded and froze Plato projection head.")
        except Exception as e:
            print("CRITICAL ERROR: Failed to initialize Aristotle.")
            print(f"Details: {e}")
            raise

        # --- Spectrum encoder ---
        self.cnn_encoder = ConvNetTeacher(
            n_conv_layers=n_conv_layers,
            n_narrow_layers=n_narrow_layers,
            input_conv_channels=input_conv_channels,
            target_conv_dim=n_dim,
            L_in=L_in,
            L_target=512,  # must match Plato's init so token counts agree
        )
        self.context_norm = nn.LayerNorm(n_dim)

        # --- Transformer encoder (global correlator) ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_dim))
        self.registers = nn.Parameter(torch.randn(1, self.n_registers, n_dim))

        total_seq_len = 1 + self.n_registers + self.PLATO_CNN_OUT_LENGTH
        self.pos_embed = nn.Parameter(torch.randn(1, total_seq_len, n_dim))

        self.transformer_layers = nn.ModuleList([
            SelfAttentionLayer(n_dim=n_dim, n_heads=n_heads, hidden_dim=hidden_dim)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(n_dim)
        print("Aristotle model body initialized successfully.")

    def _find_plato_checkpoint(self, checkpoint_name: str) -> str:
        """Search the working directory tree for the Plato checkpoint."""
        for root, dirs, files in os.walk('.'):
            if checkpoint_name in files:
                return os.path.join(root, checkpoint_name)
        raise FileNotFoundError(
            f"Could not find '{checkpoint_name}' in '{os.getcwd()}' or subdirectories."
        )

    def _load_frozen_plato_head(self, checkpoint_path: str, **kwargs) -> DeepProjectionHead:
        """Extract, load, and freeze the DeepProjectionHead from a Plato checkpoint.

        Accepts either a full training checkpoint (dict with
        'model_state_dict') or a bare state dict, and strips both plain and
        DDP ('module.') prefixes from the projection-head keys.
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                plato_state_dict = checkpoint['model_state_dict']
            else:
                print("Warning: 'model_state_dict' key not found. "
                      "Assuming file contains raw weights.")
                plato_state_dict = checkpoint
        except Exception as e:
            print(f"Error: Could not load checkpoint from {checkpoint_path}")
            raise e

        head_weights_clean = OrderedDict()
        ddp_prefix = 'module.projection_head.'
        prefix = 'projection_head.'
        for key, value in plato_state_dict.items():
            if key.startswith(ddp_prefix):
                head_weights_clean[key[len(ddp_prefix):]] = value
            elif key.startswith(prefix):
                head_weights_clean[key[len(prefix):]] = value

        if not head_weights_clean:
            raise ValueError(
                f"Could not find any weights with prefix '{prefix}' or "
                f"'{ddp_prefix}' in checkpoint. Check key names."
            )

        plato_head = DeepProjectionHead(**kwargs)
        plato_head.load_state_dict(head_weights_clean)
        for param in plato_head.parameters():
            param.requires_grad = False
        return plato_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, L_in) preprocessed spectrum -> (B, 32) embedding."""
        B = x.shape[0]

        conv_context = self.cnn_encoder(x)
        if conv_context.shape[2] != self.PLATO_CNN_OUT_LENGTH:
            raise ValueError(
                f"CNN output length mismatch! Expected {self.PLATO_CNN_OUT_LENGTH}, "
                f"but got {conv_context.shape[2]}. Check ConvNetTeacher logic."
            )

        conv_context = conv_context.permute(0, 2, 1)  # (B, L, n_dim)
        conv_context = self.context_norm(conv_context)

        # Prepend [CLS] + register tokens, add positional embedding.
        cls = self.cls_token.expand(B, -1, -1)
        reg = self.registers.expand(B, -1, -1)
        x_with_tokens = torch.cat([cls, reg, conv_context], dim=1)
        x_with_pos = x_with_tokens + self.pos_embed

        for layer in self.transformer_layers:
            x_with_pos = layer(x_with_pos)

        x_normed = self.final_norm(x_with_pos)
        pooled_output = x_normed[:, 0]  # [CLS] token, (B, n_dim)

        return self.projection_head(pooled_output)  # frozen head -> (B, 32)


def create_aristotle_optimizer(model: Aristotle, lr_muon: float = 1e-4,
                               lr_adam: float = 1e-4, wd: float = 0.00):
    """Build the MuonWithAuxAdam optimizer for Aristotle.

    Muon updates >=2D "body" weights; AdamW handles 1D params plus the
    special tokens ([CLS], registers, positional embedding). The frozen
    projection head is excluded entirely.
    """
    if hasattr(model, 'module'):
        model = model.module  # unwrap DDP

    body_modules = [
        model.cnn_encoder,
        model.context_norm,
        model.transformer_layers,
        model.final_norm,
        # model.projection_head is FROZEN and excluded.
    ]
    non_body_params = [
        model.registers,
        model.cls_token,
        model.pos_embed,
    ]

    hidden_weights = []       # Muon: 2D+ weights
    hidden_gains_biases = []  # AdamW: 1D params
    for module in body_modules:
        for p in module.parameters():
            (hidden_weights if p.ndim >= 2 else hidden_gains_biases).append(p)

    param_groups = [
        dict(params=hidden_weights, use_muon=True,
             lr=lr_muon, weight_decay=wd),
        dict(params=hidden_gains_biases + non_body_params, use_muon=False,
             lr=lr_adam, betas=(0.9, 0.95), weight_decay=wd),
    ]
    return MuonWithAuxAdam(param_groups)
