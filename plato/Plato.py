"""Plato: the teacher model.

Plato learns a 32-dim spectral embedding by attentively pooling AION-1
embedding sequences (the *queries*) against the object's own spectrum (the
*context*). Each of its 12 fusion layers interleaves self-attention over the
query tokens with cross-attention into a ConvNet projection of the spectrum.
A [CLS] token is pooled and pushed through a deep residual projection head to
produce the final 32-dim embedding.

Training supervision comes from a triplet loss whose ground truth is the
normalized cross-correlation (CCF) peak between preprocessed spectra -- see
``Plato_training_helpers.py``.
"""

import numpy as np
import torch
import torch.nn as nn
from muon import MuonWithAuxAdam


class ConvNetTeacher(nn.Module):
    """1D-CNN spectrum encoder (~3M params with defaults).

    Maps a preprocessed spectrum (B, 1, L_in) to a token sequence
    (B, target_conv_dim, ~L_target) that attention layers can consume.

    The stack is built in two stages:
      * ``n_narrow_layers`` stem convs (kernel 3, stride 1) that grow channels
        geometrically to 64 while preserving resolution, to capture narrow
        spectral features (emission/absorption lines).
      * ``n_conv_layers`` body convs (kernel ``default_kernel``) that grow
        channels geometrically to ``target_conv_dim`` and stride-2 downsample
        until the output length reaches ``L_target``.

    Passing explicit ``kernels``/``strides``/``paddings``/``out_channels``
    lists overrides the automatic construction (shorter lists are padded by
    repeating their last entry).

    NOTE: An identical copy of this block lives in ``aristotle/Aristotle.py``;
    the duplication is deliberate so each model file reads standalone.
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
            # --- Automatic construction ---
            n_total_layers = self.n_conv_layers + self.n_narrow_layers

            kernel_sizes = [3] * self.n_narrow_layers + [default_kernel] * self.n_conv_layers
            paddings = [1] * self.n_narrow_layers + [default_kernel // 2] * self.n_conv_layers
            strides_list = [1] * self.n_narrow_layers

            out_channels_list = []
            total_downsample = L_in / L_target
            total_downsample_done = 1

            # Stem: geometric channel growth input_channels -> 64.
            stem_target_channels = 64
            narrow_ratio = (stem_target_channels / self.input_conv_channels) ** (1 / self.n_narrow_layers)
            for i in range(self.n_narrow_layers):
                out_channels_list.append(
                    np.ceil(self.input_conv_channels * (narrow_ratio ** (i + 1))).astype(int))

            # Body: geometric channel growth 64 -> target_conv_dim, striding by
            # 2 each layer until the cumulative downsample reaches L_in/L_target.
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


class CrossAttentionLayer(nn.Module):
    """Pre-norm cross-attention layer: query tokens attend to spectrum tokens."""

    def __init__(self, n_dim: int = 768, n_cross_attn_heads: int = 12,
                 cross_attn_hidden: int = 3072):
        super().__init__()
        self.n_dim = n_dim
        self.n_cross_attn_heads = n_cross_attn_heads
        self.cross_attn_hidden = cross_attn_hidden

        self.cross_attn = nn.MultiheadAttention(embed_dim=n_dim,
                                                num_heads=n_cross_attn_heads,
                                                batch_first=True)
        self.norm1 = nn.LayerNorm(n_dim)
        self.ffn = nn.Sequential(
            nn.Linear(n_dim, cross_attn_hidden),
            nn.GELU(),
            nn.Linear(cross_attn_hidden, n_dim),
        )
        self.norm2 = nn.LayerNorm(n_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor,
                need_weights: bool = False):
        x = query
        attn_output, attn_weights = self.cross_attn(self.norm1(x), context, context,
                                                    need_weights=need_weights)
        x = x + attn_output
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


class ResidualBlock(nn.Module):
    """Linear residual block with a projection shortcut.

    (B, dim_in) -> LayerNorm(GELU(Linear(x)) + Linear_shortcut(x)) -> (B, dim_out)
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.main_path = nn.Linear(dim_in, dim_out)
        self.activation = nn.GELU()
        self.shortcut = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.activation(self.main_path(x)) + self.shortcut(x))


class DeepProjectionHead(nn.Module):
    """Residual projection head: 768 -> 3072 -> 1536 -> 768 -> 32 (~16.6M params).

    This head is trained inside Plato and later *frozen and re-used verbatim*
    by the Aristotle student so both models share one output embedding space.
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
        # x: (B, 768) pooled token -> (B, 32) embedding
        return self.proj32(self.projection_body(x))


class Plato(nn.Module):
    """Attentive-pooling teacher: (AION queries, spectrum) -> 32-dim embedding.

    Forward inputs:
        queries: (B, T, n_dim) AION-1 embedding sequence for the object.
        context: (B, 1, L_in) preprocessed spectrum
                 (see ``preprocessing.preprocess_for_model``).

    Each fusion layer runs self-attention over [CLS] + registers + queries,
    then cross-attends that sequence into registers + ConvNet(spectrum).
    The [CLS] token is pooled and projected to the 32-dim output space.
    """

    def __init__(self,
                 n_dim: int = 768,
                 n_layers: int = 12,
                 n_self_attn_heads: int = 12,
                 self_attn_hidden: int = 3072,
                 n_cross_attn_heads: int = 12,
                 cross_attn_hidden: int = 3072,
                 n_registers: int = 4,
                 output_embedding_dim: int = 32):
        super().__init__()
        self.n_dim = n_dim
        self.n_registers = n_registers
        self.self_attn_hidden = self_attn_hidden
        self.output_embedding_dim = output_embedding_dim

        # Spectrum encoder + normalization of its token sequence.
        self.ConvNet = ConvNetTeacher(target_conv_dim=self.n_dim)
        self.context_norm = nn.LayerNorm(self.n_dim)

        # Learnable tokens: registers (attention "scratch space") and [CLS].
        self.registers = nn.Parameter(torch.randn(1, self.n_registers, self.n_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.n_dim))

        # Interleaved self-/cross-attention fusion stack.
        self.fusion_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.fusion_layers.append(nn.ModuleDict({
                'self_attn': SelfAttentionLayer(n_dim, n_self_attn_heads, self_attn_hidden),
                'cross_attn': CrossAttentionLayer(n_dim, n_cross_attn_heads, cross_attn_hidden),
            }))

        self.final_norm = nn.LayerNorm(self.n_dim)
        self.projection_head = DeepProjectionHead(n_dim_in=self.n_dim,
                                                  n_hidden=self.self_attn_hidden,
                                                  n_dim_out_primary=self.output_embedding_dim)

    def forward(self, queries: torch.Tensor, context: torch.Tensor,
                need_weights: bool = False):
        B = queries.shape[0]

        # Encode the spectrum once; every fusion layer cross-attends into it.
        conv_context = self.ConvNet(context).permute(0, 2, 1)  # (B, ~L_target, n_dim)
        normed_context = self.context_norm(conv_context)
        cross_reg = self.registers.expand(B, -1, -1)
        context_with_reg = torch.cat([cross_reg, normed_context], dim=1)

        # Query sequence: [CLS] + registers + AION embeddings.
        attn_reg = self.registers.expand(B, -1, -1)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, attn_reg, queries], dim=1)

        cross_attn_weights_list = []
        for layer in self.fusion_layers:
            x = layer['self_attn'](x)
            x, attn_weights = layer['cross_attn'](x, context_with_reg,
                                                  need_weights=need_weights)
            if need_weights:
                # Drop the register positions so weights index spectrum tokens.
                cross_attn_weights_list.append(attn_weights[:, :, :, self.n_registers:])

        x = self.final_norm(x)
        pooled_output = x[:, 0]  # [CLS] token, (B, n_dim)
        embed32 = self.projection_head(pooled_output)

        if need_weights:
            return embed32, cross_attn_weights_list
        return embed32


def create_plato_optimizer(model: Plato, lr_muon: float = 3e-4,
                           lr_adam: float = 3e-4, wd: float = 0.00):
    """Build the MuonWithAuxAdam optimizer for Plato.

    Muon updates the >=2D "body" weights (conv/linear matrices); AdamW handles
    everything 1D (norm gains/biases) plus the special parameters (registers,
    [CLS], and the final 32-dim projection layer).
    """
    if hasattr(model, 'module'):
        model = model.module  # unwrap DDP

    body_modules = [
        model.ConvNet,
        model.context_norm,
        model.fusion_layers,
        model.final_norm,
        model.projection_head.projection_body,
    ]
    non_body_params = (list(model.projection_head.proj32.parameters())
                       + [model.registers, model.cls_token])

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
