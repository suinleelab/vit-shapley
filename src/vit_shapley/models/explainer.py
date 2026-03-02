"""Explainer ViT for ViT-Shapley Stage 3.

The explainer φ(x) ∈ R^{B × n × C} is a ViT that takes a full image and
produces per-patch Shapley value estimates in a single forward pass, without
the exponential cost of exact Shapley computation.

Architecture (paper default):
  - ViT backbone (pretrained/fine-tuned from surrogate)
  - 1 extra ViT attention block (CLS + patch tokens)
  - LayerNorm before MLP
  - 3-layer MLP head: D → 4D → 4D → C (with GELU activations)
  - Tanh output activation
  - Additive normalisation to hard-enforce efficiency axiom:
      φ'_i = φ_i + (v(grand) − v(null) − Σ_j φ_j) / n

Training objective:
    L(φ) = n · E_{x,S} [ ‖v(∅;x) + Σ_{i∈S} φ'_i(x) − v(S;x)‖² ]

where v(S;x) = softmax(surrogate(x, mask=S)) is the frozen surrogate output
and masks S are sampled from the Shapley distribution ∝ 1/(k(n−k)).
"""

from __future__ import annotations

import copy
import os
from typing import Optional

import timm
import torch
import torch.nn as nn


def _reset_weights(m: nn.Module) -> None:
    """Re-initialise a module's learnable parameters with timm defaults."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ExplainerViT(nn.Module):
    """ViT-based explainer that produces per-patch Shapley value estimates.

    The model uses a pretrained/fine-tuned ViT backbone to extract patch-level
    features, then passes them through optional extra attention blocks and a
    multi-layer MLP head to produce Shapley value estimates.

    Args:
        vit: A timm ViT model (output of :func:`timm.create_model`).
        num_classes: Number of output classes (Shapley value dimension).
        num_attn_blocks: Number of extra ViT attention blocks appended to the
            backbone (paper default: 1).
        num_mlp_layers: Depth of the Shapley projection MLP — 1, 2, or 3
            (paper default: 3).
        mlp_ratio: Width expansion ratio for MLP hidden layers (paper default: 4.0).
        include_cls: Include the CLS token in the extra attention blocks (paper
            default: True). CLS output is discarded after the MLP.
        use_norm: Apply LayerNorm before the MLP when ``num_attn_blocks > 0``
            (paper default: True).
        activation: Non-linearity applied after the MLP — ``"tanh"`` or ``None``
            (paper default: ``"tanh"``).
        normalization: Enforce the efficiency axiom via ``"additive"``
            normalisation, or ``None`` to skip (paper default: ``"additive"``).
            When ``"additive"``, :meth:`forward` requires ``grand`` and ``null``.
    """

    def __init__(
        self,
        vit: nn.Module,
        num_classes: int,
        num_attn_blocks: int = 1,
        num_mlp_layers: int = 3,
        mlp_ratio: float = 4.0,
        include_cls: bool = True,
        use_norm: bool = True,
        activation: Optional[str] = "tanh",
        normalization: Optional[str] = "additive",
    ) -> None:
        super().__init__()
        self.vit = vit
        self.num_classes = num_classes
        self.include_cls = include_cls
        self.activation = activation
        self.normalization = normalization

        embed_dim: int = vit.embed_dim

        # ── Extra attention blocks ──────────────────────────────────────────
        if num_attn_blocks == 0:
            self.attention_blocks = nn.ModuleList()
        else:
            last_block = vit.blocks[-1]
            blocks: list[nn.Module] = []
            for _ in range(num_attn_blocks):
                b = copy.deepcopy(last_block)
                # Remove stochastic depth; extra blocks are always active.
                for attr in ("drop_path", "drop_path1", "drop_path2"):
                    if hasattr(b, attr) and not isinstance(
                        getattr(b, attr), nn.Identity
                    ):
                        setattr(b, attr, nn.Identity())
                b.apply(_reset_weights)
                blocks.append(b)
            self.attention_blocks = nn.ModuleList(blocks)
            # The backbone's final LayerNorm already normalises the tokens;
            # skip norm1 of the first extra block to avoid double normalisation.
            self.attention_blocks[0].norm1 = nn.Identity()

        # ── MLP head (Shapley projection) ───────────────────────────────────
        hidden_dim = int(embed_dim * mlp_ratio)
        # Use the same activation class as the backbone blocks (typically GELU).
        act_class = vit.blocks[0].mlp.act.__class__

        mlp_layers: list[nn.Module] = []
        if use_norm and num_attn_blocks > 0:
            mlp_layers.append(nn.LayerNorm(embed_dim))

        if num_mlp_layers == 1:
            mlp_layers.append(nn.Linear(embed_dim, num_classes))
        elif num_mlp_layers == 2:
            mlp_layers += [
                nn.Linear(embed_dim, hidden_dim),
                act_class(),
                nn.Linear(hidden_dim, num_classes),
            ]
        elif num_mlp_layers == 3:
            mlp_layers += [
                nn.Linear(embed_dim, hidden_dim),
                act_class(),
                nn.Linear(hidden_dim, hidden_dim),
                act_class(),
                nn.Linear(hidden_dim, num_classes),
            ]
        else:
            raise ValueError(f"num_mlp_layers must be 1, 2, or 3; got {num_mlp_layers}")

        self.shapley_head = nn.Sequential(*mlp_layers)

    def forward(
        self,
        x: torch.Tensor,
        grand: Optional[torch.Tensor] = None,
        null: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return (normalised) Shapley value estimates.

        Args:
            x: Image batch ``(B, C_in, H, W)``.
            grand: Surrogate softmax output for the grand coalition (all patches
                visible), shape ``(B, num_classes)`` — required when
                ``normalization="additive"``.
            null: Surrogate softmax output for the null coalition (no patches
                visible), shape ``(B, num_classes)`` or ``(1, num_classes)`` —
                required when ``normalization="additive"``.

        Returns:
            Shapley values ``(B, num_patches, num_classes)``.
        """
        # backbone forward_features returns (B, 1+n, D) in timm 1.0+
        features = self.vit.forward_features(x)
        num_prefix = self.vit.num_prefix_tokens
        cls_tokens = features[:, :num_prefix, :]  # (B, 1, D)
        patch_tokens = features[:, num_prefix:, :]  # (B, n, D)

        if self.include_cls:
            embedding = torch.cat([cls_tokens, patch_tokens], dim=1)  # (B, 1+n, D)
        else:
            embedding = patch_tokens  # (B, n, D)

        # Extra attention blocks
        for block in self.attention_blocks:
            embedding = block(embedding)

        # MLP head
        pred = self.shapley_head(embedding)  # (B, 1+n, C) or (B, n, C)

        # Discard CLS-position output
        if self.include_cls:
            pred = pred[:, num_prefix:, :]  # (B, n, C)

        # Output activation
        if self.activation == "tanh":
            pred = pred.tanh()
        elif self.activation is not None:
            raise ValueError(f"Unsupported activation: {self.activation!r}")

        # Additive normalisation: hard-enforce Σ_i φ'_i = grand − null
        if self.normalization == "additive":
            if grand is None or null is None:
                raise ValueError(
                    "grand and null must be provided when normalization='additive'. "
                    "Pass surrogate outputs for the grand and null coalitions, "
                    "or set normalization=None to skip normalisation."
                )
            n = pred.shape[1]
            # grand/null: (B, C) or broadcastable; pred.sum(dim=1): (B, C)
            residual = (grand - null - pred.sum(dim=1)) / n  # (B, C)
            pred = pred + residual.unsqueeze(1)  # (B, n, C)
        elif self.normalization is not None:
            raise ValueError(f"Unsupported normalization: {self.normalization!r}")

        return pred  # (B, n, C)


def build_vit_explainer(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 10,
    surrogate_ckpt_path: Optional[str | os.PathLike] = None,
    num_attn_blocks: int = 1,
    num_mlp_layers: int = 3,
    mlp_ratio: float = 4.0,
    include_cls: bool = True,
    use_norm: bool = True,
    activation: Optional[str] = "tanh",
    normalization: Optional[str] = "additive",
) -> ExplainerViT:
    """Build an :class:`ExplainerViT`, optionally initialised from a surrogate checkpoint.

    The surrogate checkpoint keys start with ``"vit."`` (full
    :class:`~vit_shapley.models.SurrogateViT` state dict) — this prefix is
    stripped to load into the bare timm ViT.  The extra attention blocks and
    ``shapley_head`` are always randomly initialised (new parameters).

    Args:
        model_name: timm ViT model name (e.g. ``"vit_tiny_patch16_224"``).
        num_classes: Number of output classes (Shapley value dimension).
        surrogate_ckpt_path: Path to a ``best_surrogate.pth`` file produced by
                             :func:`~vit_shapley.training.train_surrogate`.
                             If ``None``, the model is randomly initialised.
        num_attn_blocks: Extra ViT attention blocks after the backbone (paper
                         default: 1).
        num_mlp_layers: Depth of the Shapley MLP head (paper default: 3).
        mlp_ratio: Hidden-layer width ratio for the MLP (paper default: 4.0).
        include_cls: Include CLS token in extra attention blocks (paper
                     default: True).
        use_norm: LayerNorm before MLP when ``num_attn_blocks > 0`` (paper
                  default: True).
        activation: Post-MLP activation (paper default: ``"tanh"``).
        normalization: Efficiency-axiom enforcement (paper default:
                       ``"additive"``).

    Returns:
        :class:`ExplainerViT` ready for training.

    Example::

        explainer = build_vit_explainer(
            "vit_tiny_patch16_224",
            num_classes=10,
            surrogate_ckpt_path="checkpoints/surrogate_attn/best_surrogate.pth",
        )
    """
    vit = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    if surrogate_ckpt_path is not None:
        ckpt = torch.load(surrogate_ckpt_path, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        # SurrogateViT state dict has keys like "vit.patch_embed.proj.weight"
        if all(k.startswith("vit.") for k in sd):
            sd = {k[4:]: v for k, v in sd.items()}  # strip "vit." prefix
        vit.load_state_dict(sd)
    return ExplainerViT(
        vit,
        num_classes,
        num_attn_blocks=num_attn_blocks,
        num_mlp_layers=num_mlp_layers,
        mlp_ratio=mlp_ratio,
        include_cls=include_cls,
        use_norm=use_norm,
        activation=activation,
        normalization=normalization,
    )
