"""Shapley value heatmap visualisation utilities for ViT-Shapley.

Provides three layers of reusable functionality:

1. **Inference** — :func:`compute_shapley_values`: runs the frozen surrogate to
   obtain null/grand coalition values, then calls the explainer to produce
   per-patch Shapley estimates ``(B, n, C)``.

2. **Rendering** — :func:`denormalize_imagenet` and :func:`shapley_to_heatmap`:
   convert normalised image tensors and flat Shapley vectors into numpy arrays
   ready for ``matplotlib.axes.Axes.imshow``.

3. **Plotting** — :func:`plot_shapley_heatmaps`: assembles a full figure with
   one row per image sample and one column per class, each cell showing the
   Shapley heatmap overlaid on a greyscale copy of the original image.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# ImageNet normalisation constants (matching training transforms)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])[:, None, None]  # CHW
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])[:, None, None]


# ---------------------------------------------------------------------------
# 1. Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_shapley_values(
    explainer: torch.nn.Module,
    surrogate: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Run the surrogate + explainer to get per-patch Shapley estimates.

    Computes the null coalition value (all patches masked) and grand coalition
    value (all patches visible) via the surrogate, then passes them together
    with the images to the explainer to obtain normalised Shapley values.

    Args:
        explainer: Trained :class:`~vit_shapley.models.ExplainerViT` in eval
                   mode.
        surrogate: Frozen :class:`~vit_shapley.models.SurrogateViT` in eval
                   mode.
        images: Image batch ``(B, C, H, W)`` on CPU.
        device: Target compute device.

    Returns:
        Tuple of:
        - ``phi``: Numpy float32 array ``(B, num_patches, num_classes)``.
        - ``grand_probs``: Numpy float32 array ``(B, num_classes)`` — surrogate
          softmax output for the grand coalition (all patches visible).
    """
    images = images.to(device)
    B = images.size(0)
    num_patches: int = surrogate.vit.patch_embed.num_patches

    null_mask = torch.zeros(B, num_patches, device=device)
    null_probs = surrogate(images, patch_mask=null_mask).softmax(dim=-1)

    grand_mask = torch.ones(B, num_patches, device=device)
    grand_probs = surrogate(images, patch_mask=grand_mask).softmax(dim=-1)

    phi = explainer(images, grand=grand_probs, null=null_probs)  # (B, n, C)
    return phi.cpu().numpy(), grand_probs.cpu().numpy()


# ---------------------------------------------------------------------------
# 2. Rendering helpers
# ---------------------------------------------------------------------------

def denormalize_imagenet(img_chw: np.ndarray) -> np.ndarray:
    """Undo ImageNet channel normalisation.

    Args:
        img_chw: Float array ``(3, H, W)`` normalised with ImageNet mean/std.

    Returns:
        Float array ``(H, W, 3)`` with values clipped to ``[0, 1]``.
    """
    img = img_chw * _IMAGENET_STD + _IMAGENET_MEAN  # CHW
    return np.clip(img.transpose(1, 2, 0), 0.0, 1.0)  # HWC


def shapley_to_heatmap(
    shapley_patch: np.ndarray,
    image_size: int,
    grid_h: int,
    grid_w: int,
    cmap,
    alpha: float = 0.9,
) -> np.ndarray:
    """Convert a flat per-patch Shapley vector to an RGBA heatmap.

    The colourmap is centred at zero: a Shapley value of zero maps to the
    midpoint of the colourmap (0.5), while ±max_abs maps to the extremes
    (1.0 and 0.0 respectively).  The patch grid is bilinearly upsampled to
    ``image_size × image_size``.

    Args:
        shapley_patch: Float array ``(num_patches,)`` — per-patch Shapley
                       values for a single class.
        image_size: Output spatial resolution (e.g. 224).
        grid_h: Height of the patch grid (e.g. 14).
        grid_w: Width of the patch grid (e.g. 14).
        cmap: Diverging matplotlib/seaborn colourmap (e.g. ``"icefire"`` or
              ``"RdBu_r"``).  Positive values appear warm (red), negative
              values cool (blue).
        alpha: Opacity of the returned RGBA image (0 = transparent, 1 = opaque).

    Returns:
        RGBA float32 array ``(image_size, image_size, 4)``.
    """
    max_abs = np.abs(shapley_patch).max()
    if max_abs > 0:
        normalized = 0.5 + shapley_patch / (2.0 * max_abs)
    else:
        normalized = np.full_like(shapley_patch, 0.5)

    # Bilinear upsample from patch grid to image resolution
    grid = normalized.reshape(1, 1, grid_h, grid_w).astype(np.float32)
    upsampled = (
        F.interpolate(
            torch.from_numpy(grid),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        .numpy()
        .reshape(image_size, image_size)
    )

    rgba = cmap(upsampled).astype(np.float32)  # (H, W, 4)
    rgba[..., 3] = alpha
    return rgba


def _grey_rgba(img_hwc: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Convert an HWC RGB image to an inverted-luminance greyscale RGBA array.

    Inverting the luminance (``1 - mean``) darkens bright image regions so
    the heatmap overlay stands out more clearly against light backgrounds.

    Args:
        img_hwc: Float array ``(H, W, 3)`` with values in ``[0, 1]``.
        alpha: Opacity of the returned RGBA image.

    Returns:
        RGBA float32 array ``(H, W, 4)``.
    """
    grey = 1.0 - img_hwc.mean(axis=2)  # (H, W)
    rgba = np.zeros((*grey.shape, 4), dtype=np.float32)
    rgba[..., :3] = grey[..., None]
    rgba[..., 3] = alpha
    return rgba


# ---------------------------------------------------------------------------
# 3. Figure assembly
# ---------------------------------------------------------------------------

def plot_shapley_heatmaps(
    images_list: Sequence[np.ndarray],
    labels_list: Sequence[int],
    phi: np.ndarray,
    class_cols: Sequence[int],
    class_names: Sequence[str],
    image_size: int = 224,
    cmap=None,
    grand_probs: Optional[np.ndarray] = None,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Assemble a Shapley heatmap figure.

    Layout::

        [image | gap | heatmap_class_A | heatmap_class_B | …]
         row 0
         row 1
         …

    Each heatmap cell shows the per-patch Shapley values for one class
    overlaid on a greyscale copy of the original image.  The column header
    (first row only) shows the class name; the surrogate grand-coalition
    probability for that class is shown above every heatmap cell.

    Args:
        images_list: Sequence of ``N`` CHW float arrays (normalised with
                     ImageNet mean/std) — i.e. the raw tensors from the dataset.
        labels_list: Ground-truth integer class labels, length ``N``.
        phi: Shapley value array ``(N, num_patches, num_classes)`` as returned
             by :func:`compute_shapley_values`.
        class_cols: Indices of the classes to show as heatmap columns.
        class_names: Human-readable name for every class index (length must
                     cover all indices in ``labels_list`` and ``class_cols``).
        image_size: Spatial resolution used during inference (e.g. 224).
        cmap: Diverging matplotlib colourmap.  Defaults to seaborn ``"icefire"``.
        grand_probs: Surrogate softmax output for the grand coalition,
                     shape ``(N, num_classes)`` as returned by
                     :func:`compute_shapley_values`.  When provided, the
                     probability for each displayed class is shown as a small
                     number above every heatmap cell.

    Returns:
        A :class:`matplotlib.figure.Figure` — call ``.savefig()`` or
        ``plt.show()`` on the returned object.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if cmap is None:
        import seaborn as sns
        cmap = sns.color_palette("icefire", as_cmap=True)

    N = len(images_list)
    n_heat = len(class_cols)
    num_patches = phi.shape[1]
    grid_size = int(num_patches ** 0.5)

    # Column layout: [image] [narrow gap] [heatmap × n_heat]
    col_widths = [1.0, 0.15] + [1.0] * n_heat
    fig = plt.figure(figsize=(sum(col_widths) * 2.0, N * 2.3))
    outer = gridspec.GridSpec(
        1, len(col_widths), width_ratios=col_widths, wspace=0.05, hspace=0
    )

    def _col_axes(col_idx: int):
        inner = gridspec.GridSpecFromSubplotSpec(
            N, 1, subplot_spec=outer[col_idx], hspace=0.2
        )
        return [fig.add_subplot(inner[r]) for r in range(N)]

    img_axes = _col_axes(0)
    gap_axes = _col_axes(1)
    heat_axes = [_col_axes(2 + c) for c in range(n_heat)]

    for ax in gap_axes:
        ax.axis("off")

    for row, (img_chw, label) in enumerate(zip(images_list, labels_list)):
        img_rgb = denormalize_imagenet(img_chw)  # HWC [0,1]
        grey_rgba = _grey_rgba(img_rgb, alpha=0.5)

        # Original image
        ax = img_axes[row]
        ax.imshow(img_rgb)
        ax.set_title(class_names[label], fontsize=8, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # Heatmap columns
        for col, cls_idx in enumerate(class_cols):
            ax_h = heat_axes[col][row]
            heat_rgba = shapley_to_heatmap(
                phi[row, :, cls_idx], image_size, grid_size, grid_size, cmap
            )
            ax_h.imshow(grey_rgba)
            ax_h.imshow(heat_rgba)

            prob_str = (
                f"{grand_probs[row, cls_idx]:.2f}"
                if grand_probs is not None
                else None
            )
            if row == 0:
                title = class_names[cls_idx]
                if prob_str is not None:
                    title += f"\n{prob_str}"
            else:
                title = prob_str
            if title is not None:
                ax_h.set_title(title, fontsize=8, pad=4)

            ax_h.set_xticks([])
            ax_h.set_yticks([])
            for spine in ax_h.spines.values():
                spine.set_linewidth(0.5)

    return fig
