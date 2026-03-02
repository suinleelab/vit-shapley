"""Explanation baseline methods for ViT-Shapley (Stage 4).

Submodules:
    utils       — mask generation and explanation utilities (pure numpy)
    attention   — attention rollout / raw attention baselines
    perturbation — leave-one-out and RISE perturbation baselines
    gradient    — gradient-based baselines (requires captum)
"""
