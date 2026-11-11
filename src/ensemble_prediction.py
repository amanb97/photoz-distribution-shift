"""Equally weighted MDN predictions and per-galaxy variance decomposition."""

import json
from pathlib import Path

import numpy as np


def pool_mixture(means, scales, weights):
    """Combine [galaxies, members, components] arrays into one PDF per galaxy."""
    means, scales, weights = (np.asarray(v, dtype=np.float64) for v in (means, scales, weights))
    if means.ndim != 3 or means.shape[2] < 1:
        raise ValueError("Expected [galaxies, members, components] parameters.")
    n, m, k = means.shape
    if m < 2 or scales.shape != means.shape or weights.shape != means.shape:
        raise ValueError("Supply matching parameter arrays from at least two members.")
    if (not all(np.isfinite(v).all() for v in (weights, means, scales))
            or np.any(scales <= 0) or np.any(weights < 0)
            or not np.allclose(weights.sum(axis=2), 1, rtol=1e-7, atol=1e-10)):
        raise ValueError("A member returned invalid Gaussian mixture parameters.")
    weights = weights / weights.sum(axis=2, keepdims=True)

    # A member is itself a mixture. Its variance includes the spacing of its components.
    member_mean = np.sum(weights * means, axis=2)
    member_var = np.sum(weights * (scales**2 + (means - member_mean[:, :, None])**2), axis=2)
    z_mean = member_mean.mean(axis=1)
    aleatoric_var = member_var.mean(axis=1)
    epistemic_var = np.mean((member_mean - z_mean[:, None])**2, axis=1)
    total_var = aleatoric_var + epistemic_var

    # An equal mixture of M K-component MDNs is a single (M*K)-component MDN output.
    mu = means.reshape(n, m*k)
    sigma = scales.reshape(n, m*k)
    w = (weights / m).reshape(n, m*k)
    direct_var = np.sum(w * (sigma**2 + (mu - z_mean[:, None])**2), axis=1)
    if not np.allclose(total_var, direct_var, rtol=1e-10, atol=1e-12):
        raise ArithmeticError("Variance decomposition differs from the pooled mixture variance.")

    return {
        "means": mu, "scales": sigma, "weights": w, "z_mean": z_mean,
        "member_mean": member_mean, "member_var": member_var,
        "aleatoric_var": aleatoric_var, "epistemic_var": epistemic_var, "total_var": total_var,
        "aleatoric_std": np.sqrt(aleatoric_var), "epistemic_std": np.sqrt(epistemic_var),
        "z_std": np.sqrt(total_var),
    }


def load_models(run_dir, device="cpu"):
    """Load the seed-ordered best checkpoints once for prediction on new objects."""
    import torch
    from .model import MDN

    manifest = json.loads((Path(run_dir) / "ensemble.json").read_text(encoding="utf-8"))
    models = []
    for member in manifest["members"]:
        saved = torch.load(member["checkpoint"], map_location="cpu", weights_only=True)
        if saved["config"]["seed"] != member["seed"] or saved["model_config"] != manifest["model_config"]:
            raise ValueError(f"Wrong checkpoint for seed {member['seed']}.")
        model = MDN(**saved["model_config"])
        model.load_state_dict(saved["model_state_dict"])
        models.append(model.to(device).eval())
    return models


def predict_batch(models, X):
    """Predict from five already-standardised features per galaxy. No redshift labels needed."""
    import torch

    if len(models) < 2:
        raise ValueError("Prediction requires at least two ensemble members.")
    device = next(models[0].parameters()).device
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[1] != 5 or not torch.isfinite(X).all():
        raise ValueError("Expected a finite (N, 5) array of standardised Teddy features.")
    means, scales, weights = [], [], []
    with torch.inference_mode():
        for model in models:
            model.eval()
            pdf = model.distribution(X)
            means.append(pdf.component_distribution.loc.double().cpu().numpy())
            scales.append(pdf.component_distribution.scale.double().cpu().numpy())
            log_w = torch.log_softmax(pdf.mixture_distribution.logits.double(), dim=-1)
            weights.append(log_w.exp().cpu().numpy())
    return pool_mixture(np.stack(means, axis=1), np.stack(scales, axis=1), np.stack(weights, axis=1))
