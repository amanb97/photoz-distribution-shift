import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from tkinter.font import names

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr, ndtri


def mixture_cdf(z, means, scales, weights):
    """One CDF value per row; z has shape (N,) and parameters (N, K)."""
    return np.sum(weights * ndtr((np.asarray(z)[..., None] - means) / scales), axis=-1)


def mixture_quantile(q, means, scales, weights):
    """Invert each mixture CDF by bisection, using guaranteed component bounds."""
    if not 0 < q < 1:
        raise ValueError("Quantile probability must lie strictly between 0 and 1.")
    component_quantiles = means + scales * ndtri(q)
    lower = component_quantiles.min(axis=1)
    upper = component_quantiles.max(axis=1)
    for _ in range(50):
        middle = lower + (upper - lower) / 2
        below = mixture_cdf(middle, means, scales, weights) < q
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    return lower + (upper - lower) / 2


def expected_absolute_normal(delta, scale):
    """E[|X|] for X ~ Normal(delta, scale**2)."""
    standardized = delta / scale
    density = np.exp(-0.5 * standardized**2) / np.sqrt(2 * np.pi)
    return 2 * scale * density + delta * (2 * ndtr(standardized) - 1)


def mixture_crps(y, means, scales, weights):
    """Analytic CRPS = E|X-y| - 0.5 E|X-X'|, with independent X and X'."""
    first = np.sum(weights * expected_absolute_normal(y[:, None] - means, scales), axis=1)
    differences = means[:, :, None] - means[:, None, :]
    pair_scales = np.sqrt(scales[:, :, None]**2 + scales[:, None, :]**2)
    pair_weights = weights[:, :, None] * weights[:, None, :]
    second = np.sum(pair_weights * expected_absolute_normal(differences, pair_scales), axis=(1, 2))
    score = first - 0.5 * second
    if not np.isfinite(score).all() or np.any(score < -1e-8):
        raise FloatingPointError("Invalid CRPS: check predicted mixture parameters.")
    return np.maximum(score, 0)  # Remove only floating-point roundoff below zero.


def evaluate_split(model, loader, device, name):
    import torch
    from tqdm.auto import tqdm

    collected = {key: [] for key in ( "X",
        "y_true", "means", "scales", "weights", "nll", "z_mean", "z_std",
        "pit", "crps", "lower68", "upper68", "lower95", "upper95", "negative_z_mass",
    )}
    model.eval()
    with torch.inference_mode():
        for features, targets in tqdm(loader, desc=f"Evaluating {name}"):
            if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
                raise ValueError(f"{name}: non-finite input values; evaluation stopped.")
            features = features.to(device)
            targets = targets.to(device)
            distribution = model.distribution(features)
            # Calculate NLL in float64 as very narrow/wrong PDFs can occur under shift.
            components = distribution.component_distribution
            means = components.loc.double()
            scales = components.scale.double()
            log_weights = distribution.mixture_distribution.logits.double()
            log_weights = torch.log_softmax(log_weights, dim=-1)
            log_components = torch.distributions.Normal(means, scales).log_prob(targets.double()[:, None])
            nll = -torch.logsumexp(log_weights + log_components, dim=-1)
            means = means.cpu().numpy()
            scales = scales.cpu().numpy()
            weights = log_weights.exp().cpu().numpy()
            weights /= weights.sum(axis=1, keepdims=True)
            y = targets.double().cpu().numpy()
            if not (np.isfinite(means).all() and np.isfinite(scales).all()
                    and np.isfinite(weights).all() and np.all(scales > 0)
                    and torch.isfinite(nll).all().item()):
                raise FloatingPointError(f"{name}: non-finite or invalid model predictions.")

            z_mean = np.sum(weights * means, axis=1)
            variance = np.sum(weights * (scales**2 + (means - z_mean[:, None])**2), axis=1)
            values = dict(
                X=features.cpu().numpy(),
                y_true=y, means=means, scales=scales, weights=weights,
                nll=nll.cpu().numpy(), z_mean=z_mean, z_std=np.sqrt(variance),
                pit=mixture_cdf(y, means, scales, weights),
                crps=mixture_crps(y, means, scales, weights),
                negative_z_mass=mixture_cdf(np.zeros_like(y), means, scales, weights),
            )
            for key, q in (("lower68", 0.16), ("upper68", 0.84),
                           ("lower95", 0.025), ("upper95", 0.975)):
                values[key] = mixture_quantile(q, means, scales, weights)
            for key, value in values.items():
                collected[key].append(value)
    return {key: np.concatenate(parts, axis=0) for key, parts in collected.items()}


def summarize(name, predictions):
    p = predictions
    residual = p["z_mean"] - p["y_true"]
    return {
        "split": name, "n_galaxies": len(residual),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "mean_nll": float(np.mean(p["nll"])),
        "median_nll": float(np.median(p["nll"])),
        "p95_nll": float(np.percentile(p["nll"], 95)),
        "p99_nll": float(np.percentile(p["nll"], 99)),
        "max_nll": float(np.max(p["nll"])),
        "mean_crps": float(np.mean(p["crps"])),
        "coverage68": float(np.mean((p["y_true"] >= p["lower68"]) & (p["y_true"] <= p["upper68"]))),
        "coverage95": float(np.mean((p["y_true"] >= p["lower95"]) & (p["y_true"] <= p["upper95"]))),
        "mean_width68": float(np.mean(p["upper68"] - p["lower68"])),
        "mean_width95": float(np.mean(p["upper95"] - p["lower95"])),
        "mean_negative_z_mass": float(np.mean(p["negative_z_mass"])),
    }


def save_plots(results, summaries, output_dir, seed):
    names = list(results)
    n_splits = len(names)
    colors = ["tab:purple", "tab:blue", "tab:orange", "tab:green"]
    summary_by_name = {row["split"]: row for row in summaries}

    fig, axes = plt.subplots(
    1, n_splits,
    figsize=(4 * n_splits, 4),
    sharex=True,
    sharey=True,
    )
    for ax, name, color in zip(axes, names, colors):
        ax.hist(results[name]["pit"], bins=np.linspace(0, 1, 21), density=True,
                histtype="step", linewidth=1.8, color=color)
        ax.axhline(1, color="black", linestyle="--", linewidth=1, label="Uniform reference")
        ax.set(title=f"Teddy {name}", xlabel="PIT = F(z_spec)", xlim=(0, 1))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(output_dir / "pit_histograms.png", dpi=180)
    plt.close(fig)

    # Save both the complete view and a common zoomed view.
    for zoomed in (False, True):
        fig, axes = plt.subplots(
            1, n_splits,
            figsize=(4.5 * n_splits, 4),
        )

        for ax, name in zip(axes, names):
            p = results[name]
            x = p["y_true"]
            y = p["z_mean"]

            if zoomed:
                limits = (0.0, 1.5)
            else:
                low = min(x.min(), y.min())
                high = max(x.max(), y.max())
                margin = max((high - low) * 0.03, 0.01)
                limits = (low - margin, high + margin)

            inside = (
            (x >= limits[0]) & (x <= limits[1])
            & (y >= limits[0]) & (y <= limits[1])
            )
            n_outside = int((~inside).sum())

            if inside.any():
                counts = ax.hexbin(
                    x[inside],
                    y[inside],
                    gridsize=65,
                    extent=(
                        limits[0], limits[1],
                        limits[0], limits[1],
                    ),
                    cmap="Blues",
                    bins="log",
                    mincnt=1,
                )
                fig.colorbar(
                    counts, ax=ax,
                    label="Count (log colour scale)",
                )

            ax.plot(limits, limits, "r--", linewidth=1)
            ax.set(
                title=f"Teddy {name}",
                xlabel="Spectroscopic redshift",
                ylabel="Predicted PDF mean",
                xlim=limits,
                ylim=limits,
            )
            ax.set_aspect("equal", adjustable="box")

            if zoomed:
                ax.text(
                    0.03, 0.97,
                    f"{n_outside:,} rows outside plot range",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",              
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.85),
              )

        fig.tight_layout()
        filename = (
            "predicted_vs_true_zoomed.png"
            if zoomed else "predicted_vs_true.png"
        )
    
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, level in zip(axes, (68, 95)):
        ax.bar( names,[100 * summary_by_name[name][f"coverage{level}"] for name in names],
        color=colors[:n_splits],)
        ax.axhline(level, color="black", linestyle="--", label=f"Target: {level}%")
        ax.set(title=f"{level}% equal-tailed intervals", xlabel="Teddy split", ylim=(0, 103))
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Galaxies inside predicted interval (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "interval_coverage.png", dpi=180)
    plt.close(fig)

    # Random examples selected independently of their prediction quality.
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(
    n_splits, 3,
    figsize=(13, 3.3 * n_splits),
    squeeze=False,
    )
    example_rows = {}
    for row, name in enumerate(names):
        p = results[name]
        indices = np.sort(rng.choice(len(p["y_true"]), size=min(3, len(p["y_true"])), replace=False))
        example_rows[name] = indices.tolist()
        for col, ax in enumerate(axes[row]):
            if col >= len(indices):
                ax.set_visible(False)
                continue
            index = indices[col]
            mu, sigma, weight = (p[key][index] for key in ("means", "scales", "weights"))
            low = min(p["y_true"][index], mixture_quantile(0.001, mu[None], sigma[None], weight[None])[0])
            high = max(p["y_true"][index], mixture_quantile(0.999, mu[None], sigma[None], weight[None])[0])
            margin = max((high - low) * 0.05, 0.005)
            low, high = low - margin, high + margin
            # Extra local grid points resolve narrow mixture peaks.
            local = (mu[:, None] + sigma[:, None] * np.linspace(-5, 5, 151)).ravel()
            grid = np.unique(np.concatenate([np.linspace(low, high, 1200), local[(local >= low) & (local <= high)]]))
            density = np.sum(weight * np.exp(-0.5 * ((grid[:, None] - mu) / sigma)**2)
                             / (sigma * np.sqrt(2 * np.pi)), axis=1)
            ax.plot(grid, density, color=colors[row], label="Predicted PDF")
            ax.axvspan(p["lower68"][index], p["upper68"][index], color=colors[row], alpha=0.12, label="68% interval")
            ax.axvline(p["y_true"][index], color="black", linestyle="--", label="True redshift")
            ax.set(title=f"Teddy {name}: row {index}", xlabel="Redshift", ylabel="Density", xlim=(low, high))
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "example_pdfs.png", dpi=180)
    plt.close(fig)
    (output_dir / "example_rows.json").write_text(json.dumps(example_rows, indent=2), encoding="utf-8")


def main():
    import torch
    from torch.utils.data import DataLoader
    from dataloader import TeddyDataset
    from model import MDN

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate MDN accuracy and uncertainty on Teddy B/C/D.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42, help="Seed for random PDF examples only.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.seed < 0:
        parser.error("Batch size must be positive and seed nonnegative.")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    split_files = {
    "A_val": "val.npz",
    "B": "test_B.npz",
    "C": "test_C.npz",
    "D": "test_D.npz",
    }

    for filename in split_files.values():
        if not (args.data_dir / filename).is_file():
            parser.error(f"Missing {filename} in {args.data_dir}")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu.")
    device = torch.device(device_name)

    # Load the checkpoint from train.py; model dimensions come from that file.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = MDN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stamp = datetime.now().strftime("evaluation_%Y%m%d_%H%M%S_%f")
    output_dir = args.output_dir or args.checkpoint.resolve().parent.parent / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(output_dir / "evaluation.log", encoding="utf-8"),
    ])
    logging.info("Loaded epoch %s | device %s", checkpoint["epoch"], device)
    logging.info("Output directory: %s", output_dir.resolve())
    config = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_epoch=checkpoint["epoch"],
                  model_config=checkpoint["model_config"], data_dir=str(args.data_dir.resolve()),
                  batch_size=args.batch_size, seed=args.seed, device=str(device),
                  torch_version=str(torch.__version__), intervals="central equal-tailed",
                  redshift_transform="none; original Gaussian mixtures, no truncation",
                  split_policy="All supplied rows retained, including repeated IDs.")
    (output_dir / "evaluation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results, summaries = {}, []
    for name, filename in split_files.items():
        data = TeddyDataset(args.data_dir / filename)
        if len(data) == 0:
            raise ValueError(f"Teddy {name} is empty.")
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        p = evaluate_split(model, loader, device, name)
        p["ids"] = np.asarray(data.ids).astype(str)
        if len(p["ids"]) != len(p["y_true"]):
            raise ValueError(f"{name}: ID/prediction length mismatch.")
        np.savez_compressed(output_dir / f"predictions_{name}.npz", **p)
        summary = summarize(name, p)
        summaries.append(summary)
        results[name] = p
        # Save completed splits immediately, even if a later split fails.
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary))
            writer.writeheader()
            writer.writerows(summaries)
        logging.info(
            "%s | N=%d | RMSE %.5f | NLL %.4f | CRPS %.5f | coverage68 %.2f%% | coverage95 %.2f%%",
            name, summary["n_galaxies"], summary["rmse"], summary["mean_nll"],
            summary["mean_crps"], 100 * summary["coverage68"], 100 * summary["coverage95"],
        )
        # Compare evaluation with the validation values saved at training.
    val_summary = next(
        row for row in summaries if row["split"] == "A_val"
    )
    for metric, checkpoint_key in (
        ("mean_nll", "val_nll"),
        ("rmse", "val_rmse"),
    ):
        if checkpoint_key in checkpoint:
            saved_value = float(checkpoint[checkpoint_key])
            evaluated_value = val_summary[metric]
            logging.info(
                "A_val %s | saved %.8f | evaluated %.8f | difference %+.3e",
                metric,
                saved_value,
                evaluated_value,
                evaluated_value - saved_value,
            )


    p = results["D"]
    X = p["X"]  # Already standardised using the training-fitted scaler.
    max_abs_feature = np.max(np.abs(X), axis=1)
    worst_rows = np.argsort(p["nll"])[::-1][:50]

    feature_fields = [
        f"standardised_feature_{j}" for j in range(X.shape[1])
    ]
    fields = [
        "row_index", "id", "y_true", "z_mean", "nll",
        "pit", "z_std", "max_abs_standardised_feature",
        "means", "scales", "weights",
    ] + feature_fields

    with (output_dir / "D_worst_nll.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        for i in worst_rows:
            row = {
                "row_index": int(i),
                "id": str(p["ids"][i]),
                "y_true": float(p["y_true"][i]),
                "z_mean": float(p["z_mean"][i]),
                "nll": float(p["nll"][i]),
                "pit": float(p["pit"][i]),
                "z_std": float(p["z_std"][i]),
                "max_abs_standardised_feature": float(max_abs_feature[i]),
                "means": json.dumps(p["means"][i].tolist()),
                "scales": json.dumps(p["scales"][i].tolist()),
                "weights": json.dumps(p["weights"][i].tolist()),
            }
            row.update({
                field: float(X[i, j])
                for j, field in enumerate(feature_fields)
            })
            writer.writerow(row)


    threshold = 5.0
    keep = max_abs_feature <= threshold

    filter_info = {
        "analysis": "exploratory sensitivity analysis",
        "rule": "retain rows where max(abs(standardised features)) <= 5",
        "threshold": threshold,
        "n_total": int(len(keep)),
        "n_retained": int(keep.sum()),
        "n_removed": int((~keep).sum()),
        "fraction_removed": float((~keep).mean()),
    }

    (output_dir / "D_filter_info.json").write_text(
        json.dumps(filter_info, indent=2),
        encoding="utf-8",
    )

    np.savez_compressed(
        output_dir / "D_feature_filter.npz",
        ids=p["ids"],
        keep=keep,
        max_abs_standardised_feature=max_abs_feature,
    )

    if keep.any():
        filtered = {
            key: values[keep] for key, values in p.items()
        }
        summaries.append(
            summarize("D_feature_cut_5sigma", filtered)
        )
    else:
        logging.warning("D feature cut retained no rows.")

    logging.info(
        "Exploratory D feature cut: removed %d/%d rows (%.2f%%)",
        filter_info["n_removed"],
        filter_info["n_total"],
        100 * filter_info["fraction_removed"],
    )


    with (output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(summaries[0])
        )
        writer.writeheader()
        writer.writerows(summaries)
    save_plots(results, summaries, output_dir, args.seed)
    logging.info("Finished. Metrics, predictions and four plots saved to %s", output_dir.resolve())
    

if __name__ == "__main__":
    print("NEWNEW")
    main()
