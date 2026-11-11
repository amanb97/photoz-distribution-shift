"""Pool evaluated Teddy MDNs into one predictive distribution per galaxy."""

import argparse
import sys
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib.ticker import MaxNLocator

from src.ensemble_prediction import pool_mixture
from scripts.evaluate import mixture_cdf, mixture_crps, mixture_quantile, summarize
import matplotlib.pyplot as plt


SPLITS = ("A_val", "B", "C", "D")
CUT_LABEL = "D_feature_cut_5sigma"
INTERVALS = (("lower68", .16), ("upper68", .84), ("lower95", .025), ("upper95", .975))
FIELDS = ("ids", "X", "y_true", "means", "scales", "weights", "nll", "z_mean",
          "crps", "pit", "lower68", "upper68", "lower95", "upper95", "negative_z_mass")


def check_rows(reference, other):
    # Keep repeated catalogue IDs. All three arrays must agree in the original order.
    for key in ("ids", "X", "y_true"):
        if not np.array_equal(reference[key], other[key]):
            raise ValueError(f"Members disagree on {key}: check processed data and row order.")


def read_predictions(folder, split, expected_k):
    with np.load(folder / f"predictions_{split}.npz", allow_pickle=False) as data:
        p = {key: data[key] for key in FIELDS}
    n = len(p["y_true"])
    if n == 0 or p["X"].shape != (n, 5) or p["ids"].shape != (n,):
        raise ValueError(f"{folder}/{split}: invalid or empty row data.")
    for key in FIELDS:
        if key == "ids":
            continue
        expected = ((n, expected_k) if key in ("means", "scales", "weights")
                    else (n, 5) if key == "X" else (n,))
        if p[key].shape != expected or not np.isfinite(p[key]).all():
            raise ValueError(f"{folder}/{split}: invalid {key}.")
    if (np.any(p["scales"] <= 0) or np.any(p["weights"] < 0)
            or not np.allclose(p["weights"].sum(axis=1), 1, rtol=1e-7, atol=1e-10)):
        raise ValueError(f"{folder}/{split}: invalid mixture parameters.")
    return p


def combine_batch(y, means, scales, weights, nll):
    """Pool each galaxy's PDFs, then score the reference redshift under that distribution."""
    p = pool_mixture(means, scales, weights)
    mu, sigma, w = p["means"], p["scales"], p["weights"]
    # Each input NLL was evaluated from component log-weights in float64.
    p["nll"] = math.log(nll.shape[1]) - logsumexp(-nll, axis=1)
    p["crps"] = mixture_crps(y, mu, sigma, w)
    p["pit"] = mixture_cdf(y, mu, sigma, w)
    p["negative_z_mass"] = mixture_cdf(np.zeros_like(y), mu, sigma, w)
    for key, q in INTERVALS:
        p[key] = mixture_quantile(q, mu, sigma, w)
    if any(not np.isfinite(value).all() for value in p.values()):
        raise FloatingPointError("Non-finite ensemble prediction or score.")
    return p


def pool_predictions(members, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if len(members) < 2:
        raise ValueError("Use at least two independently trained members.")
    ref = members[0]
    for member in members[1:]:
        check_rows(ref, member)
    y = ref["y_true"]
    pooled = {key: ref[key] for key in ("ids", "X", "y_true")}
    for start in range(0, len(y), batch_size):
        rows = slice(start, start + batch_size)
        # Score in batches: mixture CRPS needs each pair of components on each row.
        inputs = {key: np.stack([p[key][rows] for p in members], axis=1).astype(np.float64, copy=False)
                  for key in ("means", "scales", "weights", "nll")}
        batch = combine_batch(y[rows], **inputs)
        for key, values in batch.items():
            if start == 0:
                pooled[key] = np.empty((len(y), *values.shape[1:]), dtype=values.dtype)
            pooled[key][rows] = values
    return pooled


def variance_summary(split, p):
    return {
        "split": split, "n_galaxies": len(p["y_true"]),
        "mean_aleatoric_var": p["aleatoric_var"].mean(),
        "mean_epistemic_var": p["epistemic_var"].mean(),
        "mean_total_var": p["total_var"].mean(),
        "mean_aleatoric_std": p["aleatoric_std"].mean(),
        "mean_epistemic_std": p["epistemic_std"].mean(),
        "median_epistemic_std": np.median(p["epistemic_std"]),
        "mean_predictive_std": p["z_std"].mean(),
    }


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_validation(member, row):
    for key, saved in (("mean_nll", "val_nll"), ("rmse", "val_rmse")):
        if not np.isclose(row[key], member[saved], rtol=2e-4, atol=2e-5):
            raise ValueError(
                f"Seed {member['seed']} A_val {key}: evaluated {row[key]}, "
                f"checkpoint {member[saved]}. Check that you used the same processed files."
            )


def plot_results(out, pits, variances, baseline):
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(14, 3.7), sharex=True, sharey=True)
    for ax, split in zip(axes, SPLITS):
        for values, label, colour in zip(pits[split], (f"Seed {baseline}", "Ensemble"), ("0.4", "tab:blue")):
            ax.hist(values, bins=np.linspace(0, 1, 21), density=True,
                    histtype="step", color=colour, linewidth=1.6, label=label)
        ax.axhline(1, color="k", ls=":", lw=1)
        ax.set(title=split, xlabel="PIT", xlim=(0, 1))
    axes[0].set_ylabel("Density")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "pit_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    rows = [r for r in variances if r["split"] in SPLITS]
    within = np.array([r["mean_aleatoric_var"] for r in rows])
    between = np.array([r["mean_epistemic_var"] for r in rows])
    ax.bar(SPLITS, within, label="Predicted Uncertainty")
    ax.bar(SPLITS, between, bottom=within, label="Variance of member means")
    ax.set(ylabel="Mean variance per galaxy (redshift squared)", xlabel="Teddy split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "variance_by_split.png", dpi=160)
    plt.close(fig)


def write_diagnostics(out, ensemble):
    order = np.argsort(ensemble["nll"])[::-1][:50]
    rows = []
    for i in order:
        row = {key: ensemble[key][i] for key in
               ("y_true", "z_mean", "nll", "pit", "aleatoric_std", "epistemic_std", "z_std")}
        rows.append(dict(row_index=i, id=ensemble["ids"][i], **row,
                         max_abs_standardised_feature=np.abs(ensemble["X"][i]).max()))
    write_table(out / "D_worst_nll.csv", rows)


def select_rows(p, mask):
    # The scalar diagnostics suffice for subset summaries.
    return {key: values[mask] for key, values in p.items() if values.ndim == 1}


def metric_rows(label, split, p, keep):
    rows = [{"model": label, **summarize(split, p)}]
    if keep is not None and keep.any():
        rows.append({"model": label, **summarize(CUT_LABEL, select_rows(p, keep))})
    return rows


SCORES = {
    "epistemic_std": "Ensemble disagreement",
    "aleatoric_std": "Predicted Uncertainty",
    "max_abs_standardised_feature": "Feature Difference",
}


def completed_folders(directory, pattern):
    return sorted(p for p in directory.glob(pattern)
                  if (p / "ensemble.json").is_file() and (p / "predictions_D.npz").is_file())


def find_combined(path):
    if path is None:
        # Run and combined folder names contain sortable timestamps.
        experiments = Path(__file__).resolve().parents[1] / "experiments"
        choices = completed_folders(experiments, "ensemble_*/combined_*")
    else:
        path = path.resolve()
        if (path / "predictions_D.npz").is_file() and (path / "ensemble.json").is_file():
            return path
        choices = completed_folders(path, "combined_*")
    if not choices:
        raise ValueError("No completed ensemble results found. Pass the combined_... folder with --run-dir.")
    return choices[-1]


def saved_d_inputs(combined):
    manifest = json.loads((combined / "ensemble.json").read_text(encoding="utf-8"))
    seed = manifest["baseline_seed"]
    members = [m for m in manifest["members"] if m["seed"] == seed]
    if len(members) != 1:
        raise ValueError("This experiment requires exactly one member matching the baseline seed.")
    baseline_file = combined.parent / members[0]["evaluation_dir"] / "predictions_D.npz"
    ensemble_file = combined / "predictions_D.npz"
    print(f"Ensemble: {ensemble_file}\nBaseline: {baseline_file}", flush=True)

    common = ("ids", "X", "y_true", "lower95", "upper95")
    with np.load(baseline_file, allow_pickle=False) as f:
        baseline = {k: f[k] for k in common}
    with np.load(ensemble_file, allow_pickle=False) as f:
        ensemble = {k: f[k] for k in (*common, "epistemic_std", "aleatoric_std")}
        if not np.array_equal(f["member_seeds"], [m["seed"] for m in manifest["members"]]):
            raise ValueError("Saved ensemble seeds disagree with the manifest.")
    return baseline, ensemble, manifest


def diagnostic_rows(baseline, ensemble):
    check_rows(baseline, ensemble)
    n = len(ensemble["ids"])
    if not n or ensemble["ids"].shape != (n,) or ensemble["X"].shape != (n, 5):
        raise ValueError("Expected one ID and five standardised features per D row.")
    for data in (baseline, ensemble):
        for k in ("X", "y_true", "lower95", "upper95"):
            v = data[k]
            if (k != "X" and v.shape != (n,)) or not np.isfinite(v).all():
                raise ValueError(f"Invalid shape or non-finite values in {k}.")
        if np.any(data["upper95"] < data["lower95"]):
            raise ValueError("Reversed 95% interval endpoints.")
    for k in ("epistemic_std", "aleatoric_std"):
        v = ensemble[k]
        if v.shape != (n,) or not np.isfinite(v).all() or (v < 0).any():
            raise ValueError(f"Invalid {k} in the ensemble predictions.")

    y = ensemble["y_true"]
    return {
        "row_index": np.arange(n), "id": ensemble["ids"], "z_spec": y,
        "baseline_lower95": baseline["lower95"], "baseline_upper95": baseline["upper95"],
        "ensemble_lower95": ensemble["lower95"], "ensemble_upper95": ensemble["upper95"],
        "baseline_miss95": ((y < baseline["lower95"]) | (y > baseline["upper95"])).astype(int),
        "ensemble_miss95": ((y < ensemble["lower95"]) | (y > ensemble["upper95"])).astype(int),
        "epistemic_std": ensemble["epistemic_std"], "aleatoric_std": ensemble["aleatoric_std"],
        "max_abs_standardised_feature": np.max(np.abs(ensemble["X"]), axis=1),
    }


def auc_if_defined(misses, score):
    # One missed and one covered row are needed to form a ranking pair.
    if np.unique(misses).size < 2:
        return None
    return float(roc_auc_score(misses, score))


FIGURE_STYLE = {
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11, "axes.labelsize": 13, "axes.labelcolor": "#243447",
    "axes.labelpad": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9AA6B2", "xtick.color": "#46576A", "ytick.color": "#46576A",
    "axes.axisbelow": True, "grid.color": "#E4E9EE", "grid.linewidth": 0.8,
    "legend.frameon": False,
}
BLUE, ORANGE, GREEN = "#2168A0", "#C26118", "#188268"


def plot_roc(out, data, seed):
    missed = data["baseline_miss95"]
    n_missed = int(missed.sum())
    both_classes = 0 < n_missed < len(missed)
    rows = []
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(7.4, 6), layout="constrained")
        fig.suptitle("ROC curve", fontsize=20, fontweight="semibold")
        ax.set_title(f"Teddy D", fontsize=11, pad=10, color="#46576A")
        ax.grid(axis="both")
        ax.plot([0, 1], [0, 1], "--", color="#89949C", linewidth=1.2)
        styles = ((BLUE, "-"), (ORANGE, "--"), (GREEN, "-."))
        for (key, label), (color, style) in zip(SCORES.items(), styles):
            score = data[key]
            auc = auc_if_defined(missed, score)
            rows.append(dict(baseline_seed=seed, score=key, auroc=auc, n_rows=len(missed),
                             n_baseline_misses=n_missed, n_baseline_covered=len(missed)-n_missed))
            if both_classes:
                fpr, tpr, _ = roc_curve(missed, score, pos_label=1)
                ax.plot(fpr, tpr, color=color, linestyle=style, lw=2.6,
                        label=f"{label}  (AUC {auc:.3f})")
                print(f"{key:32s} AUROC {auc:.4f}")
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               xlim=(0, 1), ylim=(0, 1.01), aspect="equal")
        if both_classes:
            ax.legend(loc="lower right", fontsize=10, frameon=True, facecolor="white",
                      edgecolor="none", framealpha=.95)
        else:
            ax.text(.5, .7, "AUROC undefined: only one outcome observed",
                    ha="center", transform=ax.transAxes)
            print("AUROC undefined: all baseline intervals covered or all missed.")
        fig.savefig(out / "auroc_D.png", dpi=200)
        plt.close(fig)
    write_table(out / "auroc_D.csv", rows)


def wilson95(covered, n):
    # Pointwise 95% Wilson interval for the observed coverage fraction.
    z = 1.959963984540054
    p = covered / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n))/denom
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return max(0., centre-radius), min(1., centre+radius)


def coverage_bins(data, n_bins):
    distance = data["max_abs_standardised_feature"]
    if n_bins < 1:
        raise ValueError("--bins must be positive.")
    n_bins = min(n_bins, len(distance))
    cuts = np.unique(np.quantile(distance, np.linspace(0, 1, n_bins + 1)[1:-1]))
    cuts = cuts[(cuts > distance.min()) & (cuts < distance.max())]
    groups = np.searchsorted(cuts, distance, side="right")
    rows = []
    for g in np.unique(groups):
        use = groups == g
        d = distance[use]
        row = dict(bin=len(rows)+1, n_rows=len(d), min_distance=d.min(),
                   median_distance=np.median(d), max_distance=d.max(),
                   median_epistemic_std=np.median(data["epistemic_std"][use]))
        row["baseline_miss_epistemic_auroc"] = auc_if_defined(
            data["baseline_miss95"][use], data["epistemic_std"][use]
        )
        for model in ("baseline", "ensemble"):
            covered = int((data[f"{model}_miss95"][use] == 0).sum())
            lo, hi = wilson95(covered, len(d))
            row.update({f"{model}_covered": covered, f"{model}_coverage95": covered/len(d),
                        f"{model}_wilson_lo": lo, f"{model}_wilson_hi": hi})
        rows.append(row)
    return rows


def plot_coverage(out, rows, seed):
    # Plot each bin at its median feature distance on a numerical axis.
    x = np.array([r["median_distance"] for r in rows])
    with plt.rc_context(FIGURE_STYLE):
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 6.6), sharex=True,
                                      height_ratios=(2.2, 1), layout="constrained")
        fig.suptitle("95% interval coverage vs training distance", fontsize=20, fontweight="semibold")
        ax.set_title(f"Teddy D · single model (seed {seed}) and deep ensemble",
                     fontsize=11, pad=10, color="#46576A")
        for model, label, color, marker in (("baseline", "Single model", BLUE, "o"),
                                             ("ensemble", "Deep ensemble", ORANGE, "s")):
            pct = 100 * np.array([r[f"{model}_coverage95"] for r in rows])
            low = 100 * np.array([r[f"{model}_wilson_lo"] for r in rows])
            high = 100 * np.array([r[f"{model}_wilson_hi"] for r in rows])
            ax.errorbar(x, pct, yerr=np.maximum(0, [pct-low, high-pct]),
                        fmt=marker+"-", color=color, lw=2.3, elinewidth=1.2, capsize=3,
                        ms=5, label=label, zorder=3)
        ax.axhline(95, color="#56626F", ls="--", lw=1.4, label="Target coverage (95%)")
        ax.set(ylabel="Coverage (%)", ylim=(0, 102), yticks=[0, 20, 40, 60, 80, 100])
        handles, labels = ax.get_legend_handles_labels()
        legend = dict(zip(labels, handles))
        order = ["Single model", "Deep ensemble", "Target coverage (95%)"]
        ax.legend([legend[k] for k in order], order, loc="lower left", fontsize=10)
        ax2.plot(x, [r["median_epistemic_std"] for r in rows], "o-", lw=2.2, ms=5,
                 color="#76528D")
        ax2.set(ylabel="Median\ndisagreement", xlabel="Distance from training data", ylim=(0, None))
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=2))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
        for axis in (ax, ax2):
            axis.grid(axis="y")
            axis.margins(x=.04)
        fig.savefig(out / "coverage_vs_distance_D.png", dpi=200)
        plt.close(fig)


def training_z_ranges(manifest, train_npz):
    if train_npz is None:
        path = Path(manifest.get("data_dir", Path(__file__).resolve().parents[1] / "data" / "processed")) / "train.npz"
        if not path.is_file():
            print(f"Training targets unavailable at {path}. Range statistics skipped. "
                  "Use --train-npz PATH to supply the original processed training split.")
            return None
    else:
        path = Path(train_npz)
    expected = manifest.get("data_sha256", {}).get("train.npz")
    if expected:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"{path} differs from the train.npz recorded for this ensemble.")
    with np.load(path, allow_pickle=False) as f:
        y = np.asarray(f["y"], dtype=float)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all():
        raise ValueError(f"{path}: expected a nonempty, finite 1D training target array.")
    if manifest.get("train_galaxies", y.size) != y.size:
        raise ValueError(f"{path}: training row count differs from the manifest.")
    low, high = np.quantile(y, [.005, .995])
    return {"train_file": str(path.resolve()), "train_rows": y.size,
            "central99": (low, high), "full": (y.min(), y.max())}


def write_followup(out, data, rows, seed, ranges):
    distance = data["max_abs_standardised_feature"]
    epistemic = data["epistemic_std"]
    missed = data["ensemble_miss95"].astype(bool)
    rho = None
    if np.ptp(distance) > 0 and np.ptp(epistemic) > 0:
        rho = float(spearmanr(epistemic, distance)[0])
    valid = [r["baseline_miss_epistemic_auroc"] for r in rows
             if r["baseline_miss_epistemic_auroc"] is not None]
    summary = {"baseline_seed": seed, "n_D": len(distance), "n_ensemble_misses": int(missed.sum()),
               "spearman_epistemic_vs_distance": rho,
               "ensemble_miss_epistemic_auroc": auc_if_defined(missed, epistemic),
               "mean_within_bin_baseline_miss_epistemic_auroc": np.mean(valid) if valid else None,
               "n_distance_bins": len(rows), "n_bins_with_both_outcomes": len(valid),
               "train_file": ranges["train_file"] if ranges else None,
               "n_train": ranges["train_rows"] if ranges else None}
    # The denominator for share_misses_outside is ALL ensemble misses on full D.
    for label in ("central99", "full"):
        lo = hi = n_out = share_misses = share_all = None
        if ranges is not None:
            lo, hi = ranges[label]
            outside = (data["z_spec"] < lo) | (data["z_spec"] > hi)
            data[f"outside_train_{label}"] = outside.astype(int)
            n_out = int((outside & missed).sum())
            share_misses = n_out / int(missed.sum()) if missed.any() else None
            share_all = outside.mean()
        summary.update({f"train_{label}_low": lo, f"train_{label}_high": hi,
                        f"n_ensemble_misses_outside_{label}": n_out,
                        f"share_ensemble_misses_outside_{label}": share_misses,
                        f"share_all_D_outside_{label}": share_all})
    write_table(out / "D_followup.csv", [summary])
    for key in ("spearman_epistemic_vs_distance", "ensemble_miss_epistemic_auroc",
                "mean_within_bin_baseline_miss_epistemic_auroc", "share_ensemble_misses_outside_central99",
                "share_ensemble_misses_outside_full"):
        value = summary[key]
        print(f"{key}: {value:.4f}" if value is not None else f"{key}: unavailable")


def write_d_plots(out, baseline, ensemble, seed, bins, ranges=None):
    data = diagnostic_rows(baseline, ensemble)
    rows = coverage_bins(data, bins)
    write_followup(out, data, rows, seed, ranges)
    plot_roc(out, data, seed)
    plot_coverage(out, rows, seed)
    write_table(out / "coverage_vs_distance_D.csv", rows)
    with (out / "galaxy_diagnostics_D.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))
    for model in ("baseline", "ensemble"):
        print(f"{model} full-D 95% coverage: {1-data[f'{model}_miss95'].mean():.2%}")


def add_saved_diagnostics(path, bins, train_npz=None):
    combined = find_combined(path)
    baseline, ensemble, manifest = saved_d_inputs(combined)
    ranges = training_z_ranges(manifest, train_npz)
    dest = combined / datetime.now().strftime("diagnostics_D_%Y%m%d_%H%M%S_%f")
    with TemporaryDirectory(dir=combined) as tmp:
        out = Path(tmp) / "results"
        out.mkdir()
        write_d_plots(out, baseline, ensemble, manifest["baseline_seed"], bins, ranges)
        out.rename(dest)
    return dest


def evaluate_run(run_dir, batch_size=256, bins=10, train_npz=None):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "ensemble.json").read_text(encoding="utf-8"))
    ranges = training_z_ranges(manifest, train_npz)
    members = manifest["members"]
    seeds = [m["seed"] for m in members]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("An ensemble requires at least two distinct training seeds.")
    baseline = manifest["baseline_seed"]
    baseline_index = seeds.index(baseline)
    for m in members:
        if "evaluation_dir" not in m:
            raise ValueError(f"Seed {m['seed']} has no completed evaluation. Resume run_ensemble.py.")
        folder = run_dir / m["evaluation_dir"]
        cfg = json.loads((folder / "evaluation_config.json").read_text(encoding="utf-8"))
        if (cfg["checkpoint"] != m["checkpoint"] or cfg["checkpoint_epoch"] != m["epoch"]
                or cfg["model_config"] != manifest["model_config"]):
            raise ValueError(f"Seed {m['seed']}: evaluation metadata differs from the recorded checkpoint.")
    out = run_dir / datetime.now().strftime("combined_%Y%m%d_%H%M%S_%f")
    metrics, variances, pits = [], [], {}
    # Publish the directory only once all scores, arrays and plots are ready.
    with TemporaryDirectory(dir=run_dir) as work:
        pending = Path(work) / "results"
        pending.mkdir()
        for split in SPLITS:
            print(f"Combining {split}", flush=True)
            individual = [read_predictions(run_dir / m["evaluation_dir"], split,
                                           manifest["model_config"]["n_components"]) for m in members]
            p = pool_predictions(individual, batch_size)
            np.savez_compressed(pending / f"predictions_{split}.npz", member_seeds=seeds, **p)
            pits[split] = (individual[baseline_index]["pit"], p["pit"])
            keep = None
            if split == "D":
                keep = np.max(np.abs(p["X"]), axis=1) <= 5
                counts = dict(threshold=5.0, n_total=len(keep), n_retained=int(keep.sum()),
                              n_removed=int((~keep).sum()))
                (pending / "D_feature_cut.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
                np.savez_compressed(pending / "D_feature_filter.npz", ids=p["ids"], keep=keep)
                write_diagnostics(pending, p)
                write_d_plots(pending, individual[baseline_index], p, baseline, bins, ranges)
            for m, arrays in zip(members, individual):
                rows = metric_rows(f"seed_{m['seed']}", split, arrays, keep)
                if split == "A_val":
                    check_validation(m, rows[0])
                metrics.extend(rows)

            ensemble_rows = metric_rows("ensemble", split, p, keep)
            metrics.extend(ensemble_rows)
            variances.append(variance_summary(split, p))
            if keep is not None and keep.any():
                variances.append(variance_summary(CUT_LABEL, select_rows(p, keep)))
            row = ensemble_rows[0]
            print(f"  RMSE {row['rmse']:.5f} | NLL {row['mean_nll']:.4f} | "
                  f"CRPS {row['mean_crps']:.5f} | coverage95 {row['coverage95']:.2%}")
            del individual, arrays, p
        write_table(pending / "metrics.csv", metrics)
        write_table(pending / "uncertainty_summary.csv", variances)
        plot_results(pending, pits, variances, baseline)
        (pending / "ensemble.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        pending.rename(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Ensemble run folder (or a combined folder in diagnostics-only mode).")
    parser.add_argument("--batch-size", type=int, default=256, help="Rows per CRPS/quantile batch.")
    parser.add_argument("--diagnostics-only", action="store_true", help="Use saved D predictions. Defaults to the latest completed ensemble run.")
    parser.add_argument("--bins", type=int, default=10, help="Requested number of D distance bins.")
    parser.add_argument("--train-npz", type=Path, help="Original processed train.npz if the recorded data folder moved.")
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("--bins must be positive.")
    if args.diagnostics_only:
        print("Diagnostics:", add_saved_diagnostics(args.run_dir, args.bins, args.train_npz))
    elif args.run_dir is None:
        parser.error("Use --run-dir for full evaluation, or --diagnostics-only for a completed run.")
    else:
        print("Ensemble results:", evaluate_run(args.run_dir, args.batch_size, args.bins, args.train_npz))
