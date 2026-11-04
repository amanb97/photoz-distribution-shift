import argparse
import csv
import json
import logging
import math
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots without opening a window.
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataloader import TeddyDataset
from model import MDN, mdn_loss


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train a probabilistic redshift MDN.")
    parser.add_argument("--data_dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--output_dir", type=Path, default=root / "experiments")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--min_sigma", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    for name in ("num_epochs", "batch_size", "hidden_dim", "n_components", "patience"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1.")
    for name in ("learning_rate", "min_sigma"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name} must be finite and positive.")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1.")
    return args


def set_seed(seed):
    """Control common randomness sources; hardware can still affect results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_nll = 0.0
    total_galaxies = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for features, redshifts in progress:
        features = features.to(device)
        redshifts = redshifts.to(device)  # Shape (batch,), not (batch, 1).

        optimizer.zero_grad(set_to_none=True)
        distribution = model.distribution(features)
        loss = mdn_loss(distribution, redshifts)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite training NLL; check data and model outputs.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=5.0, error_if_nonfinite=True
        )
        optimizer.step()

        #eight by batch size so the shorter final batch counts correctly.
        batch_size = redshifts.shape[0]
        total_nll += loss.item() * batch_size
        total_galaxies += batch_size
        progress.set_postfix(nll=f"{total_nll / total_galaxies:.4f}")

    return total_nll / total_galaxies


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_nll = 0.0
    total_squared_error = 0.0
    total_galaxies = 0

    for features, redshifts in loader:
        features = features.to(device)
        redshifts = redshifts.to(device)

        distribution = model.distribution(features)
        loss = mdn_loss(distribution, redshifts)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite validation NLL.")

        # The mixture mean is sum(weight_j * mean_j): a point estimate only.
        residuals = distribution.mean.double() - redshifts.double()
        total_squared_error += residuals.square().sum().item()
        batch_size = redshifts.shape[0]
        total_nll += loss.item() * batch_size
        total_galaxies += batch_size

    rmse = math.sqrt(total_squared_error / total_galaxies)
    if not math.isfinite(rmse):
        raise FloatingPointError("Non-finite validation RMSE.")
    return total_nll / total_galaxies, rmse


def plot_history(history, output_path):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [r["train_nll"] for r in history], label="Train")
    axes[0].plot(epochs, [r["val_nll"] for r in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Mean NLL", title="PDF training loss (lower is better)")
    axes[0].legend()
    axes[1].plot(epochs, [r["val_rmse"] for r in history])
    axes[1].set(xlabel="Epoch", ylabel="Redshift RMSE", title="Validation: PDF mean")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or --device auto.")
    device = torch.device(device_name)

    train_data = TeddyDataset(args.data_dir / "train.npz")
    val_data = TeddyDataset(args.data_dir / "val.npz")
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("Both training and validation splits must be nonempty.")

    # num_workers=0 keeps this small, in-memory dataset simple on Windows.
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed), drop_last=False,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False,
    )

    run_dir = args.output_dir / datetime.now().strftime("mdn_%Y%m%d_%H%M%S_%f")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(run_dir / "training.log", encoding="utf-8"),
                  logging.StreamHandler()],
    )
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(actual_device=str(device), torch_version=str(torch.__version__),
                  train_galaxies=len(train_data), val_galaxies=len(val_data))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    logging.info("Device: %s | Train: %d | Validation: %d", device, len(train_data), len(val_data))
    logging.info("Saving run to %s", run_dir.resolve())

    model_config = dict(input_dim=5, hidden_dim=args.hidden_dim,
                        n_components=args.n_components, min_sigma=args.min_sigma)
    model = MDN(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Absolute comparisons also behave correctly when NLL is negative.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4,
        threshold=0.0, threshold_mode="abs", min_lr=min(1e-6, args.learning_rate),
    )

    best_val_nll = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    fields = ["epoch", "train_nll", "val_nll", "val_rmse", "learning_rate", "seconds"]
    best_path = checkpoint_dir / "best_model.pt"

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.num_epochs + 1):
            start = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            train_nll = train_one_epoch(model, train_loader, optimizer, device)
            val_nll, val_rmse = validate(model, val_loader, device)
            scheduler.step(val_nll)

            improved = val_nll < best_val_nll
            if improved:
                best_val_nll = val_nll
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save({
                    "epoch": epoch,
                    "model_config": model_config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_nll": val_nll,
                    "val_rmse": val_rmse,
                    "config": config,
                }, best_path)
            else:
                epochs_without_improvement += 1

            row = dict(epoch=epoch, train_nll=train_nll, val_nll=val_nll,
                       val_rmse=val_rmse, learning_rate=learning_rate,
                       seconds=time.perf_counter() - start)
            history.append(row)
            writer.writerow(row)
            history_file.flush()
            logging.info(
                "Epoch %03d | train NLL %.4f | val NLL %.4f | val RMSE %.4f | lr %.2e%s",
                epoch, train_nll, val_nll, val_rmse, learning_rate,
                " | saved best" if improved else "",
            )
            if epochs_without_improvement >= args.patience:
                logging.info("Early stopping: %d epochs without lower validation NLL.", args.patience)
                break

    plot_history(history, run_dir / "learning_curves.png")
    logging.info("Best epoch: %d | validation NLL: %.4f", best_epoch, best_val_nll)
    logging.info("Use this saved checkpoint for later evaluation: %s", best_path.resolve())


if __name__ == "__main__":
    main()
