"""Train independent Teddy MDNs, evaluate their checkpoints, and pool the PDFs."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SETTINGS = ("num_epochs", "batch_size", "learning_rate", "hidden_dim",
                  "n_components", "min_sigma", "patience")
DATA_FILES = ("train.npz", "val.npz", "test_B.npz", "test_C.npz", "test_D.npz")


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_manifest(path, manifest):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def checkpoint_info(path):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    return {
        "seed": int(config["seed"]), "epoch": int(checkpoint["epoch"]),
        "val_nll": float(checkpoint["val_nll"]), "val_rmse": float(checkpoint["val_rmse"]),
        "train_settings": {key: config[key] for key in TRAIN_SETTINGS},
        "train_galaxies": config["train_galaxies"], "val_galaxies": config["val_galaxies"],
        "model_config": checkpoint["model_config"],
    }


def register_checkpoint(manifest, member, path, info):
    if info["seed"] != member["seed"]:
        raise ValueError(f"{path}: recorded seed {info['seed']} differs from {member['seed']}.")
    for key, value in manifest["train_settings"].items():
        if info["train_settings"][key] != value:
            raise ValueError(f"{path}: training setting {key} differs from the ensemble.")
    for key in ("train_galaxies", "val_galaxies"):
        if info[key] != manifest[key]:
            raise ValueError(f"{path}: {key} does not match the supplied processed data.")
    if manifest.get("model_config", info["model_config"]) != info["model_config"]:
        raise ValueError(f"{path}: MDN architectures differ.")
    manifest["train_settings"] = info["train_settings"]
    manifest["model_config"] = info["model_config"]
    member.update(checkpoint=str(path.resolve()), checkpoint_sha256=checksum(path),
                  epoch=info["epoch"], val_nll=info["val_nll"], val_rmse=info["val_rmse"])


def start_run(args):
    seeds = args.seeds if args.seeds is not None else [42, 43, 44, 45, 46]
    if len(set(seeds)) != len(seeds) or len(seeds) < 2 or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError("Supply at least two distinct seeds in [0, 2**32).")
    if args.num_epochs is not None and args.num_epochs < 1:
        raise ValueError("--num-epochs must be at least 1.")
    data = (args.data_dir or ROOT / "data" / "processed").resolve()
    manifest = {
        "data_dir": str(data), "data_sha256": {name: checksum(data / name) for name in DATA_FILES},
        "device": args.device or "auto", "baseline_seed": seeds[0],
        "train_settings": {} if args.num_epochs is None else {"num_epochs": args.num_epochs},
        "members": [{"seed": seed} for seed in seeds],
    }
    for split, key in (("train", "train_galaxies"), ("val", "val_galaxies")):
        with np.load(data / f"{split}.npz", allow_pickle=False) as arrays:
            manifest[key] = len(arrays["y"])

    by_seed = {m["seed"]: m for m in manifest["members"]}
    for path in args.reuse_checkpoint or []:
        info = checkpoint_info(path)
        seed = info["seed"]
        if seed not in by_seed or "checkpoint" in by_seed[seed]:
            raise ValueError(f"{path}: seed {seed} is duplicated or outside the requested seed list.")
        register_checkpoint(manifest, by_seed[seed], path, info)

    run_dir = (args.output_dir or ROOT / "experiments").resolve() / f"ensemble_{stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_manifest(run_dir / "ensemble.json", manifest)
    return run_dir, manifest


def run_script(filename, args):
    subprocess.run([sys.executable, str(ROOT / "scripts" / filename), *map(str, args)], check=True, cwd=ROOT)


def run_members(run_dir, manifest):
    manifest_path = run_dir / "ensemble.json"
    for m in manifest["members"]:
        seed = m["seed"]
        member_dir = run_dir / f"seed_{seed}"
        if "checkpoint" not in m:
            # train.py creates its usual timestamped run inside this seed's directory.
            member_dir.mkdir(exist_ok=True)
            pattern = "mdn_*/checkpoints/best_model.pt"
            before = set(member_dir.glob(pattern))
            options = ["--seed", seed, "--data_dir", manifest["data_dir"], "--output_dir", member_dir,
                       "--device", manifest["device"]]
            for key, value in manifest["train_settings"].items():
                options.extend([f"--{key}", value])
            print(f"\nTraining seed {seed}", flush=True)
            run_script("train.py", options)
            created = set(member_dir.glob(pattern)) - before
            if len(created) != 1:
                raise RuntimeError(f"Expected one new best checkpoint under {member_dir}.")
            path = created.pop()
            register_checkpoint(manifest, m, path, checkpoint_info(path))
            save_manifest(manifest_path, manifest)
        elif checksum(Path(m["checkpoint"])) != m["checkpoint_sha256"]:
            raise ValueError(f"Seed {seed}: checkpoint contents changed since this run started.")
        else:
            print(f"Using seed {seed}, epoch {m['epoch']}: {m['checkpoint']}", flush=True)

    for m in manifest["members"]:
        if "evaluation_dir" in m:
            completed = run_dir / m["evaluation_dir"]
            expected = ["evaluation_config.json", "metrics.csv",
                        *(f"predictions_{name}.npz" for name in ("A_val", "B", "C", "D"))]
            if all((completed / filename).is_file() for filename in expected):
                continue
        dest = Path(f"seed_{m['seed']}") / f"eval_{stamp()}"
        print(f"\nEvaluating seed {m['seed']}", flush=True)
        run_script("evaluate.py", ["--checkpoint", m["checkpoint"], "--data_dir", manifest["data_dir"],
                                   "--device", manifest["device"], "--output_dir", run_dir / dest])
        m["evaluation_dir"] = dest.as_posix()
        save_manifest(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", help="Default: 42 43 44 45 46.")
    parser.add_argument("--reuse-checkpoint", type=Path, action="append",
                        help="Use a best_model.pt from the same processed data. Repeat for multiple seeds.")
    parser.add_argument("--num-epochs", type=int, help="For a fresh run, override train.py's maximum epochs.")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Parent of a new ensemble run (default: experiments).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--resume", type=Path, help="Resume an existing ensemble run directory.")
    args = parser.parse_args()
    if args.resume is None:
        run_dir, manifest = start_run(args)
    else:
        if any(v is not None for k, v in vars(args).items() if k != "resume"):
            parser.error("Use --resume by itself; the run's manifest supplies its settings.")
        run_dir = args.resume.resolve()
        manifest = json.loads((run_dir / "ensemble.json").read_text(encoding="utf-8"))
        for name, expected in manifest["data_sha256"].items():
            if checksum(Path(manifest["data_dir"]) / name) != expected:
                raise ValueError(f"{name} changed since this ensemble run started.")
    print(f"Ensemble run: {run_dir}", flush=True)
    print(f'To resume: "{sys.executable}" "{Path(__file__).resolve()}" --resume "{run_dir}"', flush=True)
    run_members(run_dir, manifest)
    from scripts.evaluate_ensemble import evaluate_run
    result = evaluate_run(run_dir)
    print(f"Ensemble results: {result}", flush=True)


if __name__ == "__main__":
    main()
