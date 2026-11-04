from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

#set paths
save_directory = Path(r"C:\Users\aman\Desktop\Redshift\data\processed")
data_directory = Path(r"C:\Users\aman\Desktop\Redshift\data\raw")

#split params
split_seed = 42
validation_fraction = 0.2

#defining column ids
columns = [
    "id", "mag_r", "u_g", "g_r", "r_i", "i_z", "z_spec",
    "feat1", "feat2", "feat3", "feat4", "feat5",
]

FEATURES = ["mag_r", "u_g", "g_r", "r_i", "i_z"]
TARGET = "z_spec"


#loading data into a pandas dataframe
def load_catalogue(path):
    """Load one Teddy sample and check the required measurements."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        dtype=str,
    )
    #keep ids as strings and values as floats

    if df.shape[1] != len(columns):
        raise ValueError(f"{path.name}: expected 12 columns, found {df.shape[1]}")
    df.columns = columns
    for column in columns[1:]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    #now check for invalid measurements/missing values
    values = df[FEATURES + [TARGET]].to_numpy()

    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: missing or non-finite measurements")

    if df["id"].isna().any():
        raise ValueError(f"{path.name}: missing IDs")

    duplicate_ids = df["id"].duplicated(keep=False)

    if duplicate_ids.any():
        n_objects = df.loc[duplicate_ids, "id"].nunique()
        print( f"Warning: {path.name} contains {n_objects} repeated ID(s) "
        f"across {duplicate_ids.sum()} rows. Keeping original rows.")

    print(f"{path.name}: loaded {len(df):,} galaxies")
    return df

#creating histograms for each feature
def inspect_catalogues(catalogues):
    columns = FEATURES + [TARGET]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    for ax, column in zip(axes.flat, columns):
        # Use identical bins for A, B, C and D.
        combined = np.concatenate([
            df[column].to_numpy() for df in catalogues.values()
        ])
        bins = np.histogram_bin_edges(combined, bins=60)

        for name, df in catalogues.items():
            ax.hist(
                df[column],
                bins=bins,
                density=True,
                histtype="step",
                label=name,
            )

        ax.set_xlabel(column)
        ax.set_ylabel("Density")

    axes.flat[0].legend()
    fig.tight_layout()
    return fig

def prepare_splits(catalogues):
    #splitting sample A into training and validation sets,
        #while keeping samples B, C, and D as test sets.
    train, val = train_test_split(
        catalogues["A"],
        test_size=validation_fraction,
        random_state=split_seed,
    )
    
    #Scaling
    scaler = StandardScaler()
    scaler.fit(train[FEATURES])

    splits = {
        "train": train,
        "val": val,
        "test_B": catalogues["B"],
        "test_C": catalogues["C"],
        "test_D": catalogues["D"],
    }

    prepared = {}

    for name, df in splits.items():
        prepared[name] = {
            "X": scaler.transform(df[FEATURES]).astype(np.float32),
            "y": df[TARGET].to_numpy(dtype=np.float32),
            "ids": df["id"].to_numpy(dtype=str),
        }

    return prepared, scaler

def main():


    catalogues = {
        name: load_catalogue(data_directory / f"teddy_{name}")
        for name in "ABCD"
    }

    # Check that no training/validation galaxy occurs in a test sample.
    a_ids = set(catalogues["A"]["id"])

    for name in "BCD":
        overlap = a_ids.intersection(catalogues[name]["id"])
        if overlap:
            raise ValueError(
                f"A and {name} share {len(overlap)} galaxy IDs"
            )

    fig = inspect_catalogues(catalogues)
    fig.savefig(save_directory / "sample_distributions.png", dpi=160)
    plt.close(fig)

    prepared, scaler = prepare_splits(catalogues)

    for name, arrays in prepared.items():
        np.savez_compressed(save_directory / f"{name}.npz", **arrays)
        print(f"Saved {name}: X={arrays['X'].shape}, y={arrays['y'].shape}")

    joblib.dump(scaler, save_directory / "feature_scaler.joblib")


if __name__ == "__main__":
    main()