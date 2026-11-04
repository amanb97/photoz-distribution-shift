from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# loading a processed teddy dataset from .npz file
class TeddyDataset(Dataset):
    def __init__(self, file_path):
        with np.load(file_path, allow_pickle=False) as data:

            #convert to tensors
            self.features = torch.from_numpy(
                data["X"].astype(np.float32)
            )
            self.redshifts = torch.from_numpy(
                data["y"].astype(np.float32)
            )
            #keep ids
            self.ids = data["ids"].copy()

        if self.features.shape != (len(self.redshifts), 5):
            raise ValueError("Expected five features per galaxy.")

    def __len__(self):
        return len(self.redshifts)

    #returns the features and redshifts for an index
    def __getitem__(self, idx):
        return self.features[idx], self.redshifts[idx]


if __name__ == "__main__":
    data_dir = Path(
        r"C:\Users\aman\Desktop\Redshift\data\processed"
    )

    train_dataset = TeddyDataset(data_dir / "train.npz")
    val_dataset = TeddyDataset(data_dir / "val.npz")

    #get into batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
    )

    #creating a validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    features, redshifts = next(iter(train_loader))

    print("Training galaxies:", len(train_dataset))
    print("Validation galaxies:", len(val_dataset))
    print("Feature batch:", features.shape)
    print("Redshift batch:", redshifts.shape)
    print("First galaxy features:", features[0])
    print("Its reference redshift:", redshifts[0])