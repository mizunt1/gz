from transforming_vae_galaxies_test import get_datasets
import torch

if __name__ == "__main__":
    ds, _ = get_datasets()
    dl = torch.utils.data.DataLoader(ds, batch_size=2000)

    for batch in dl:
        x = batch[0]
        break
    mu = x.mean()
    std = x.std()
    print("MU", mu, "STD", std)
