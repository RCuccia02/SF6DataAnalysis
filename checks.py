import pandas as pd
from models.LogisticCatModelOneHot import LogisticCatModelOneHot
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.utils import prepare_dataset, prepare_tensors, prepare_split, prepare_dataLoaders
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv('./data.csv')
    model_type = "embed"

    all_chars = [
            'Jamie', 'Terry', 'Zangief', 'Kimberly', 'A.K.I.', 'Edmond Honda',
            'Ken', 'Dee Jay', 'Ryu', 'Manon', 'Marisa', 'Mai', 'Ed', 'Cammy',
            'Akuma', 'Lily', 'Luke', 'JP', 'Blanka', 'Juri', 'M. Bison',
            'Dhalsim', 'Guile', 'Chun-Li', 'Random', 'Rashid'
        ]

    num_chars = len(all_chars)

    df = pd.read_csv('./data.csv')
    df = prepare_dataset(df, num_chars=num_chars)
    print(len(df))

    # X, y = prepare_tensors(df, all_chars, model_type=model_type)
    # #X_df = pd.DataFrame(X.numpy(), columns=['mr_diff'] + all_chars*2)
    # from torch.utils.data import DataLoader, TensorDataset

    # temp_dataset = TensorDataset(X, y)
    # temp_loader = DataLoader(temp_dataset, batch_size=8, shuffle=True)

    # X_batch, y_batch = next(iter(temp_loader))
    # mr_diff = X[:, 0:1]
    # p1_idx  = X[:, 1].long()
    # p2_idx  = X[:, 2].long()

    # print("mr_diff min/max:", mr_diff.min().item(), mr_diff.max().item())