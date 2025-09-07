import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.discriminant_analysis import StandardScaler
from torch.utils.data import DataLoader
from models.LogisticCatModel import LogisticCatModel
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import torch
import pandas as pd

def create_logger(save_dir="lightning_logs", name="sf6analisys_"):
    return CSVLogger(
        save_dir=save_dir,
        name=name,
    )

def create_callbacks():
    early_stop = EarlyStopping(monitor="val_loss", patience=50, mode="min")
    checkpoint = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)
    return [early_stop, checkpoint]

def create_trainer(max_epochs=200, callbacks=None, logger=None):
    return pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        devices=1 if torch.cuda.is_available() else None,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=callbacks,
        enable_progress_bar=True
    )
       

def prepare_dataLoaders(train_dataset, val_dataset, test_dataset, batch_size=64, num_workers=4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)
    return train_loader, val_loader, test_loader

def prepare_split(X, y):
    generator=torch.Generator().manual_seed(42)
    dataset = TensorDataset(X, y)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    return train_dataset, val_dataset, test_dataset

def prepare_dataset(df, df_long, num_chars=26):
        df = df[(df['p1_mr'] > 0) & (df['p2_mr'] > 0)].copy()
        df['mr_diff'] = df['p1_mr'] - df['p2_mr']

        player_winrate = df_long.groupby("player_id")["result"].mean()
        df["p1_winrate"] = df["p1_id"].map(player_winrate)
        df["p2_winrate"] = df["p2_id"].map(player_winrate)
        char_winrate = df_long.groupby("char")["result"].mean()
        df["p1_char_winrate"] = df["p1_char"].map(char_winrate)
        df["p2_char_winrate"] = df["p2_char"].map(char_winrate)

        return df

def prepare_tensors(df, all_chars, model_type="simple"):

    X_mr_diff = torch.tensor(df['mr_diff'].values, dtype=torch.float32).unsqueeze(1)
    X_mr_diff = torch.sign(X_mr_diff) * torch.log1p(torch.abs(X_mr_diff))
    X_p1_wr = torch.tensor(df['p1_winrate'].values, dtype=torch.float32).unsqueeze(1)
    X_p2_wr = torch.tensor(df['p2_winrate'].values, dtype=torch.float32).unsqueeze(1)
    X_p_wr_d = X_p1_wr - X_p2_wr
    X_p1_char_wr = torch.tensor(df['p1_char_winrate'].values, dtype=torch.float32).unsqueeze(1)
    X_p2_char_wr = torch.tensor(df['p2_char_winrate'].values, dtype=torch.float32).unsqueeze(1)
    X_c_wr_d = X_p1_char_wr - X_p2_char_wr

    if model_type == "onehot":

        # One-hot encoding
        X_p1_char = torch.tensor(
            pd.get_dummies(df['p1_char'])[all_chars].values,
            dtype=torch.float32
        )
        X_p2_char = torch.tensor(
            pd.get_dummies(df['p2_char'])[all_chars].values,
            dtype=torch.float32
        )

        X = torch.cat([X_mr_diff, X_p1_wr, X_p2_wr, X_p_wr_d,X_p1_char, X_p2_char, X_p1_char_wr, X_p2_char_wr, X_c_wr_d], dim=1)

    if model_type == "embed":
        char_to_idx = {char: i for i, char in enumerate(all_chars)}
        X_p1_idx = torch.tensor(df['p1_char'].map(char_to_idx).values, dtype=torch.long).unsqueeze(1)
        X_p2_idx = torch.tensor(df['p2_char'].map(char_to_idx).values, dtype=torch.long).unsqueeze(1)
        X = torch.cat([X_mr_diff, X_p1_wr, X_p2_wr, X_p_wr_d, X_p1_idx, X_p2_idx, X_p1_char_wr, X_p2_char_wr, X_c_wr_d], dim=1)

    if model_type == "simple":
        X = X_mr_diff
    
    y = torch.tensor(df['p1_result'].values, dtype=torch.long)

    return X, y
