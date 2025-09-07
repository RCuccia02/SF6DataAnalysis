import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from models.LogisticCatModel import LogisticCatModel
from utils.utils import prepare_tensors, prepare_split, prepare_dataset

if __name__ == "__main__":
    mode = "train"
    torch.set_float32_matmul_precision('high')
    # -----------------------
    # 1️⃣ Caricamento dati
    # -----------------------
    df = pd.read_csv('./data.csv')
    all_chars = [
        'Jamie', 'Terry', 'Zangief', 'Kimberly', 'A.K.I.', 'Edmond Honda',
        'Ken', 'Dee Jay', 'Ryu', 'Manon', 'Marisa', 'Mai', 'Ed', 'Cammy',
        'Akuma', 'Lily', 'Luke', 'JP', 'Blanka', 'Juri', 'M. Bison',
        'Dhalsim', 'Guile', 'Chun-Li', 'Random', 'Rashid'
    ]
    num_chars = len(all_chars)

    df = prepare_dataset(df)

    X, y = prepare_tensors(df, all_chars, model_type="embed")

    train_dataset, val_dataset, test_dataset = prepare_split(X, y)

    early_stop = EarlyStopping(monitor="val_loss", patience=30, mode="min")
    checkpoint = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)
    callbacks = [early_stop, checkpoint]

    train_loader_warmup = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader_small = DataLoader(val_dataset, batch_size=8, num_workers=2, persistent_workers=True)

    

    model = LogisticCatModel(
        num_chars=num_chars,
        embed_dim=128,
        hidden_dim=512,
        lr=1e-2,           # LR alto per warm-up
        weight_decay=1e-4
    )

    trainer = pl.Trainer(
        max_epochs=5,
        callbacks=callbacks,
        accelerator="auto",
        devices=1
    )

    trainer.fit(model, train_dataloaders=train_loader_warmup, val_dataloaders=val_loader_small)

    # -----------------------
    # 6️⃣ Stage 2: Training normale
    # -----------------------
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=2)

    # Aggiorna LR per training normale
    model.hparams.lr = 1e-3

    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=callbacks,
        accelerator="auto",
        devices=1
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # -----------------------
    # 7️⃣ Test
    # -----------------------
    trainer.test(model, dataloaders=test_loader)