import pytorch_lightning as pl
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pandas as pd

class LogisticCatModelOneHot(pl.LightningModule):
    def __init__(self, num_chars, hidden_dim=256, lr=1e-3, weight_decay=1e-4, dropout=0.3):
        super().__init__()
        self.save_hyperparameters()

        # Calcolo input_dim totale: mr_diff(1) + player WR(2) + diff WR(1) +
        # char one-hot (num_chars*2) + char WR(2) + diff char WR(1)
        input_dim = 1 + 2 + 1 + num_chars * 2 + 2 + 1  

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim // 2, 1)
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_f1  = torchmetrics.F1Score(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")

        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.test_f1  = torchmetrics.F1Score(task="binary")
        self.test_auc = torchmetrics.AUROC(task="binary")

    def forward(self, x):
        return self.mlp(x)

    def step(self, batch, stage):
        X, y = batch
        logits = self(X).squeeze(1)
        loss = self.criterion(logits, y.float())
        preds = (torch.sigmoid(logits) > 0.5).int()

        getattr(self, f"{stage}_acc").update(preds, y.int())
        if stage == "val":
            self.val_f1.update(preds, y.int())
            self.val_auc.update(torch.sigmoid(logits), y.int())
        if stage == "test":
            self.test_f1.update(preds, y.int())
            self.test_auc.update(torch.sigmoid(logits), y.int())

        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log(f"{stage}_acc", getattr(self, f"{stage}_acc").compute(), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")
    
    def on_train_epoch_end(self):
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")
    
    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_auc", self.val_auc.compute(), prog_bar=True)

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auc.reset()

    def test_step(self, batch, batch_idx):
        self.step(batch, "test")
    
    def on_test_epoch_end(self):
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        self.log("test_f1", self.test_f1.compute(), prog_bar=True)
        self.log("test_auc", self.test_auc.compute(), prog_bar=True)

        self.test_acc.reset()
        self.test_f1.reset()
        self.test_auc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',     # monitora la val_loss
                factor=0.5,
                patience=10,
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    @staticmethod
    def load_model_from_checkpoint(checkpoint_path, num_chars=26, hidden_dim=256):
        return LogisticCatModelOneHot.load_from_checkpoint(
            checkpoint_path,
            num_chars=num_chars,
            hidden_dim=hidden_dim
        )