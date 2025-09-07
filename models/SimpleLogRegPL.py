import pytorch_lightning as pl
from sklearn.discriminant_analysis import StandardScaler
import torch.nn.functional as F
import torch
import torchmetrics

class SimpleLogRegPL(pl.LightningModule):
    def __init__(self, in_dim, lr=1e-3):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, 1)
        self.lr = lr
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        X_mr_diff, y = batch
        # qui devi decidere come concatenare le feature
        X = X_mr_diff
        logits = self(X)
        loss = self.criterion(logits, y.float())
        preds = (torch.sigmoid(logits) > 0.5).int()
        acc = self.train_acc(preds, y.int())
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        X_mr_diff, y = batch
        X = X_mr_diff
        logits = self(X)
        loss = self.criterion(logits, y.float())
        preds = (torch.sigmoid(logits) > 0.5).int()
        acc = self.val_acc(preds, y.int())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        X_mr_diff, y = batch
        X = X_mr_diff
        logits = self(X)
        loss = self.criterion(logits, y.float())
        preds = (torch.sigmoid(logits) > 0.5).int()
        acc = self.test_acc(preds, y.int())
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def load_model_from_checkpoint(checkpoint_path, in_dim, lr=1e-3):
        return SimpleLogRegPL.load_from_checkpoint(
            checkpoint_path,
            in_dim=in_dim,
            lr=lr
        )

        
