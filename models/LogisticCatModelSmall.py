
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F

class LogisticCatModelSmall(pl.LightningModule):
    def __init__(self, num_chars, embed_dim=64, hidden_dim=256, lr=1e-3, weight_decay=1e-4, dropout=0.3, grad_clip_val=1.0):
        super().__init__()
        self.save_hyperparameters()

        # Embeddings
        self.char_embed = nn.Embedding(num_chars, embed_dim)
        self.char_embed.weight.requires_grad=True

        num_mr_feats = 1
        num_emb_feats = embed_dim * 2
        num_emb_derivated_feats = embed_dim * 2

        input_dim = num_mr_feats + num_emb_feats + num_emb_derivated_feats

        self.norm = nn.BatchNorm1d(input_dim)

        # MLP con residual + batchnorm
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.residual_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.residual_fc3 = nn.Linear(hidden_dim // 2 , hidden_dim // 4)
        self.dropout3 = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden_dim // 4, 1)

        self.lr = lr
        self.weight_decay = weight_decay

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Metriche
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_f1  = torchmetrics.F1Score(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")

        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.test_f1  = torchmetrics.F1Score(task="binary")
        self.test_auc = torchmetrics.AUROC(task="binary")

    def forward(self, mr_diff, p1_char, p2_char):
        p1_emb = self.char_embed(p1_char)
        p2_emb = self.char_embed(p2_char)
        emb_diff = p1_emb - p2_emb
        emb_abs_diff = torch.abs(emb_diff)
        if mr_diff.dim() == 1:
            mr_diff = mr_diff.unsqueeze(1)

        
        x = torch.cat([
            mr_diff, 
            p1_emb, p2_emb,
            emb_diff, emb_abs_diff
        ], dim=1)

        x = self.norm(x)

        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout1(out)
        residual = out

        residual_proj = self.residual_fc2(residual)  # nn.Linear(512, 256)
        
        out = F.relu(self.bn2(self.fc2(out))) + residual_proj
        out = self.dropout2(out)
        residual2 = out
        # Adatta residual da 256 → 128
        residual_proj2 = self.residual_fc3(residual2)  # nn.Linear(256, 128)
        out = F.relu(self.bn3(self.fc3(out))) + residual_proj2
        out = self.dropout3(out)

        logits = self.fc_out(out).squeeze(1)

        logits = self.fc_out(out).squeeze(1)
        return logits

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X[:,0:1], X[:,1].long(), X[:,2].long())
        loss = self.criterion(logits, y.float())
        preds = (torch.sigmoid(logits) > 0.5).int()

        # Aggiorna metriche train
        self.train_acc.update(preds, y.int())

        # Log
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_acc", self.train_acc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X[:,0:1], X[:,1].long(), X[:,2].long())
        loss = self.criterion(logits, y.float())
        preds = (torch.sigmoid(logits) > 0.5).int()

        self.val_acc.update(preds, y.int())
        self.val_f1.update(preds, y.int())
        self.val_auc.update(torch.sigmoid(logits), y.int())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_auc", self.val_auc.compute(), prog_bar=True)

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auc.reset()



    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X[:,0:1], X[:,1].long(), X[:,2].long())
        loss = self.criterion(logits, y.float())
        preds = (torch.sigmoid(logits) > 0.5).int()

        self.test_acc.update(preds, y.int())
        self.test_f1.update(preds, y.int())
        self.test_auc.update(torch.sigmoid(logits), y.int())
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

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
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5
            ),
            "monitor": "val_loss",   # metrica da monitorare
            "interval": "epoch",     # controllo ogni epoca
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    @staticmethod
    def load_model_from_checkpoint(checkpoint_path):
        return LogisticCatModelSmall.load_from_checkpoint(
            checkpoint_path,
        )
