import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

class LogisticCatModel(pl.LightningModule):
    def __init__(self, num_chars, embed_dim=128, hidden_dim=512, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Embeddings
        self.char_embed = nn.Embedding(num_chars, embed_dim)
        self.char_embed.weight.requires_grad=True

        # Dimensione input: mr_diff + p1_mr + p2_mr + embeddings
        num_mr_feats = 8
        num_emb_feats = embed_dim * 2
        num_interaction_feats = embed_dim * 2

        input_dim = num_mr_feats + num_emb_feats + num_interaction_feats

        self.norm = nn.LayerNorm(input_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

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

        diff = p1_emb - p2_emb
        interaction = p1_emb * p2_emb

        if mr_diff.dim() == 1:
            mr_diff = mr_diff.unsqueeze(1)
        
        mr_diff_abs = torch.abs(mr_diff)
        mr_diff_sign = torch.sign(mr_diff)
        mr_diff_sq = mr_diff ** 2
        mr_diff_log = torch.log(torch.abs(mr_diff) + 1)

        cos_sim = torch.nn.functional.cosine_similarity(p1_emb, p2_emb, dim=1, eps=1e-8).unsqueeze(1)
        dot_prod = (p1_emb * p2_emb).sum(dim=1, keepdim=True)  # prodotto scalare
        l2_diff = torch.norm(diff, dim=1, keepdim=True)

        
        x = torch.cat([
            mr_diff, mr_diff_abs, mr_diff_sign,
            mr_diff_sq, mr_diff_log, 
            p1_emb, p2_emb,
            diff, interaction, cos_sim, dot_prod, l2_diff
        ], dim=1)

        x = self.norm(x)
        logits = self.mlp(x).squeeze(1)
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
        return optimizer
        
    
    @staticmethod
    def load_model_from_checkpoint(checkpoint_path):
        return LogisticCatModel.load_from_checkpoint(
            checkpoint_path,
        )
