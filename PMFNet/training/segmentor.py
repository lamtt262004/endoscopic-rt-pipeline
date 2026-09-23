"""
PyTorch Lightning training module
"""
import pytorch_lightning as pl
import torch
from ..utils import dice_coef_fn, iou_score_fn, dice_tversky_loss


class Segmentor(pl.LightningModule):
    """PyTorch Lightning module for polyp segmentation"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def get_metrics(self):
        items = super().get_metrics()
        items.pop("v_num", None)
        return items

    def _step(self, batch):
        image, y_true = batch
        y_pred = self.model(image)
        loss = dice_tversky_loss(y_true.float(), y_pred)
        dice_score = dice_coef_fn(y_pred, y_true)
        iou_score = iou_score_fn(y_pred, y_true)
        return loss, dice_score, iou_score

    def training_step(self, batch, batch_idx):
        loss, dice_score, iou_score = self._step(batch)
        metrics = {"loss": loss, "train_dice": dice_score, "train_iou": iou_score}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dice_score, iou_score = self._step(batch)
        metrics = {"val_loss": loss, "val_dice": dice_score, "val_iou": iou_score}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, dice_score, iou_score = self._step(batch)
        metrics = {"loss": loss, "test_dice": dice_score, "test_iou": iou_score}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, verbose=True
        )
        lr_schedulers = {"scheduler": scheduler, "monitor": "val_dice"}
        return [optimizer], lr_schedulers
