"""
Training callbacks
"""
import os
import csv
import pytorch_lightning as pl


class HistoryLogger(pl.callbacks.Callback):
    """Callback to log training metrics to CSV file"""
    def __init__(self, dir="history.csv"):
        self.dir = dir

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "loss_epoch" in metrics.keys():
            logs = {"epoch": trainer.current_epoch}
            keys = ["loss_epoch", "train_dice_epoch", "val_loss", "val_dice"]
            for key in keys:
                logs[key] = metrics[key].item()
            
            header = list(logs.keys())
            isFile = os.path.isfile(self.dir)
            
            with open(self.dir, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                if not isFile:
                    writer.writeheader()
                writer.writerow(logs)
