import logging
import os

import torch

log = logging.getLogger(__name__)


def dice_loss(pred, target, eps=1e-6):
    """Binary Dice loss. pred: sigmoid 출력 (B, L, ...), target: binary mask (B, L, ...)"""
    pred   = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
    target = target.contiguous().view(target.shape[0], target.shape[1], -1)
    intersection = (pred * target).sum(-1)
    return 1 - (2 * intersection + eps) / (pred.sum(-1) + target.sum(-1) + eps)


class Trainer:
    def __init__(self, model, train_loader, val_loader, lr, device, ckpt_dir):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
        self.ckpt_dir     = ckpt_dir
        self.best_val     = float("inf")
        self.start_epoch  = 1

        os.makedirs(ckpt_dir, exist_ok=True)
        self._auto_resume()

    def _auto_resume(self):
        latest = os.path.join(self.ckpt_dir, "latest.pt")
        if not os.path.exists(latest):
            return
        ckpt = torch.load(latest, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_val    = ckpt.get("best_val", float("inf"))
        self.start_epoch = ckpt["epoch"] + 1
        log.info("Resumed from checkpoint — epoch %d  best_val=%.4f", ckpt["epoch"], self.best_val)

    def _save(self, epoch, val_loss, filename):
        path = os.path.join(self.ckpt_dir, filename)
        torch.save({
            "epoch":     epoch,
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val":  self.best_val,
            "val_loss":  val_loss,
        }, path)

    def _loss(self, images, labels):
        logits = self.model(images)
        pred   = torch.sigmoid(logits)
        return dice_loss(pred, labels).mean()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self._loss(images, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        mean_loss = total_loss / len(self.train_loader)
        log.info("Epoch %3d  train_loss=%.4f", epoch, mean_loss)
        return mean_loss

    @torch.no_grad()
    def val_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            total_loss += self._loss(images, labels).item()

        mean_loss = total_loss / len(self.val_loader)
        improved = mean_loss < self.best_val
        if improved:
            self.best_val = mean_loss
            self._save(epoch, mean_loss, "best.pt")
        self._save(epoch, mean_loss, "latest.pt")

        log.info("Epoch %3d    val_loss=%.4f%s", epoch, mean_loss, "  *best*" if improved else "")
        return mean_loss
