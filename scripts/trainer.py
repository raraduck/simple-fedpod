import logging
import os

import torch

from models.loss import SoftDiceBCEWithLogitsLoss

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, train_loader, val_loader, lr, device, ckpt_dir, epoch_offset=0, lnam=None,
                 scheduler_type="none", lr_t_max=100, lr_step_size=5, lr_gamma=0.5):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler    = self._make_scheduler(scheduler_type, lr_t_max, lr_step_size, lr_gamma)
        self.criterion    = SoftDiceBCEWithLogitsLoss()
        self.ckpt_dir     = ckpt_dir
        self.best_val     = float("inf")
        self.start_epoch  = epoch_offset + 1  # round 간 이어받기
        self.lnam         = lnam or []

        os.makedirs(ckpt_dir, exist_ok=True)
        self._auto_resume()  # round 내 재시작 (있으면 덮어씀)

    def _make_scheduler(self, scheduler_type, t_max, step_size, gamma):
        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max)
        if scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        return None

    def _auto_resume(self):
        latest = os.path.join(self.ckpt_dir, "latest.pt")
        if not os.path.exists(latest):
            return
        ckpt = torch.load(latest, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.best_val    = ckpt.get("best_val", float("inf"))
        self.start_epoch = ckpt["epoch"] + 1
        log.info("Resumed from checkpoint — epoch %d  best_val=%.4f", ckpt["epoch"], self.best_val)

    def _save(self, epoch, val_loss, filename):
        path = os.path.join(self.ckpt_dir, filename)
        ckpt = {
            "epoch":     epoch,
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val":  self.best_val,
            "val_loss":  val_loss,
        }
        if self.scheduler is not None:
            ckpt["scheduler"] = self.scheduler.state_dict()
        torch.save(ckpt, path)

    def _loss(self, images, labels):
        logits = self.model(images)
        bce, dsc = self.criterion(logits, labels)
        return bce + dsc.mean()

    def scheduler_step(self):
        if self.scheduler is not None:
            self.scheduler.step()
            log.info("LR → %.2e", self.optimizer.param_groups[0]["lr"])

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
    def eval(self):
        """체크포인트 저장 없이 val loss만 계산 (TensorBoard prv/pst 측정용)."""
        self.model.eval()
        total_loss = 0.0
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            total_loss += self._loss(images, labels).item()
        return total_loss / len(self.val_loader)

    @torch.no_grad()
    def eval_dice(self):
        """클래스별 hard Dice 계산 (sigmoid > 0.5, global aggregate over val set)."""
        self.model.eval()
        n = len(self.lnam)
        tp = torch.zeros(n)
        fp = torch.zeros(n)
        fn = torch.zeros(n)
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            preds = (torch.sigmoid(self.model(images)) > 0.5).float()
            spatial = list(range(2, preds.dim()))
            tp += (preds * labels).sum(dim=[0] + spatial).cpu()
            fp += (preds * (1 - labels)).sum(dim=[0] + spatial).cpu()
            fn += ((1 - preds) * labels).sum(dim=[0] + spatial).cpu()
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        return {name: dice[i].item() for i, name in enumerate(self.lnam)}

    @torch.no_grad()
    def val_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            total_loss += self._loss(images, labels).item()

        avg_val_loss = total_loss / len(self.val_loader)
        improved = avg_val_loss < self.best_val
        if improved:
            self.best_val = avg_val_loss
            self._save(epoch, avg_val_loss, "best.pt")
        self._save(epoch, avg_val_loss, "latest.pt")

        log.info("Epoch %3d    avg_val_loss=%.4f%s", epoch, avg_val_loss, "  *best*" if improved else "")
        # Katib StdOut collector 포맷
        print("{metricName: val_loss, metricValue: %.4f}" % avg_val_loss, flush=True)
        return avg_val_loss
