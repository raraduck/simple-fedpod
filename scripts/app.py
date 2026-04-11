import json
import logging
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import FeTSDataset, load_split
from models.unet3d import UNet, PlainBlock, ResidualBlock
from trainer import Trainer

_BLOCKS = {'residual': ResidualBlock, 'plain': PlainBlock}

log = logging.getLogger(__name__)


class App:
    def __init__(self, args):
        self.args = args

    def run(self):
        log.info("=== Configuration ===")
        for key, value in vars(self.args).items():
            log.info("  %-10s = %s", key, value)
        log.info("=====================")

        train_subjects, val_subjects = load_split(self.args.split, self.args.partition, self.args.round)

        channels = self.args.chan.strip("[]").split(",")
        lgrp = [list(map(int, g.strip("[]").split(","))) for g in self.args.lgrp.strip("[]").split("],[")]
        lnam = self.args.lnam.strip("[]").split(",")
        lidx = list(map(int, self.args.lidx.strip("[]").split(",")))

        train_ds = FeTSDataset(self.args.data, train_subjects, channels, lgrp, lnam, lidx)
        val_ds   = FeTSDataset(self.args.data, val_subjects,   channels, lgrp, lnam, lidx)

        train_loader = DataLoader(train_ds, batch_size=self.args.batch, shuffle=True,  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.args.batch, shuffle=False, num_workers=2, pin_memory=True)

        log.info("DataLoader — train: %d batches, val: %d batches (batch_size=%d)",
                 len(train_loader), len(val_loader), self.args.batch)

        if train_subjects:
            images, labels = next(iter(train_loader))
            log.info("Batch sample —")
            log.info("  images : %s  dtype=%s", tuple(images.shape), images.dtype)
            log.info("  labels : %s  dtype=%s", tuple(labels.shape), labels.dtype)
            for i, name in enumerate(lnam):
                pos = int(labels[:, i].sum().item())
                total = int(labels[:, i].numel())
                log.info("  label[%d] %-4s — foreground: %d / %d voxels (%.1f%%)",
                         i, name, pos, total, 100 * pos / total)
        else:
            log.warning("Partition %d round %d — no training subjects sampled. Prv-val only.",
                        self.args.partition, self.args.round)

        # Model
        device = torch.device("cuda" if self.args.gpu and torch.cuda.is_available() else "cpu")
        enc_channels = list(map(int, self.args.channels.strip("[]").split(",")))
        model = UNet(
            in_ch=len(channels),
            out_classes=len(lidx),
            channels=enc_channels,
            block=_BLOCKS[self.args.block],
            norm_key=self.args.norm,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("Model — UNet  block=%s  channels=%s  norm=%s  device=%s",
                 self.args.block, enc_channels, self.args.norm, device)
        log.info("  parameters: %s", f"{n_params:,}")

        if self.args.init_ckpt:
            ckpt = torch.load(self.args.init_ckpt, map_location=device)
            model.load_state_dict(ckpt["model"])
            log.info("Loaded init model ← %s  (round=%s)", self.args.init_ckpt, ckpt.get("round"))

        if train_subjects:
            with torch.no_grad():
                dummy = images[:1].to(device)
                out = model(dummy)
            log.info("Forward pass — input: %s  output: %s", tuple(dummy.shape), tuple(out.shape))

        # Training
        ckpt_dir = os.path.join(self.args.ckpt_root, self.args.job,
                                f"inst{self.args.partition:02d}",
                                f"R{self.args.rounds:02d}r{self.args.round:02d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        log.info("Round %d / %d  ckpt_dir=%s", self.args.round, self.args.rounds, ckpt_dir)
        trainer = Trainer(model, train_loader, val_loader,
                          lr=self.args.lr, device=device, ckpt_dir=ckpt_dir,
                          epoch_offset=self.args.epoch, lnam=lnam)

        prv_val  = trainer.eval()
        prv_dice = trainer.eval_dice()

        # ── 학습 데이터 없음: prv-val 기록 후 종료 ──────────────────────────
        if not train_subjects:
            prv_dice_avg = sum(prv_dice.values()) / len(prv_dice) if prv_dice else 0.0
            metrics = {
                "partition":    self.args.partition,
                "round":        self.args.round,
                "n_train":      0,
                "prv_val_loss": round(prv_val,      6),
                "prv_dice":     {k: round(v, 6) for k, v in prv_dice.items()},
                "prv_dice_avg": round(prv_dice_avg, 6),
            }
            with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            runs_dir    = os.path.join(self.args.runs_root, self.args.job,
                                       f"inst{self.args.partition:02d}")
            epoch_start = self.args.epoch
            r           = self.args.round
            with SummaryWriter(runs_dir) as writer:
                writer.add_scalar("rnd_val_loss_prv/avg",    prv_val,      r)
                writer.add_scalar("rnd_val_loss_prvpst/avg", prv_val,      r)
                writer.add_scalar("rnd_val_dice_prv/avg",    prv_dice_avg, r)
                writer.add_scalar("rnd_val_dice_prvpst/avg", prv_dice_avg, r)
                for name in lnam:
                    writer.add_scalar(f"rnd_val_dice_prv/{name}",    prv_dice[name], r)
                    writer.add_scalar(f"rnd_val_dice_prvpst/{name}", prv_dice[name], r)
                writer.add_scalar("ech_val_loss_prv/avg",    prv_val,      epoch_start)
                writer.add_scalar("ech_val_loss_prvpst/avg", prv_val,      epoch_start)
                writer.add_scalar("ech_val_dice_prv/avg",    prv_dice_avg, epoch_start)
                writer.add_scalar("ech_val_dice_prvpst/avg", prv_dice_avg, epoch_start)
                for name in lnam:
                    writer.add_scalar(f"ech_val_dice_prv/{name}",    prv_dice[name], epoch_start)
                    writer.add_scalar(f"ech_val_dice_prvpst/{name}", prv_dice[name], epoch_start)
            log.info("Empty-train exit — partition=%d  prv_loss=%.4f  prv_dice_avg=%.4f",
                     self.args.partition, prv_val, prv_dice_avg)
            return

        trn_losses, val_losses = [], []
        for epoch in range(trainer.start_epoch, self.args.epoch + self.args.epochs + 1):
            trn_losses.append(trainer.train_epoch(epoch))
            val_losses.append(trainer.val_epoch(epoch))

        if trn_losses:
            pst_val      = trainer.eval()
            pst_dice     = trainer.eval_dice()
            avg_trn_loss = sum(trn_losses) / len(trn_losses)
            avg_val_loss = sum(val_losses)  / len(val_losses)
            prv_dice_avg = sum(prv_dice.values()) / len(prv_dice) if prv_dice else 0.0
            pst_dice_avg = sum(pst_dice.values()) / len(pst_dice) if pst_dice else 0.0

            # ── metrics.json (agg.py PID 계산용) ──────────────────────────
            metrics = {
                "partition":    self.args.partition,
                "round":        self.args.round,
                "n_train":      len(train_subjects),
                "avg_trn_loss": round(avg_trn_loss, 6),
                "avg_val_loss": round(avg_val_loss, 6),
                "prv_val_loss": round(prv_val,      6),
                "pst_val_loss": round(pst_val,      6),
                "prv_dice":     {k: round(v, 6) for k, v in prv_dice.items()},
                "pst_dice":     {k: round(v, 6) for k, v in pst_dice.items()},
                "prv_dice_avg": round(prv_dice_avg, 6),
                "pst_dice_avg": round(pst_dice_avg, 6),
            }
            with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            # ── TensorBoard ───────────────────────────────────────────────
            runs_dir = os.path.join(self.args.runs_root, self.args.job,
                                    f"inst{self.args.partition:02d}")
            epoch_start = self.args.epoch
            epoch_end   = self.args.epoch + self.args.epochs
            with SummaryWriter(runs_dir) as writer:
                r = self.args.round
                # ── round 단위 (x = round 번호) ──────────────────────────
                writer.add_scalar("rnd_trn_loss_avg/avg",    avg_trn_loss, r)
                writer.add_scalar("rnd_val_loss_avg/avg",    avg_val_loss, r)
                writer.add_scalar("rnd_val_loss_prv/avg",    prv_val,      r)
                writer.add_scalar("rnd_val_loss_pst/avg",    pst_val,      r)
                writer.add_scalar("rnd_val_loss_prvpst/avg", prv_val,      r)
                writer.add_scalar("rnd_val_loss_prvpst/avg", pst_val,      r)
                writer.add_scalar("rnd_val_dice_prv/avg",    prv_dice_avg, r)
                writer.add_scalar("rnd_val_dice_pst/avg",    pst_dice_avg, r)
                writer.add_scalar("rnd_val_dice_prvpst/avg", prv_dice_avg, r)
                writer.add_scalar("rnd_val_dice_prvpst/avg", pst_dice_avg, r)
                for name in lnam:
                    writer.add_scalar(f"rnd_val_dice_prv/{name}",    prv_dice[name], r)
                    writer.add_scalar(f"rnd_val_dice_pst/{name}",    pst_dice[name], r)
                    writer.add_scalar(f"rnd_val_dice_prvpst/{name}", prv_dice[name], r)
                    writer.add_scalar(f"rnd_val_dice_prvpst/{name}", pst_dice[name], r)
                # ── epoch 단위 (x = global epoch 번호) ───────────────────
                writer.add_scalar("ech_val_loss_avg/avg",    avg_val_loss, epoch_end)
                writer.add_scalar("ech_val_loss_prv/avg",    prv_val,      epoch_start)
                writer.add_scalar("ech_val_loss_pst/avg",    pst_val,      epoch_end)
                writer.add_scalar("ech_val_loss_prvpst/avg", prv_val,      epoch_start)
                writer.add_scalar("ech_val_loss_prvpst/avg", pst_val,      epoch_end)
                writer.add_scalar("ech_val_dice_prv/avg",    prv_dice_avg, epoch_start)
                writer.add_scalar("ech_val_dice_pst/avg",    pst_dice_avg, epoch_end)
                writer.add_scalar("ech_val_dice_prvpst/avg", prv_dice_avg, epoch_start)
                writer.add_scalar("ech_val_dice_prvpst/avg", pst_dice_avg, epoch_end)
                for name in lnam:
                    writer.add_scalar(f"ech_val_dice_prv/{name}",    prv_dice[name], epoch_start)
                    writer.add_scalar(f"ech_val_dice_pst/{name}",    pst_dice[name], epoch_end)
                    writer.add_scalar(f"ech_val_dice_prvpst/{name}", prv_dice[name], epoch_start)
                    writer.add_scalar(f"ech_val_dice_prvpst/{name}", pst_dice[name], epoch_end)
            log.info("TensorBoard — round=%d  prv=%.4f  avg_trn=%.4f  avg_val=%.4f  pst=%.4f",
                     self.args.round, prv_val, avg_trn_loss, avg_val_loss, pst_val)
            log.info("Dice prv=%s  pst=%s",
                     {k: f"{v:.4f}" for k, v in prv_dice.items()},
                     {k: f"{v:.4f}" for k, v in pst_dice.items()})


def main():
    import argparse
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("-D", "--data",      default="/data/fets128/trainval",                       help="데이터 경로")
    parser.add_argument("-P", "--partition", type=int, default=1,                                    help="FL 클라이언트 Partition ID")
    parser.add_argument("-d", "--dset",      default="fets",                                         help="데이터셋 종류")
    parser.add_argument("-c", "--split",     default="/experiments/fets/partition2/fets_split.csv",  help="train/val 분할 CSV")
    parser.add_argument("-C", "--chan",      default="[t1,t1ce,t2,flair]",                           help="입력 채널")
    parser.add_argument("-G", "--lgrp",      default="[[1,2,4]]",                                    help="레이블 그룹")
    parser.add_argument("-N", "--lnam",      default="[wt]",                                         help="레이블 이름")
    parser.add_argument("-I", "--lidx",      default="[1]",                                          help="레이블 인덱스")

    # Model
    parser.add_argument("--block",           default="residual",                                     help="블록 타입 (residual / plain)")
    parser.add_argument("--channels",        default="[32,64,128,256]",                              help="encoder 채널 수")
    parser.add_argument("--norm",            default="instance",                                     help="정규화 (instance / batch)")

    # Training
    parser.add_argument("--batch",           type=int,   default=2,                                  help="배치 크기")
    parser.add_argument("--lr",              type=float, default=1e-3,                               help="학습률")
    parser.add_argument("--gpu",             type=int,   default=1,                                  help="GPU 사용 여부 (1/0)")

    # FL rounds
    parser.add_argument("--rounds",          type=int,   default=1,                                  help="총 FL 라운드 수")
    parser.add_argument("--round",           type=int,   default=0,                                  help="현재 FL 라운드 (0-indexed)")
    parser.add_argument("-E", "--epochs",    type=int,   default=30,                                 help="라운드당 총 에폭 수")
    parser.add_argument("--epoch",           type=int,   default=0,                                  help="에폭 오프셋 (resume 시작점)")

    # Run
    parser.add_argument("-J", "--job",       default="test_run",                                     help="실험 이름")
    parser.add_argument("--ckpt-root",       default="/checkpoints",                                 help="체크포인트 루트 경로")
    parser.add_argument("--runs-root",       default="/runs",                                        help="TensorBoard runs 루트 경로")
    parser.add_argument("--init-ckpt",       default="",                                             help="초기 모델 경로 (agg.pt) — 미지정 시 random init")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    App(args).run()


if __name__ == '__main__':
    main()
