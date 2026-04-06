import logging
import os

import torch
from torch.utils.data import DataLoader

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

        train_subjects, val_subjects = load_split(self.args.split, self.args.partition)

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

        images, labels = next(iter(train_loader))
        log.info("Batch sample —")
        log.info("  images : %s  dtype=%s", tuple(images.shape), images.dtype)
        log.info("  labels : %s  dtype=%s", tuple(labels.shape), labels.dtype)
        for i, name in enumerate(lnam):
            pos = int(labels[:, i].sum().item())
            total = int(labels[:, i].numel())
            log.info("  label[%d] %-4s — foreground: %d / %d voxels (%.1f%%)",
                     i, name, pos, total, 100 * pos / total)

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

        with torch.no_grad():
            dummy = images[:1].to(device)
            out = model(dummy)
        log.info("Forward pass — input: %s  output: %s", tuple(dummy.shape), tuple(out.shape))

        # Training
        ckpt_dir = os.path.join(self.args.ckpt_root, self.args.job)
        trainer = Trainer(model, train_loader, val_loader,
                          lr=self.args.lr, device=device, ckpt_dir=ckpt_dir)
        for epoch in range(trainer.start_epoch, self.args.epochs + 1):
            trainer.train_epoch(epoch)
            trainer.val_epoch(epoch)


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
    parser.add_argument("-E", "--epochs",    type=int,   default=30,                                 help="에폭 수")
    parser.add_argument("--batch",           type=int,   default=2,                                  help="배치 크기")
    parser.add_argument("--lr",              type=float, default=5e-3,                               help="학습률")
    parser.add_argument("--gpu",             type=int,   default=1,                                  help="GPU 사용 여부 (1/0)")

    # Run
    parser.add_argument("-J", "--job",       default="test_run",                                     help="실험 이름")
    parser.add_argument("--ckpt-root",       default="/checkpoints",                                 help="체크포인트 루트 경로")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    App(args).run()


if __name__ == '__main__':
    main()
