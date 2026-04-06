import logging

import torch
from torch.utils.data import DataLoader

from dataset import FeTSDataset, load_split

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

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    App(args).run()


if __name__ == '__main__':
    main()
