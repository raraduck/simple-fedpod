import logging
import os
import nibabel as nib
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)


def load_split(split_csv, partition_id):
    df = pd.read_csv(split_csv)
    df = df[df["Partition_ID"] == partition_id][["Subject_ID", "TrainOrVal"]]

    train = df[df["TrainOrVal"] == "train"]["Subject_ID"].tolist()
    val   = df[df["TrainOrVal"] == "val"]["Subject_ID"].tolist()

    log.info("Partition %d — train: %d, val: %d", partition_id, len(train), len(val))
    log.info("  train[:3]: %s", train[:3])
    log.info("  val[:3]:   %s", val[:3])

    return train, val


class FeTSDataset(Dataset):
    def __init__(self, data_dir, subjects, channels, lgrp, lnam, lidx):
        self.data_dir = data_dir
        self.subjects = subjects
        self.channels = channels  # 입력 채널 (t1, t1ce, t2, flair, seg 등)
        self.lgrp = lgrp          # [[1,2,4], ...] — sub 값 조합
        self.lnam = lnam          # ['wt', ...] — 레이블 이름
        self.lidx = lidx          # [1, ...] — 레이블 인덱스

        missing = [s for s in subjects if not os.path.isdir(os.path.join(data_dir, s))]
        log.info("FeTSDataset — subjects: %d, channels: %s", len(subjects), channels)
        log.info("  labels: %s (lgrp=%s, lidx=%s)", lnam, lgrp, lidx)
        if missing:
            log.warning("  missing subjects: %d — %s ...", len(missing), missing[:3])
        else:
            log.info("  all subject dirs found")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]

        # 입력 텐서: (C, H, W, D)
        arrays = []
        for ch in self.channels:
            path = os.path.join(self.data_dir, subject, f"{subject}_{ch}.nii.gz")
            arrays.append(nib.load(path).get_fdata(dtype=np.float32))
        image = torch.from_numpy(np.stack(arrays, axis=0))  # (C, H, W, D)

        # 레이블 텐서: sub.nii.gz에서 lgrp 조합으로 binary mask 생성 → (L, H, W, D)
        sub_path = os.path.join(self.data_dir, subject, f"{subject}_sub.nii.gz")
        sub = nib.load(sub_path).get_fdata(dtype=np.float32)
        masks = []
        for group in self.lgrp:
            mask = np.zeros_like(sub, dtype=np.float32)
            for val in group:
                mask += (sub == val).astype(np.float32)
            masks.append((mask > 0).astype(np.float32))
        label = torch.from_numpy(np.stack(masks, axis=0))  # (L, H, W, D)

        return image, label


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
    parser.add_argument("-D", "--data",    default="/data/fets128/trainval",              help="데이터 경로")
    parser.add_argument("-P", "--partition", type=int, default=1,                         help="FL 클라이언트 Partition ID")
    parser.add_argument("-d", "--dset",    default="fets",                               help="데이터셋 종류")
    parser.add_argument("-c", "--split",   default="/experiments/fets/partition2/fets_split.csv", help="train/val 분할 CSV")
    parser.add_argument("-C", "--chan",    default="[t1,t1ce,t2,flair]",                 help="입력 채널")
    parser.add_argument("-G", "--lgrp",    default="[[1,2,4]]",                          help="레이블 그룹")
    parser.add_argument("-N", "--lnam",    default="[wt]",                               help="레이블 이름")
    parser.add_argument("-I", "--lidx",    default="[1]",                                help="레이블 인덱스")

    # Model
    parser.add_argument("--block",         default="residual",                           help="블록 타입")
    parser.add_argument("--channels",      default="[32,64,128,256]",                    help="채널 수")
    parser.add_argument("--norm",          default="instance",                           help="정규화")

    # Training
    parser.add_argument("-E", "--epochs",  type=int,   default=30,                       help="에폭 수")
    parser.add_argument("--batch",         type=int,   default=2,                        help="배치 크기")
    parser.add_argument("--lr",            type=float, default=5e-3,                     help="학습률")
    parser.add_argument("--gpu",           type=int,   default=1,                        help="GPU 사용 여부 (1/0)")

    # Run
    parser.add_argument("-J", "--job",     default="test_run",                           help="실험 이름")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    App(args).run()


if __name__ == '__main__':
    main()
