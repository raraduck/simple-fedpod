import logging
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
