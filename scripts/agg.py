import logging
import os

import numpy as np
import pandas as pd
import torch

from models.unet3d import UNet, PlainBlock, ResidualBlock

log = logging.getLogger(__name__)

_BLOCKS = {'residual': ResidualBlock, 'plain': PlainBlock}


class Aggregator:
    def __init__(self, args):
        self.args = args

    def run(self):
        log.info("=== Configuration ===")
        for key, value in vars(self.args).items():
            log.info("  %-12s = %s", key, value)
        log.info("=====================")

        if self.args.dry_run:
            self._init_model()
            self._init_split()
            return

        partitions = list(range(1, self.args.num_partitions + 1))
        log.info("Round %d / %d — aggregating %d partitions (algorithm=%s)",
                 self.args.round, self.args.rounds, len(partitions), self.args.algorithm)

        # 각 partition의 best.pt 로드
        state_dicts = []
        for p in partitions:
            ckpt_path = os.path.join(self.args.ckpt_root,
                                     self.args.job,
                                     f"inst{p:02d}",
                                     f"R{self.args.rounds:02d}r{self.args.round:02d}",
                                     "best.pt")
            if not os.path.exists(ckpt_path):
                log.warning("  partition %2d — checkpoint not found: %s", p, ckpt_path)
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dicts.append(ckpt["model"])
            log.info("  partition %2d — loaded  val_loss=%.4f  epoch=%d",
                     p, ckpt.get("val_loss", float("nan")), ckpt.get("epoch", -1))

        if not state_dicts:
            raise RuntimeError("집계할 체크포인트가 없습니다.")

        # Aggregation
        agg_state = self._aggregate(state_dicts)
        log.info("Aggregated %d / %d partitions", len(state_dicts), len(partitions))

        # 집계 모델 저장
        agg_path = self._agg_path(self.args.round)
        os.makedirs(os.path.dirname(agg_path), exist_ok=True)
        torch.save({"round": self.args.round, "model": agg_state}, agg_path)
        log.info("Saved aggregated model → %s", agg_path)

        # split CSV 업데이트 (next round 컬럼 추가)
        next_split = self._update_split(self.args.split, self.args.round)

        # Argo output parameters 기록
        next_round = self.args.round + 1
        next_epoch = next_round * self.args.epochs
        self._write_output("next-round",      str(next_round))
        self._write_output("next-epoch",      str(next_epoch))
        self._write_output("next-init-ckpt",  agg_path)
        self._write_output("next-split-csv",  next_split)
        log.info("Next — round=%d  epoch_offset=%d  init_ckpt=%s  split=%s",
                 next_round, next_epoch, agg_path, next_split)

    # ── Split CSV 초기화 (dry-run) ──────────────────────────────────────────
    def _sample_train(self, df, col):
        """전체 기관의 평균 train 수로 공통 λ를 정하고, 기관별로 Poisson(λ) 샘플링."""
        partitions = sorted(df["Partition_ID"].unique())
        per_n = [len(df[(df["Partition_ID"] == p) & (df["TrainOrVal"] == "train")])
                 for p in partitions]
        lam = np.mean(per_n) * self.args.sampling_rate
        log.info("Poisson λ = mean(N)×rate = %.1f×%.2f = %.2f",
                 np.mean(per_n), self.args.sampling_rate, lam)

        df[col] = None
        for p, n_total in zip(partitions, per_n):
            train_idx = df[(df["Partition_ID"] == p) & (df["TrainOrVal"] == "train")].index
            n = int(np.clip(np.random.poisson(lam), 1, n_total))
            selected = set(pd.Index(train_idx).to_series().sample(n=n, random_state=None).values)
            df.loc[train_idx, col] = train_idx.map(lambda i: 1 if i in selected else 0)
            log.info("  Partition %s — train=%d  selected=%d", p, n_total, n)

        df[col] = df[col].astype("Int64")
        return df

    def _init_split(self):
        df = pd.read_csv(self.args.split)[["Partition_ID", "Subject_ID", "TrainOrVal"]]
        df = self._sample_train(df, "R00")
        init_split = os.path.join(self.args.ckpt_root, self.args.job, "agg", "init", "split.csv")
        df.to_csv(init_split, index=False)
        log.info("Init split CSV saved → %s", init_split)
        self._write_output("next-split-csv", init_split)

    # ── Split CSV 업데이트 (round 집계 후) ──────────────────────────────────
    def _update_split(self, current_split_csv, round_idx):
        next_round = round_idx + 1
        df = pd.read_csv(current_split_csv)
        next_col = f"R{next_round:02d}"
        if self.args.sampling_mode == "dynamic":
            df = self._sample_train(df, next_col)
        else:  # static: 초기 선택 그대로 유지
            prev_col = f"R{round_idx:02d}"
            df[next_col] = df[prev_col] if prev_col in df.columns else None
            df[next_col] = df[next_col].astype("Int64")
            log.info("Static mode — copied selection from %s → %s", prev_col, next_col)
        agg_dir   = os.path.dirname(self._agg_path(round_idx))
        next_split = os.path.join(agg_dir, "split.csv")
        df.to_csv(next_split, index=False)
        log.info("Updated split CSV (col=%s) → %s", next_col, next_split)
        return next_split

    # ── 초기 모델 생성 (dry-run) ────────────────────────────────────────────
    def _init_model(self):
        enc_channels = list(map(int, self.args.channels.strip("[]").split(",")))
        model = UNet(
            in_ch=self.args.in_ch,
            out_classes=self.args.out_classes,
            channels=enc_channels,
            block=_BLOCKS[self.args.block],
            norm_key=self.args.norm,
        )
        init_path = os.path.join(self.args.ckpt_root, self.args.job, "agg", "init", "agg.pt")
        os.makedirs(os.path.dirname(init_path), exist_ok=True)
        torch.save({"round": -1, "model": model.state_dict()}, init_path)
        log.info("Initialized model saved → %s", init_path)
        self._write_output("next-init-ckpt", init_path)

    # ── 집계 모델 저장 경로 ─────────────────────────────────────────────────
    def _agg_path(self, round_idx):
        return os.path.join(self.args.ckpt_root,
                            self.args.job, "agg",
                            f"R{self.args.rounds:02d}r{round_idx:02d}",
                            "agg.pt")

    # ── FedAvg ─────────────────────────────────────────────────────────────
    def _aggregate(self, state_dicts):
        if self.args.algorithm == "fedavg":
            return self._fedavg(state_dicts)
        raise ValueError(f"지원하지 않는 알고리즘: {self.args.algorithm}")

    def _fedavg(self, state_dicts):
        avg = {}
        for key in state_dicts[0]:
            avg[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
        return avg

    def _write_output(self, name, value):
        out_dir = "/tmp/outputs"
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            f.write(value)
        log.info("Output param written: %s = %s", name, value)


def main():
    import argparse
    parser = argparse.ArgumentParser()

    # FL rounds
    parser.add_argument("--rounds",         type=int,   default=1,              help="총 FL 라운드 수")
    parser.add_argument("--round",          type=int,   default=0,              help="현재 FL 라운드 (0-indexed)")
    parser.add_argument("--epochs",         type=int,   default=30,             help="라운드당 에폭 수 (next-epoch 계산용)")

    # Data
    parser.add_argument("-c", "--split",    default="/experiments/fets/partition2/fets_split.csv",
                                                                                help="split CSV 경로 (원본 또는 이전 round 출력)")

    # Aggregation
    parser.add_argument("--dry-run",        action="store_true",                help="초기 모델 생성만 수행 (집계 없음)")
    parser.add_argument("--sampling-rate",  type=float, default=1.0,            help="train subjects 샘플링 비율 (0.0~1.0)")
    parser.add_argument("--sampling-mode",  default="static",                   help="샘플링 모드 (static / dynamic)")
    parser.add_argument("--algorithm",      default="fedavg",                   help="집계 알고리즘 (fedavg)")
    parser.add_argument("-J", "--job",      default="stage1",                   help="job 이름 (e.g. stage1 → stage1-p01, stage1/agg/)")
    parser.add_argument("--num-partitions", type=int,   default=2,              help="집계할 partition 수")
    parser.add_argument("--ckpt-root",      default="/checkpoints",             help="체크포인트 루트 경로")

    # Model (dry-run 시 초기화에 필요)
    parser.add_argument("--in-ch",          type=int,   default=4,              help="입력 채널 수")
    parser.add_argument("--out-classes",    type=int,   default=1,              help="출력 클래스 수")
    parser.add_argument("--channels",       default="[32,64,128,256]",          help="encoder 채널 수")
    parser.add_argument("--block",          default="residual",                 help="블록 타입 (residual / plain)")
    parser.add_argument("--norm",           default="instance",                 help="정규화 (instance / batch)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    Aggregator(args).run()


if __name__ == '__main__':
    main()
