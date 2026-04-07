import logging
import os

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

        partitions = list(range(1, self.args.num_partitions + 1))
        log.info("Round %d — aggregating %d partitions (algorithm=%s)",
                 self.args.round, len(partitions), self.args.algorithm)

        # 각 partition의 best.pt 로드
        state_dicts = []
        for p in partitions:
            job_name = f"{self.args.job_prefix}-p{p:02d}"
            ckpt_path = os.path.join(self.args.ckpt_root, job_name,
                                     f"round_{self.args.round:03d}", "best.pt")
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
        agg_dir  = os.path.join(self.args.ckpt_root, f"{self.args.job_prefix}_agg",
                                f"round_{self.args.round:03d}")
        agg_path = os.path.join(agg_dir, "agg.pt")
        os.makedirs(agg_dir, exist_ok=True)
        torch.save({"round": self.args.round, "model": agg_state}, agg_path)
        log.info("Saved aggregated model → %s", agg_path)

        # Argo output parameters 기록
        next_round = self.args.round + 1
        next_epoch = (self.args.round + 1) * self.args.epochs
        self._write_output("next-round", str(next_round))
        self._write_output("next-epoch", str(next_epoch))
        log.info("Next — round=%d  epoch_offset=%d", next_round, next_epoch)

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
    parser.add_argument("--rounds",         type=int,   default=1,          help="총 FL 라운드 수")
    parser.add_argument("--round",          type=int,   default=0,          help="현재 FL 라운드 (0-indexed)")
    parser.add_argument("--epochs",         type=int,   default=30,         help="라운드당 에폭 수 (next-epoch 계산용)")

    # Aggregation
    parser.add_argument("--algorithm",      default="fedavg",               help="집계 알고리즘 (fedavg)")
    parser.add_argument("--job-prefix",     default="stage1",               help="job 이름 prefix (e.g. stage1 → stage1_p01)")
    parser.add_argument("--num-partitions", type=int,   default=2,          help="집계할 partition 수")
    parser.add_argument("--ckpt-root",      default="/checkpoints",         help="체크포인트 루트 경로")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    Aggregator(args).run()


if __name__ == '__main__':
    main()
