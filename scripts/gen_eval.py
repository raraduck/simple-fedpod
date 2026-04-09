"""Generalization evaluation — agg.pt 를 원본 CSV의 모든 val set에 평가.

train-job 과 병렬로 실행되어 공유 글로벌 모델의 일반화 성능을 기록한다.
val 대상: split CSV에서 Partition_ID가 존재하는(notna) 모든 val subject.
결과는 runs/{job}/gen_eval/ 에 TensorBoard로 기록된다.
"""
import logging
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import FeTSDataset
from models.loss import SoftDiceBCEWithLogitsLoss
from models.unet3d import UNet, PlainBlock, ResidualBlock

_BLOCKS = {'residual': ResidualBlock, 'plain': PlainBlock}
log = logging.getLogger(__name__)


@torch.no_grad()
def _eval_partition(model, val_loader, lnam, criterion, device):
    """단일 partition val 평가 — (val_loss, dice_dict, dice_avg) 반환."""
    model.eval()
    n = len(lnam)
    total_loss = 0.0
    tp = torch.zeros(n)
    fp = torch.zeros(n)
    fn = torch.zeros(n)

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        bce, dsc = criterion(logits, labels)
        total_loss += (bce + dsc.mean()).item()
        preds   = (torch.sigmoid(logits) > 0.5).float()
        spatial = list(range(2, preds.dim()))
        tp += (preds * labels).sum(dim=[0] + spatial).cpu()
        fp += (preds * (1 - labels)).sum(dim=[0] + spatial).cpu()
        fn += ((1 - preds) * labels).sum(dim=[0] + spatial).cpu()

    val_loss = total_loss / len(val_loader)
    dice     = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    dice_d   = {name: dice[i].item() for i, name in enumerate(lnam)}
    dice_avg = dice.mean().item()
    return val_loss, dice_d, dice_avg


def main():
    import argparse
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("-D", "--data",      default="/data/fets128/trainval",                      help="데이터 경로")
    parser.add_argument("-c", "--split",     default="/experiments/fets/partition2/fets_split.csv", help="split CSV 경로")
    parser.add_argument("-C", "--chan",      default="[t1,t1ce,t2,flair]",                          help="입력 채널")
    parser.add_argument("-G", "--lgrp",      default="[[1,2,4]]",                                   help="레이블 그룹")
    parser.add_argument("-N", "--lnam",      default="[wt]",                                        help="레이블 이름")
    parser.add_argument("-I", "--lidx",      default="[1]",                                         help="레이블 인덱스")

    # Model
    parser.add_argument("--block",           default="residual",                                    help="블록 타입")
    parser.add_argument("--channels",        default="[32,64,128,256]",                             help="encoder 채널 수")
    parser.add_argument("--norm",            default="instance",                                    help="정규화")
    parser.add_argument("--gpu",             type=int, default=1,                                   help="GPU 사용 여부")

    # FL
    parser.add_argument("--round",           type=int, default=0,                                   help="현재 FL 라운드")
    parser.add_argument("--rounds",          type=int, default=1,                                   help="총 FL 라운드 수 (미사용, yaml 호환용)")
    parser.add_argument("--epochs",          type=int, default=4,                                   help="라운드당 에폭 수 (epoch 축 계산용)")
    parser.add_argument("--init-ckpt",       default="",                                            help="평가할 agg.pt 경로")

    # Run
    parser.add_argument("-J", "--job",       default="stage2",                                      help="job 이름")
    parser.add_argument("--runs-root",       default="/runs",                                       help="TensorBoard runs 루트 경로")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    log.info("=== Gen-Eval Configuration ===")
    for key, value in vars(args).items():
        log.info("  %-12s = %s", key, value)
    log.info("==============================")

    channels = args.chan.strip("[]").split(",")
    lgrp     = [list(map(int, g.strip("[]").split(","))) for g in args.lgrp.strip("[]").split("],[")]
    lnam     = args.lnam.strip("[]").split(",")
    lidx     = list(map(int, args.lidx.strip("[]").split(",")))

    # ── CSV에서 Partition_ID가 존재하는 모든 val subject 수집 ────────────────
    df = pd.read_csv(args.split)
    df["Partition_ID"] = pd.to_numeric(df["Partition_ID"], errors="coerce")
    val_df = df[df["TrainOrVal"] == "val"].dropna(subset=["Partition_ID"])
    partitions = sorted(val_df["Partition_ID"].unique().astype(int).tolist())
    log.info("Val partitions from CSV: %s  (total val subjects: %d)", partitions, len(val_df))

    device       = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    enc_channels = list(map(int, args.channels.strip("[]").split(",")))
    model = UNet(
        in_ch=len(channels),
        out_classes=len(lidx),
        channels=enc_channels,
        block=_BLOCKS[args.block],
        norm_key=args.norm,
    ).to(device)

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        log.info("Loaded model ← %s  (round=%s)", args.init_ckpt, ckpt.get("round"))
    else:
        log.warning("--init-ckpt 미지정 — random init 모델로 평가합니다.")

    criterion = SoftDiceBCEWithLogitsLoss()

    # ── 기관별 평가 ─────────────────────────────────────────────────────────
    results = {}
    for p in partitions:
        val_subjects = val_df[val_df["Partition_ID"] == p]["Subject_ID"].tolist()
        if not val_subjects:
            continue
        val_ds     = FeTSDataset(args.data, val_subjects, channels, lgrp, lnam, lidx)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        val_loss, dice_d, dice_avg = _eval_partition(model, val_loader, lnam, criterion, device)
        results[p] = {"val_loss": val_loss, "dice": dice_d, "dice_avg": dice_avg,
                      "n_val": len(val_subjects)}
        log.info("Partition %2d (n_val=%3d) — val_loss=%.4f  dice_avg=%.4f  %s",
                 p, len(val_subjects), val_loss, dice_avg,
                 {k: f"{v:.4f}" for k, v in dice_d.items()})

    if not results:
        log.warning("평가 결과 없음 — TensorBoard 기록 생략")
        return

    # ── 전체 평균 ────────────────────────────────────────────────────────────
    n            = len(results)
    avg_loss     = sum(r["val_loss"]  for r in results.values()) / n
    avg_dice     = sum(r["dice_avg"]  for r in results.values()) / n
    avg_dice_cls = {name: sum(r["dice"][name] for r in results.values()) / n for name in lnam}
    log.info("Global avg (n=%d partitions) — val_loss=%.4f  dice_avg=%.4f", n, avg_loss, avg_dice)

    # ── TensorBoard ─────────────────────────────────────────────────────────
    r           = args.round
    epoch_start = args.round * args.epochs

    runs_dir = os.path.join(args.runs_root, args.job, "gen_eval")
    with SummaryWriter(runs_dir) as writer:
        # round 단위 (x = round 번호)
        writer.add_scalar("rnd_val_loss_prv/avg",     avg_loss, r)
        writer.add_scalar("rnd_val_loss_prvpst/avg",  avg_loss, r)
        writer.add_scalar("rnd_val_dice_prv/avg",     avg_dice, r)
        writer.add_scalar("rnd_val_dice_prvpst/avg",  avg_dice, r)
        for name in lnam:
            writer.add_scalar(f"rnd_val_dice_prv/{name}",    avg_dice_cls[name], r)
            writer.add_scalar(f"rnd_val_dice_prvpst/{name}", avg_dice_cls[name], r)
        # epoch 단위 (x = global epoch — round 시작 시점)
        writer.add_scalar("ech_val_loss_prv/avg",     avg_loss, epoch_start)
        writer.add_scalar("ech_val_loss_prvpst/avg",  avg_loss, epoch_start)
        writer.add_scalar("ech_val_dice_prv/avg",     avg_dice, epoch_start)
        writer.add_scalar("ech_val_dice_prvpst/avg",  avg_dice, epoch_start)
        for name in lnam:
            writer.add_scalar(f"ech_val_dice_prv/{name}",    avg_dice_cls[name], epoch_start)
            writer.add_scalar(f"ech_val_dice_prvpst/{name}", avg_dice_cls[name], epoch_start)

    log.info("TensorBoard written → %s  round=%d  loss=%.4f  dice=%.4f",
             runs_dir, r, avg_loss, avg_dice)


if __name__ == "__main__":
    main()
