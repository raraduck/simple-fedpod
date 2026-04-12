import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

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
            self._write_output("next-round", "0")
            self._write_output("next-epoch",  "0")
            return

        if self.args.sampling_mode == "pool":
            raise ValueError("--sampling-mode pool은 dry-run 전용입니다. 라운드에서는 static 또는 dynamic을 사용하세요.")
        if self.args.selection == "entropy":
            raise ValueError("라운드에서 --selection entropy는 지원하지 않습니다. --selection random을 사용하세요.")

        partitions = [int(p.strip()) for p in self.args.partitions.split(",") if p.strip()]
        log.info("Round %d / %d — aggregating %d partitions %s (algorithm=%s)",
                 self.args.round, self.args.rounds, len(partitions), partitions, self.args.algorithm)

        # 각 partition의 best.pt 및 metrics.json 로드
        state_dicts = []
        n_trains    = []
        for p in partitions:
            ckpt_dir  = os.path.join(self.args.ckpt_root,
                                     self.args.job,
                                     f"inst{p:02d}",
                                     f"R{self.args.rounds:02d}r{self.args.round:02d}")
            ckpt_path    = os.path.join(ckpt_dir, "best.pt")
            metrics_path = os.path.join(ckpt_dir, "metrics.json")
            if not os.path.exists(ckpt_path):
                log.warning("  partition %2d — checkpoint not found (no training data?): %s", p, ckpt_path)
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu")
            n_train = 0
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    n_train = json.load(f).get("n_train", 0)
            state_dicts.append(ckpt["model"])
            n_trains.append(n_train)
            log.info("  partition %2d — loaded  val_loss=%.4f  epoch=%d  n_train=%d",
                     p, ckpt.get("val_loss", float("nan")), ckpt.get("epoch", -1), n_train)

        if not state_dicts:
            raise RuntimeError("집계할 체크포인트가 없습니다.")

        # Aggregation
        agg_state = self._aggregate(state_dicts, n_trains)
        log.info("Aggregated %d / %d partitions", len(state_dicts), len(partitions))

        # 집계 모델 저장
        agg_path = self._agg_path(self.args.round)
        os.makedirs(os.path.dirname(agg_path), exist_ok=True)
        torch.save({"round": self.args.round, "model": agg_state}, agg_path)
        log.info("Saved aggregated model → %s", agg_path)

        # split CSV 업데이트 (next round 컬럼 추가)
        next_split = self._update_split(self.args.split, self.args.round, agg_state)

        # 기관 평균 TensorBoard 기록
        self._write_inst_avg_tensorboard(partitions)

        # Argo output parameters 기록
        next_round = self.args.round + 1
        next_epoch = next_round * self.args.epochs
        self._write_output("next-round",      str(next_round))
        self._write_output("next-epoch",      str(next_epoch))
        self._write_output("next-init-ckpt",  agg_path)
        self._write_output("next-split-csv",  next_split)
        log.info("Next — round=%d  epoch_offset=%d  init_ckpt=%s  split=%s",
                 next_round, next_epoch, agg_path, next_split)

    # ── 기관 평균 TensorBoard ───────────────────────────────────────────────
    def _write_inst_avg_tensorboard(self, partitions):
        """각 기관의 metrics.json을 읽어 평균을 inst_avg TensorBoard에 기록."""
        metrics_list = []
        for p in partitions:
            m_path = os.path.join(self.args.ckpt_root, self.args.job,
                                  f"inst{p:02d}",
                                  f"R{self.args.rounds:02d}r{self.args.round:02d}",
                                  "metrics.json")
            if not os.path.exists(m_path):
                log.warning("inst_avg — metrics.json not found: %s", m_path)
                continue
            with open(m_path) as f:
                metrics_list.append(json.load(f))

        if not metrics_list:
            log.warning("inst_avg — no metrics.json found, skipping TensorBoard")
            return

        # n_train=0 (학습 데이터 없는 기관) 은 평균 계산에서 제외
        n_skip = sum(1 for m in metrics_list if m.get("n_train", 0) == 0)
        if n_skip:
            log.warning("inst_avg — %d partition(s) with n_train=0 excluded from average", n_skip)
        metrics_list = [m for m in metrics_list if m.get("n_train", 0) > 0]
        if not metrics_list:
            log.warning("inst_avg — all partitions had no training data, skipping TensorBoard")
            return

        def _mean(key):
            return sum(m[key] for m in metrics_list) / len(metrics_list)

        avg_trn_loss = _mean("avg_trn_loss")
        avg_val_loss = _mean("avg_val_loss")
        prv_val      = _mean("prv_val_loss")
        pst_val      = _mean("pst_val_loss")
        prv_dice_avg = _mean("prv_dice_avg")
        pst_dice_avg = _mean("pst_dice_avg")

        lnam     = list(metrics_list[0]["prv_dice"].keys())
        prv_dice = {n: sum(m["prv_dice"][n] for m in metrics_list) / len(metrics_list) for n in lnam}
        pst_dice = {n: sum(m["pst_dice"][n] for m in metrics_list) / len(metrics_list) for n in lnam}

        r           = self.args.round
        epoch_start = self.args.round * self.args.epochs
        epoch_end   = (self.args.round + 1) * self.args.epochs

        runs_dir = os.path.join(self.args.runs_root, self.args.job, "inst_avg")
        with SummaryWriter(runs_dir) as writer:
            # round 단위 (x = round 번호)
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
            # epoch 단위 (x = global epoch 번호)
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

        log.info("inst_avg TensorBoard — round=%d  n=%d  prv_loss=%.4f  pst_loss=%.4f"
                 "  prv_dice=%.4f  pst_dice=%.4f",
                 r, len(metrics_list), prv_val, pst_val, prv_dice_avg, pst_dice_avg)

    # ── Split CSV 초기화 (dry-run) ──────────────────────────────────────────
    def _sample_train(self, df, col):
        """전체 기관의 원본 train 수로 공통 λ를 정하고, 기관별로 Poisson(λ) 샘플링.
        pool 컬럼이 있으면 pool=1 subject만 샘플링 대상으로 제한하되 λ는 원본 수 기준."""
        has_pool = "pool" in df.columns
        partitions = sorted(df[df["TrainOrVal"] == "train"]["Partition_ID"].unique())
        # λ: 원본 train 보유량 기준 (pool 마스킹 무관)
        per_n = [len(df[(df["Partition_ID"] == p) & (df["TrainOrVal"] == "train")])
                 for p in partitions]
        lam = np.mean(per_n) * self.args.sampling_rate
        log.info("Poisson λ = mean(N)×rate = %.1f×%.2f = %.2f",
                 np.mean(per_n), self.args.sampling_rate, lam)

        df[col] = None
        for p, n_total in zip(partitions, per_n):
            if has_pool:
                pool_idx   = df[(df["Partition_ID"] == p) & (df["TrainOrVal"] == "train") & (df["pool"] == 1)].index
                nopool_idx = df[(df["Partition_ID"] == p) & (df["TrainOrVal"] == "train") & (df["pool"] != 1)].index
                df.loc[nopool_idx, col] = 0
            else:
                pool_idx = df[(df["Partition_ID"] == p) & (df["TrainOrVal"] == "train")].index
            n_pool = len(pool_idx)
            n = int(np.clip(round(lam), 1, n_pool)) if n_pool > 0 else 0
            if n_pool > 0:
                selected = set(pd.Index(pool_idx).to_series().sample(n=n, random_state=None).values)
                df.loc[pool_idx, col] = [1 if i in selected else 0 for i in pool_idx]
            log.info("  Partition %s — train=%d  pool=%d  selected=%d", p, n_total, n_pool, n)

        df[col] = df[col].astype("Int64")
        return df

    def _init_split(self):
        df = pd.read_csv(self.args.split)[["Partition_ID", "Subject_ID", "TrainOrVal"]]
        df["Partition_ID"] = df["Partition_ID"].astype("Int64")

        if self.args.sampling_mode == "pool":
            # Step 1: entropy/random으로 pool 결정 → 비선택 subject Partition_ID=NA 마스킹
            if self.args.selection == "entropy":
                if not (self.args.committee or self.args.committee_job):
                    raise RuntimeError("pool+entropy dry-run에는 --committee 또는 --committee-job 이 필요합니다.")
                device = torch.device("cuda" if self.args.gpu and torch.cuda.is_available() else "cpu")
                models = self._build_committee(device)
                channels = self.args.committee_chan.strip("[]").split(",")
                df = self._sample_train_entropy(df, "_pool_tmp", models, device, channels)
            else:  # random
                df = self._sample_train(df, "_pool_tmp")
            non_pool = (df["TrainOrVal"] == "train") & (df["_pool_tmp"] == 0)
            df["pool"] = pd.NA
            df.loc[df["TrainOrVal"] == "train", "pool"] = 1
            df.loc[non_pool, "pool"] = 0
            df["pool"] = df["pool"].astype("Int64")
            df = df.drop(columns=["_pool_tmp"])
            log.info("Pool — %d / %d train subjects in pool (selection=%s)",
                    int((~non_pool & (df["TrainOrVal"] == "train")).sum()), int((df["TrainOrVal"] == "train").sum()), self.args.selection)
            # Step 2: pool 내에서 per-partition Poisson 샘플링 → R00
            df = self._sample_train(df, "R00")
        else:
            # static / dynamic 공통 기본 흐름
            if self.args.selection == "entropy":
                if not (self.args.committee or self.args.committee_job):
                    log.warning("--selection entropy 이지만 --committee 미지정 → random으로 대체")
                    df = self._sample_train(df, "R00")
                else:
                    device = torch.device("cuda" if self.args.gpu and torch.cuda.is_available() else "cpu")
                    models = self._build_committee(device)
                    channels = self.args.committee_chan.strip("[]").split(",")
                    df = self._sample_train_entropy(df, "R00", models, device, channels)
            else:
                df = self._sample_train(df, "R00")

        init_split = os.path.join(self.args.ckpt_root, self.args.job, "agg", "init", "split.csv")
        df.to_csv(init_split, index=False)
        log.info("Init split CSV saved → %s", init_split)
        self._write_output("next-split-csv", init_split)

    # ── Split CSV 업데이트 (round 집계 후) ──────────────────────────────────
    def _update_split(self, current_split_csv, round_idx, agg_state=None):
        next_round = round_idx + 1
        df = pd.read_csv(current_split_csv)
        # CSV 읽기 시 정수 컬럼이 float으로 변환되는 문제 방지
        df["Partition_ID"] = df["Partition_ID"].astype("Int64")
        for col in df.columns:
            if col[0] == "R" and col[1:].isdigit():
                df[col] = df[col].astype("Int64")
        if "pool" in df.columns:
            df["pool"] = df["pool"].astype("Int64")
        next_col = f"R{next_round:02d}"
        if self.args.sampling_mode == "dynamic":
            if self.args.selection == "entropy":
                raise ValueError("라운드에서 --selection entropy는 지원하지 않습니다. --selection random을 사용하세요.")
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

    def _build_model(self, state_dict, device):
        enc_channels = list(map(int, self.args.channels.strip("[]").split(",")))
        model = UNet(
            in_ch=self.args.in_ch,
            out_classes=self.args.out_classes,
            channels=enc_channels,
            block=_BLOCKS[self.args.block],
            norm_key=self.args.norm,
        )
        model.load_state_dict(state_dict)
        model.to(device).eval()
        return model

    def _build_committee(self, device):
        """committee 모델 리스트 로드.

        우선순위:
        1. --committee-job 지정 시 체크포인트 구조에서 자동 생성
           (--committee-partitions "1,2,5" 로 기관 선택)
        2. --committee 직접 경로 지정 (.pt 파일 또는 폴더, 콤마 구분 복수 경로)
        """
        pt_files = []

        if self.args.committee_job:
            partitions = [int(p.strip()) for p in self.args.committee_partitions.split(",") if p.strip()]
            for p in partitions:
                path = os.path.join(
                    self.args.ckpt_root,
                    self.args.committee_job,
                    f"inst{p:02d}",
                    f"R{self.args.committee_rounds:02d}r{self.args.committee_round:02d}",
                    "best.pt",
                )
                if not os.path.exists(path):
                    log.warning("Committee — checkpoint not found: %s", path)
                    continue
                pt_files.append(path)
        else:
            entries = [p.strip() for p in self.args.committee.split(",") if p.strip()]
            for entry in entries:
                if os.path.isfile(entry):
                    pt_files.append(entry)
                else:
                    pt_files.extend(sorted(
                        os.path.join(entry, f) for f in os.listdir(entry) if f.endswith(".pt")
                    ))

        if not pt_files:
            raise RuntimeError("committee 경로에서 .pt 파일을 찾을 수 없습니다.")
        models = []
        for pt in pt_files:
            ckpt = torch.load(pt, map_location="cpu")
            models.append(self._build_committee_model(ckpt["model"], device))
            log.info("Committee model loaded ← %s", pt)
        log.info("Committee size: %d", len(models))
        return models

    def _build_committee_model(self, state_dict, device):
        """committee 전용 모델 빌드 — committee-* 아키텍처 인자 사용."""
        enc_channels = list(map(int, self.args.channels.strip("[]").split(",")))
        model = UNet(
            in_ch=self.args.committee_in_ch,
            out_classes=self.args.committee_out_classes,
            channels=enc_channels,
            block=_BLOCKS[self.args.block],
            norm_key=self.args.norm,
        )
        model.load_state_dict(state_dict)
        model.to(device).eval()
        return model

    def _sample_train_entropy(self, df, col, models, device, channels):
        """모델 committee로 전체 train subject 추론 → 예측 평균 엔트로피 기준 global top-k 선택."""
        import nibabel as nib

        partitions = sorted(df[df["TrainOrVal"] == "train"]["Partition_ID"].unique())
        per_n = [len(df[(df["Partition_ID"] == p) & (df["TrainOrVal"] == "train")])
                 for p in partitions]
        lam = np.mean(per_n) * self.args.sampling_rate
        # global top-k: random 모드와 동일하게 파티션별 min(round(λ), n_i) 합산
        k = int(sum(int(np.clip(round(lam), 1, n)) for n in per_n))
        use_bald = len(models) > 1
        metric_name = "BALD" if use_bald else "entropy"
        log.info("%s selection — committee=%d  λ=%.2f  global k=%d  (total train=%d)",
                 metric_name.upper(), len(models), lam, k, sum(per_n))

        train_rows = df[df["TrainOrVal"] == "train"]
        scores = {}   # subject_id → score
        eps = 1e-6
        n_total = len(train_rows)
        log_interval = max(1, n_total // 10)
        with torch.no_grad():
            for i, (_, row) in enumerate(train_rows.iterrows()):
                subj = row["Subject_ID"]
                if i % log_interval == 0:
                    log.info("  %s scoring — %d / %d", metric_name.upper(), i, n_total)
                try:
                    imgs = []
                    for ch in channels:
                        path = os.path.join(self.args.data,
                                            subj, f"{subj}_{ch}.nii.gz")
                        imgs.append(nib.load(path).get_fdata(dtype=np.float32))
                    x = torch.tensor(np.stack(imgs)[None]).to(device)  # (1,C,H,W,D)
                    preds = torch.stack([torch.sigmoid(m(x)) for m in models])  # (M,1,L,H,W,D)
                    if use_bald:
                        # BALD = H[E_m[p]] - E_m[H[p_m]]
                        # 모델 간 불일치가 클수록 높은 값 → inter-domain uncertainty
                        p_mean = preds.mean(dim=0).cpu().numpy()          # (1,L,H,W,D)
                        p_all  = preds.cpu().numpy()                       # (M,1,L,H,W,D)
                        h_mean = -(p_mean * np.log(p_mean + eps) + (1 - p_mean) * np.log(1 - p_mean + eps))
                        h_ind  = -(p_all  * np.log(p_all  + eps) + (1 - p_all)  * np.log(1 - p_all  + eps))
                        score  = float((h_mean - h_ind.mean(axis=0)).mean())
                    else:
                        # 단일 모델: 예측 엔트로피
                        p = preds[0].cpu().numpy()                         # (1,L,H,W,D)
                        score = float(-(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps)).mean())
                    scores[subj] = score
                except Exception as e:
                    log.warning("  %s — %s 계산 실패: %s", subj, metric_name, e)
                    scores[subj] = 0.0

        # 점수 내림차순 정렬 → global top-k
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        selected = set(ranked[:k])
        if ranked:
            log.info("%s range — top=%.4f  k-th=%.4f  bottom=%.4f", metric_name.upper(),
                     scores[ranked[0]],
                     scores[ranked[k - 1]] if k <= len(ranked) else scores[ranked[-1]],
                     scores[ranked[-1]])

        df[col] = None
        for p in partitions:
            train_idx = df[(df["Partition_ID"] == p) & (df["TrainOrVal"] == "train")].index
            df.loc[train_idx, col] = [
                1 if df.loc[i, "Subject_ID"] in selected else 0 for i in train_idx
            ]
            n_sel = int(df.loc[train_idx, col].sum())
            log.info("  Partition %s — train=%d  selected=%d", p, len(train_idx), n_sel)

        df[col] = df[col].astype("Int64")
        return df

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

    # ── Aggregation ────────────────────────────────────────────────────────
    def _aggregate(self, state_dicts, n_trains):
        if self.args.algorithm == "fedavg":
            return self._fedavg(state_dicts)
        if self.args.algorithm == "fedwavg":
            return self._fedwavg(state_dicts, n_trains)
        raise ValueError(f"지원하지 않는 알고리즘: {self.args.algorithm}")

    def _fedavg(self, state_dicts):
        avg = {}
        for key in state_dicts[0]:
            avg[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
        return avg

    def _fedwavg(self, state_dicts, n_trains):
        total = sum(n_trains)
        if total == 0:
            log.warning("FedWAvg — 모든 n_train=0, FedAvg로 대체합니다.")
            return self._fedavg(state_dicts)
        weights = [n / total for n in n_trains]
        log.info("FedWAvg weights: %s", [f"{w:.4f}" for w in weights])
        avg = {}
        for key in state_dicts[0]:
            avg[key] = sum(w * sd[key].float() for w, sd in zip(weights, state_dicts))
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
    parser.add_argument("--gpu",            type=int,   default=1,              help="GPU 사용 여부 (1/0) — entropy 추론에 적용")
    parser.add_argument("--sampling-rate",  type=float, default=1.0,            help="train subjects 샘플링 비율 (0.0~1.0)")
    parser.add_argument("--sampling-mode",  default="static",                   help="샘플링 모드 (static / dynamic / pool)")
    parser.add_argument("--selection",      default="random",                   help="subject 선택 방식 (random / entropy)")
    parser.add_argument("--committee",            default="",    help="entropy 선택용 모델 경로 (.pt 또는 폴더, 콤마 구분 복수 경로)")
    parser.add_argument("--committee-job",        default="",    help="committee job 이름 — 체크포인트 구조에서 자동 경로 생성 시 사용")
    parser.add_argument("--committee-rounds",     type=int, default=1, help="committee job의 총 라운드 수")
    parser.add_argument("--committee-round",      type=int, default=0, help="committee로 사용할 라운드 번호 (0-indexed)")
    parser.add_argument("--committee-partitions", default="",    help="committee 기관 ID 목록 (콤마 구분, e.g. '1,2,5,7')")
    parser.add_argument("--committee-in-ch",      type=int, default=4,                  help="committee 모델 입력 채널 수")
    parser.add_argument("--committee-out-classes", type=int, default=1,                 help="committee 모델 출력 클래스 수")
    parser.add_argument("--committee-chan",        default="[t1,t1ce,t2,flair]",         help="committee 모델 입력 채널 (entropy 추론용)")
    parser.add_argument("-D", "--data",     default="/data/fets128/trainval",    help="데이터 경로 (entropy 선택 시 추론에 사용)")
    parser.add_argument("--chan",           default="[t1,t1ce,t2,flair]",        help="입력 채널 (entropy 추론용)")
    parser.add_argument("--algorithm",      default="fedavg",                   help="집계 알고리즘 (fedavg / fedwavg)")
    parser.add_argument("-J", "--job",      default="stage1",                   help="job 이름 (e.g. stage1 → stage1-p01, stage1/agg/)")
    parser.add_argument("--partitions",     default="",                         help="집계할 partition ID 목록 (콤마 구분, e.g. '1,2,5')")
    parser.add_argument("--ckpt-root",      default="/checkpoints",             help="체크포인트 루트 경로")
    parser.add_argument("--runs-root",      default="/runs",                    help="TensorBoard runs 루트 경로")

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
