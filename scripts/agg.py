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
        if self.args.selection in ("entropy", "anti_entropy", "bi_entropy"):
            raise ValueError("라운드에서 --selection entropy/anti_entropy/bi_entropy는 지원하지 않습니다. --selection random을 사용하세요.")

        partitions = [int(p.strip()) for p in self.args.partitions.split(",") if p.strip()]
        log.info("Round %d / %d — aggregating %d partitions %s (algorithm=%s)",
                 self.args.round, self.args.rounds, len(partitions), partitions, self.args.algorithm)

        # 각 partition의 best.pt 및 metrics.json 로드
        state_dicts  = []
        n_trains     = []
        metrics_list = []
        part_ids     = []
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
            metrics = {}
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
            n_train = metrics.get("n_train", 0)
            state_dicts.append(ckpt["model"])
            n_trains.append(n_train)
            metrics_list.append(metrics)
            part_ids.append(p)
            log.info("  partition %2d — loaded  val_loss=%.4f  epoch=%d  n_train=%d",
                     p, ckpt.get("val_loss", float("nan")), ckpt.get("epoch", -1), n_train)

        if not state_dicts:
            raise RuntimeError("집계할 체크포인트가 없습니다.")

        # Aggregation
        agg_state = self._aggregate(state_dicts, n_trains, metrics_list, part_ids)
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
        avg_lr       = sum(m["lr"] for m in metrics_list if "lr" in m) / max(1, sum(1 for m in metrics_list if "lr" in m))

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
            # meta
            writer.add_scalar("rnd_meta/lr", avg_lr, r)

        log.info("inst_avg TensorBoard — round=%d  n=%d  prv_loss=%.4f  pst_loss=%.4f"
                 "  prv_dice=%.4f  pst_dice=%.4f",
                 r, len(metrics_list), prv_val, pst_val, prv_dice_avg, pst_dice_avg)

    # ── Split CSV 초기화 (dry-run) ──────────────────────────────────────────
    def _sample_train(self, df, col):
        """전체 기관의 원본 train 수로 공통 λ를 정하고, 기관별로 Poisson(λ) 샘플링.
        pool 컬럼이 있으면 pool>0 subject만 샘플링 대상으로 제한하되 λ는 원본 수 기준.
        pool 값은 0(비선택) 또는 양수(순위 또는 단순 1) 모두 허용.
        rate=1.0 시 λ = mean(ALL_n) ≥ pool_i (pool은 rate=1.0 기준 생성) → clip이 pool 전체 선택 보장."""
        has_pool = "pool" in df.columns
        partitions = sorted(df[df["TrainOrVal"] == "train"]["Partition_ID"].unique())
        per_n = [len(df[(df["Partition_ID"] == p) & (df["TrainOrVal"] == "train")])
                 for p in partitions]
        lam = np.mean(per_n) * self.args.sampling_rate
        log.info("Poisson λ = mean(N)×rate = %.1f×%.2f = %.2f",
                 np.mean(per_n), self.args.sampling_rate, lam)

        df[col] = None
        for p, n_total in zip(partitions, per_n):
            if has_pool:
                pool_idx   = df[(df["Partition_ID"] == p) & (df["TrainOrVal"] == "train") & (df["pool"] > 0)].index
                nopool_idx = df[(df["Partition_ID"] == p) & (df["TrainOrVal"] == "train") & (df["pool"] <= 0)].index
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
        # ── bi_entropy: entropy pool(k/2) + anti-entropy pool(k/2) 생성 ──────
        if self.args.selection == "bi_entropy":
            self._init_split_bi_entropy()
            return

        # ── pool 재사용: 이전 실험 split.csv 의 pool 컬럼을 그대로 사용 ──────
        if self.args.reuse_pool:
            src = pd.read_csv(self.args.reuse_pool)
            if "pool" not in src.columns:
                raise ValueError(f"--reuse-pool CSV에 pool 컬럼이 없습니다: {self.args.reuse_pool}")
            df = src[["Partition_ID", "Subject_ID", "TrainOrVal", "pool"]].copy()
            df["Partition_ID"] = df["Partition_ID"].astype("Int64")
            df["pool"] = df["pool"].astype("Int64")
            n_pool_total  = int((df["pool"] > 0).sum())
            n_train_total = int((df["TrainOrVal"] == "train").sum())
            log.info("Reuse pool ← %s  (%d / %d train subjects)",
                     self.args.reuse_pool, n_pool_total, n_train_total)
            # pool 재사용 시 R00 서브샘플링
            #   entropy/anti_entropy: pool 컬럼 순위(1=최고) 기준 global top-k k-cut
            #                         — 모델 로드/추론 전혀 없음
            #   random: pool 내 per-partition Poisson(λ)
            if self.args.selection in ("entropy", "anti_entropy"):
                pool_max = int(df.loc[df["TrainOrVal"] == "train", "pool"].max(skipna=True))
                if pool_max <= 1:
                    raise ValueError(
                        "reuse-pool CSV의 pool 컬럼에 BALD 순위가 없습니다 (max=1). "
                        "entropy/anti_entropy selection으로 생성된 CSV를 사용하세요."
                    )
                partitions = sorted(df[df["TrainOrVal"] == "train"]["Partition_ID"].dropna().unique())
                per_n = [len(df[(df["Partition_ID"] == p) & (df["TrainOrVal"] == "train")])
                         for p in partitions]
                lam = np.mean(per_n) * self.args.sampling_rate
                k   = int(sum(int(np.clip(round(lam), 1, n)) for n in per_n))
                # pool 컬럼: 1=최고 BALD, N=최저 BALD
                #   entropy    → 오름차순 head(k): top-k (최고 BALD)
                #   anti_entropy → 내림차순 head(k): bottom-k (최저 BALD)
                ascending = (self.args.selection == "entropy")
                pool_rows = df[df["TrainOrVal"] == "train"].sort_values("pool", ascending=ascending)
                selected  = set(pool_rows.head(k)["Subject_ID"])
                train_idx = df[df["TrainOrVal"] == "train"].index
                df.loc[train_idx, "R00"] = df.loc[train_idx, "Subject_ID"].map(
                    lambda s: 1 if s in selected else 0)
                df["R00"] = df["R00"].astype("Int64")
                log.info("Reuse pool %s — λ=%.2f  global k=%d  selected=%d",
                         self.args.selection, lam, k, len(selected))
            else:
                df = self._sample_train(df, "R00")
            n_r00 = int((df["R00"] == 1).sum())
            log.info("Reuse pool — R00: %d / %d pool subjects selected (selection=%s, sampling_rate=%.2f)",
                     n_r00, n_pool_total, self.args.selection, self.args.sampling_rate)
            init_split = os.path.join(self.args.ckpt_root, self.args.job, "agg", "init", "split.csv")
            os.makedirs(os.path.dirname(init_split), exist_ok=True)
            df.to_csv(init_split, index=False)
            log.info("Init split CSV saved → %s", init_split)
            self._write_output("next-split-csv", init_split)
            return

        df = pd.read_csv(self.args.split)[["Partition_ID", "Subject_ID", "TrainOrVal"]]
        df["Partition_ID"] = df["Partition_ID"].astype("Int64")

        if self.args.sampling_mode == "pool":
            if self.args.selection in ("entropy", "anti_entropy"):
                # ── entropy/anti_entropy: 전체 train subject 점수 계산 → 전체 순위 기록 ──
                # pool 컬럼 = 전체 N개 train subject의 global 점수 순위 (1=최고, N=최저)
                # scoring=bald: committee BALD, scoring=texture: ROI intensity entropy
                scoring = getattr(self.args, "scoring", "bald")
                if scoring == "bald":
                    if not (self.args.committee or self.args.committee_job):
                        raise RuntimeError("pool+entropy/anti_entropy+scoring=bald 에는 --committee 또는 --committee-job 이 필요합니다.")
                    device   = torch.device("cuda" if self.args.gpu and torch.cuda.is_available() else "cpu")
                    models   = self._build_committee(device)
                    channels = self.args.committee_chan.strip("[]").split(",")
                else:  # texture: 모델 불필요
                    device, models, channels = None, None, None
                # rate=1.0 으로 호출해도 _pool_tmp 결과는 버리고 pool_scores(전체) 만 사용
                orig_rate = self.args.sampling_rate
                self.args.sampling_rate = 1.0
                df, pool_scores = self._sample_train_entropy(df, "_pool_tmp", models, device, channels)
                self.args.sampling_rate = orig_rate
                df = df.drop(columns=["_pool_tmp"])
                # 전체 train subject에 BALD 순위 부여 (1=최고 BALD, N=최저 BALD)
                df["pool"] = pd.NA
                train_idx = df[df["TrainOrVal"] == "train"].index
                sorted_idx = sorted(train_idx,
                                    key=lambda i: pool_scores.get(df.loc[i, "Subject_ID"], 0),
                                    reverse=True)   # 항상 내림차순: rank 1 = 최고 BALD
                for rank, i in enumerate(sorted_idx, start=1):
                    df.loc[i, "pool"] = rank
                df["pool"] = df["pool"].astype("Int64")
                n_train_total = len(train_idx)
                log.info("Pool — %d train subjects ranked by BALD (selection=%s, 1=highest)",
                         n_train_total, self.args.selection)
                # R00 — k-cut: entropy=top-k(rank 1..k), anti_entropy=bottom-k(rank N-k+1..N)
                df, _ = self._sample_train_entropy(df, "R00", models, device, channels,
                                                   cached_scores=pool_scores)
                n_r00 = int((df["R00"] == 1).sum())
                log.info("Pool R00 — %d / %d subjects selected (selection=%s, sampling_rate=%.2f)",
                         n_r00, n_train_total, self.args.selection, self.args.sampling_rate)
            else:
                # ── random: rate=1.0 으로 pool subset 선택 (pool=1/0) ──
                orig_rate = self.args.sampling_rate
                self.args.sampling_rate = 1.0
                df = self._sample_train(df, "_pool_tmp")
                self.args.sampling_rate = orig_rate
                non_pool = (df["TrainOrVal"] == "train") & (df["_pool_tmp"] == 0)
                df["pool"] = pd.NA
                df.loc[df["TrainOrVal"] == "train", "pool"] = 1
                df.loc[non_pool, "pool"] = 0
                df["pool"] = df["pool"].astype("Int64")
                df = df.drop(columns=["_pool_tmp"])
                n_pool_total = int((df["pool"] > 0).sum())
                n_train_total = int((df["TrainOrVal"] == "train").sum())
                log.info("Pool — %d / %d train subjects in pool (random)",
                         n_pool_total, n_train_total)
                df = self._sample_train(df, "R00")
                n_r00 = int((df["R00"] == 1).sum())
                log.info("Pool R00 — %d / %d pool subjects selected (random, sampling_rate=%.2f)",
                         n_r00, n_pool_total, self.args.sampling_rate)
        else:
            # static / dynamic 공통 기본 흐름
            if self.args.selection in ("entropy", "anti_entropy"):
                if not (self.args.committee or self.args.committee_job):
                    log.warning("--selection %s 이지만 --committee 미지정 → random으로 대체", self.args.selection)
                    df = self._sample_train(df, "R00")
                else:
                    device = torch.device("cuda" if self.args.gpu and torch.cuda.is_available() else "cpu")
                    models = self._build_committee(device)
                    channels = self.args.committee_chan.strip("[]").split(",")
                    df, _ = self._sample_train_entropy(df, "R00", models, device, channels)
            else:
                df = self._sample_train(df, "R00")

        init_split = os.path.join(self.args.ckpt_root, self.args.job, "agg", "init", "split.csv")
        df.to_csv(init_split, index=False)
        log.info("Init split CSV saved → %s", init_split)
        self._write_output("next-split-csv", init_split)

    def _init_split_bi_entropy(self):
        """bi_entropy dry-run: entropy pool (k/2) + anti-entropy pool (k/2) 각각 생성.

        entropy pool     → {ckpt_root}/{job}/agg/init/split.csv       (next-split-csv, FL r5~)
        anti-entropy pool → {ckpt_root}/{job}/agg/init-anti/split.csv (next-split-csv-anti, warm-up r0~r4)
        """
        orig_rate = self.args.sampling_rate
        half_rate = orig_rate / 2

        if self.args.reuse_pool:
            src = pd.read_csv(self.args.reuse_pool)
            if "pool" not in src.columns:
                raise ValueError(f"--reuse-pool CSV에 pool 컬럼이 없습니다: {self.args.reuse_pool}")
            df = src[["Partition_ID", "Subject_ID", "TrainOrVal", "pool"]].copy()
            df["Partition_ID"] = df["Partition_ID"].astype("Int64")
            df["pool"] = df["pool"].astype("Int64")
            pool_max = int(df.loc[df["TrainOrVal"] == "train", "pool"].max(skipna=True))
            if pool_max <= 1:
                raise ValueError(
                    "reuse-pool CSV의 pool 컬럼에 BALD 순위가 없습니다 (max=1). "
                    "entropy/anti_entropy selection으로 생성된 CSV를 사용하세요."
                )
            partitions = sorted(df[df["TrainOrVal"] == "train"]["Partition_ID"].dropna().unique())
            per_n = [len(df[(df["Partition_ID"] == p) & (df["TrainOrVal"] == "train")])
                     for p in partitions]
            train_idx = df[df["TrainOrVal"] == "train"].index

            def _kcut(ascending, rate):
                lam = np.mean(per_n) * rate
                k   = int(sum(int(np.clip(round(lam), 1, n)) for n in per_n))
                pool_rows = df[df["TrainOrVal"] == "train"].sort_values("pool", ascending=ascending)
                selected  = set(pool_rows.head(k)["Subject_ID"])
                d = df.copy()
                d.loc[train_idx, "R00"] = d.loc[train_idx, "Subject_ID"].map(
                    lambda s: 1 if s in selected else 0)
                d["R00"] = d["R00"].astype("Int64")
                return d, k

            df_entropy, k_e = _kcut(ascending=True,  rate=half_rate)  # entropy: top-k/2
            df_anti,    k_a = _kcut(ascending=False, rate=half_rate)   # anti-entropy: bottom-k/2
            log.info("bi_entropy reuse-pool — entropy k=%d  anti_entropy k=%d", k_e, k_a)

        else:
            df = pd.read_csv(self.args.split)[["Partition_ID", "Subject_ID", "TrainOrVal"]]
            df["Partition_ID"] = df["Partition_ID"].astype("Int64")

            scoring = getattr(self.args, "scoring", "bald")
            if scoring == "bald":
                if not (self.args.committee or self.args.committee_job):
                    raise RuntimeError(
                        "bi_entropy+bald 에는 --committee 또는 --committee-job 이 필요합니다.")
                device   = torch.device("cuda" if self.args.gpu and torch.cuda.is_available() else "cpu")
                models   = self._build_committee(device)
                channels = self.args.committee_chan.strip("[]").split(",")
            else:
                device, models, channels = None, None, None

            # BALD 점수 계산 (sampling_rate=1.0, 전체 train 대상)
            self.args.sampling_rate = 1.0
            self.args.selection = "entropy"   # 점수 계산 방향 무관, entropy로 임시 설정
            df, pool_scores = self._sample_train_entropy(df, "_pool_tmp", models, device, channels)
            self.args.sampling_rate = orig_rate
            self.args.selection = "bi_entropy"
            df = df.drop(columns=["_pool_tmp"])

            # pool 순위 부여 (1=최고 BALD, N=최저 BALD)
            df["pool"] = pd.NA
            train_idx  = df[df["TrainOrVal"] == "train"].index
            sorted_idx = sorted(train_idx,
                                key=lambda i: pool_scores.get(df.loc[i, "Subject_ID"], 0),
                                reverse=True)
            for rank, i in enumerate(sorted_idx, start=1):
                df.loc[i, "pool"] = rank
            df["pool"] = df["pool"].astype("Int64")
            log.info("bi_entropy pool — %d train subjects ranked (1=highest BALD)", len(train_idx))

            # entropy pool (top-k/2)
            self.args.sampling_rate = half_rate
            self.args.selection = "entropy"
            df_entropy = df.copy()
            df_entropy, _ = self._sample_train_entropy(df_entropy, "R00", models, device, channels,
                                                       cached_scores=pool_scores)
            # anti-entropy pool (bottom-k/2)
            self.args.selection = "anti_entropy"
            df_anti = df.copy()
            df_anti, _ = self._sample_train_entropy(df_anti, "R00", models, device, channels,
                                                    cached_scores=pool_scores)
            self.args.selection = "bi_entropy"
            self.args.sampling_rate = orig_rate

        # entropy pool → 표준 경로 (FL r5~ 사용)
        init_split = os.path.join(self.args.ckpt_root, self.args.job, "agg", "init", "split.csv")
        os.makedirs(os.path.dirname(init_split), exist_ok=True)
        df_entropy.to_csv(init_split, index=False)
        log.info("bi_entropy — entropy pool saved → %s", init_split)
        self._write_output("next-split-csv", init_split)

        # anti-entropy pool → init-anti 경로 (warm-up r0~r4 사용)
        init_split_anti = os.path.join(
            self.args.ckpt_root, self.args.job, "agg", "init-anti", "split.csv")
        os.makedirs(os.path.dirname(init_split_anti), exist_ok=True)
        df_anti.to_csv(init_split_anti, index=False)
        log.info("bi_entropy — anti-entropy pool saved → %s", init_split_anti)
        self._write_output("next-split-csv-anti", init_split_anti)

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
            if self.args.selection in ("entropy", "anti_entropy"):
                raise ValueError("라운드에서 --selection entropy/anti_entropy는 지원하지 않습니다. --selection random을 사용하세요.")
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

    def _score_texture(self, subj, channels):
        """ROI (tumor mask, _sub.nii.gz > 0) 내 채널별 intensity entropy 평균.

        entropy = -Σ p(i) log p(i)  (ROI 강도 히스토그램, 64 bins)
        값이 클수록 ROI 내 강도 분포가 복잡(heterogeneous) → 내재적 난이도 높음.
        """
        import nibabel as nib

        seg_path = os.path.join(self.args.data, subj, f"{subj}_sub.nii.gz")
        try:
            seg  = nib.load(seg_path).get_fdata(dtype=np.float32)
            mask = seg > 0
        except Exception as e:
            log.warning("  texture — %s seg 로드 실패: %s  → score=0", subj, e)
            return 0.0

        if not mask.any():
            return 0.0

        entropies = []
        for ch in channels:
            path = os.path.join(self.args.data, subj, f"{subj}_{ch}.nii.gz")
            try:
                img = nib.load(path).get_fdata(dtype=np.float32)
            except Exception as e:
                log.warning("  texture — %s %s 로드 실패: %s", subj, ch, e)
                continue
            roi = img[mask]
            rng = roi.max() - roi.min()
            if rng < 1e-8:
                entropies.append(0.0)
                continue
            roi_norm = (roi - roi.min()) / rng
            hist, _  = np.histogram(roi_norm, bins=64, range=(0.0, 1.0))
            hist     = hist.astype(np.float64)
            hist    /= hist.sum() + 1e-8
            hist     = hist[hist > 0]
            entropies.append(float(-np.sum(hist * np.log(hist))))
        return float(np.mean(entropies)) if entropies else 0.0

    def _sample_train_entropy(self, df, col, models, device, channels,
                               cached_scores=None):
        """scoring 방식(bald / texture)에 따라 전체 train subject 점수 계산 → global top-k 선택.

        scoring=bald   : committee 모델 앙상블 BALD (inter-model uncertainty)
        scoring=texture: ROI 내 intensity entropy (데이터 내재적 복잡도, 모델 불필요)

        cached_scores: {subject_id: score} 딕셔너리를 전달하면 재계산 없이 재사용.
        반환값: (df, scores)
        """
        import nibabel as nib

        scoring = getattr(self.args, "scoring", "bald")

        partitions = sorted(df[df["TrainOrVal"] == "train"]["Partition_ID"].unique())
        per_n = [len(df[(df["Partition_ID"] == p) & (df["TrainOrVal"] == "train")])
                 for p in partitions]
        lam = np.mean(per_n) * self.args.sampling_rate
        k   = int(sum(int(np.clip(round(lam), 1, n)) for n in per_n))

        use_bald    = (scoring == "bald") and (models is not None and len(models) > 1)
        metric_name = "BALD" if use_bald else ("entropy" if scoring == "bald" else "texture")
        n_committee = len(models) if models is not None else 0
        log.info("%s scoring — committee=%d  λ=%.2f  global k=%d  (total train=%d)",
                 metric_name.upper(), n_committee, lam, k, sum(per_n))

        if cached_scores is not None:
            scores = cached_scores
            log.info("%s — cached scores 재사용 (%d subjects)", metric_name.upper(), len(scores))
        else:
            train_rows   = df[df["TrainOrVal"] == "train"]
            scores       = {}
            n_total      = len(train_rows)
            log_interval = max(1, n_total // 10)

            if scoring == "texture":
                # ── texture: ROI intensity entropy (모델 추론 없음) ─────────────
                chan_list = self.args.chan.strip("[]").split(",")
                for i, (_, row) in enumerate(train_rows.iterrows()):
                    subj = row["Subject_ID"]
                    if i % log_interval == 0:
                        log.info("  TEXTURE scoring — %d / %d", i, n_total)
                    scores[subj] = self._score_texture(subj, chan_list)
            else:
                # ── bald: committee 모델 앙상블 uncertainty ───────────────────
                eps = 1e-6
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
                            x     = torch.tensor(np.stack(imgs)[None]).to(device)
                            preds = torch.stack([torch.sigmoid(m(x)) for m in models])
                            if use_bald:
                                p_mean = preds.mean(dim=0).cpu().numpy()
                                p_all  = preds.cpu().numpy()
                                h_mean = -(p_mean * np.log(p_mean + eps) + (1 - p_mean) * np.log(1 - p_mean + eps))
                                h_ind  = -(p_all  * np.log(p_all  + eps) + (1 - p_all)  * np.log(1 - p_all  + eps))
                                score  = float((h_mean - h_ind.mean(axis=0)).mean())
                            else:
                                p     = preds[0].cpu().numpy()
                                score = float(-(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps)).mean())
                            scores[subj] = score
                        except Exception as e:
                            log.warning("  %s — %s 계산 실패: %s", subj, metric_name, e)
                            scores[subj] = 0.0

        # anti_entropy: 하위 k, 그 외: 상위 k
        descending = (self.args.selection != "anti_entropy")
        ranked   = sorted(scores, key=scores.__getitem__, reverse=descending)
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
        return df, scores

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
    def _aggregate(self, state_dicts, n_trains, metrics_list, part_ids):
        if self.args.algorithm == "fedavg":
            return self._fedavg(state_dicts)
        if self.args.algorithm == "fedwavg":
            return self._fedwavg(state_dicts, n_trains)
        if self.args.algorithm == "fedpod":
            return self._fedpod(state_dicts, metrics_list, part_ids)
        if self.args.algorithm == "fedpid":
            return self._fedpid(state_dicts, metrics_list, part_ids)
        if self.args.algorithm == "fedbn":
            return self._fedbn(state_dicts)
        raise ValueError(f"지원하지 않는 알고리즘: {self.args.algorithm}")

    def _fedbn(self, state_dicts):
        """FedBN — norm 레이어는 로컬 유지, 나머지는 FedAvg."""
        avg = {}
        for key in state_dicts[0]:
            if any(t in key for t in (".norm.", ".bn.", ".in.")):
                avg[key] = state_dicts[0][key].float()   # 첫 번째 partition 값 유지 (로컬)
            else:
                avg[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
        return avg

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

    def _fedpod(self, state_dicts, metrics_list, part_ids):
        kp = self.args.kp
        ki = self.args.ki
        kd = self.args.kd

        # 이전 라운드 POD state 로드 (I항)
        pod_state = self._load_pod_state()
        integrals = pod_state.get("integrals", {})

        pod_values = []
        for m, p in zip(metrics_list, part_ids):
            key = str(p)
            n_train = m.get("n_train", 1)
            prv = m.get("prv_val_loss", 0.0)
            pst = m.get("pst_val_loss", prv)   # pst 없으면 개선량=0
            improvement = max(0.0, prv - pst)   # 양수 = 개선 (음수 클램프)
            area        = (prv + pst) / 2       # 라운드 내 손실 밑넓이 (사다리꼴)

            proportional = kp * n_train         # P: 학습 데이터 수 비례
            integral     = integrals.get(key, 0.0) + ki * area
            derivative   = kd * improvement     # D: 라운드 내 개선량 (≥0)

            pod = proportional + integral + derivative
            pod_values.append(pod)

            # state 갱신
            integrals[key] = integral

            log.info("  FedPOD partition %2d — n_train=%d prv=%.4f pst=%.4f impr=%.4f area=%.4f "
                     "P=%.4f I=%.4f D=%.4f pod=%.4f",
                     p, n_train, prv, pst, improvement, area,
                     proportional, integral, derivative, pod)

        # softmax 정규화
        import math
        max_pod = max(pod_values)
        exp_vals = [math.exp(v - max_pod) for v in pod_values]  # numerical stability
        total_exp = sum(exp_vals)
        weights = [e / total_exp for e in exp_vals]
        log.info("FedPOD weights: %s", [f"{w:.4f}" for w in weights])

        # POD state 저장 (다음 라운드 사용)
        self._save_pod_state({"integrals": integrals})

        avg = {}
        for key in state_dicts[0]:
            avg[key] = sum(w * sd[key].float() for w, sd in zip(weights, state_dicts))
        return avg

    def _pod_state_path(self):
        agg_dir = os.path.dirname(self._agg_path(self.args.round))
        return os.path.join(agg_dir, "pod_state.json")

    def _load_pod_state(self):
        if self.args.round == 0:
            return {}
        prev_agg_dir = os.path.dirname(self._agg_path(self.args.round - 1))
        path = os.path.join(prev_agg_dir, "pod_state.json")
        if not os.path.exists(path):
            log.warning("FedPOD — 이전 라운드 pod_state.json 없음, 초기화합니다: %s", path)
            return {}
        with open(path) as f:
            return json.load(f)

    def _save_pod_state(self, state):
        path = self._pod_state_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        log.info("FedPOD state saved → %s", path)

    # ── FedPID ─────────────────────────────────────────────────────────────
    def _fedpid(self, state_dicts, metrics_list, part_ids):
        kp = self.args.kp
        ki = self.args.ki
        kd = self.args.kd

        pid_state   = self._load_pid_state()
        trn_history = pid_state.get("trn_loss_history", {})  # partition → list (최대 5)

        pid_values = []
        for m, p in zip(metrics_list, part_ids):
            key     = str(p)
            n_train = m.get("n_train", 1)
            curr_trn = m.get("avg_trn_loss", 0.0)
            history  = trn_history.get(key, [])

            proportional = kp * n_train
            integral     = ki * sum(history[-5:])                          # 최근 5 라운드 누적
            prev_trn     = history[-1] if history else curr_trn
            derivative   = kd * (prev_trn - curr_trn)                     # 양수 = 손실 감소 중

            pid = proportional + integral + derivative
            pid_values.append(pid)

            trn_history[key] = (history + [curr_trn])[-5:]                # 히스토리 갱신

            log.info("  FedPID partition %2d — n_train=%d trn=%.4f prev_trn=%.4f "
                     "P=%.4f I=%.4f D=%.4f pid=%.4f",
                     p, n_train, curr_trn, prev_trn,
                     proportional, integral, derivative, pid)

        import math
        max_pid  = max(pid_values)
        exp_vals = [math.exp(v - max_pid) for v in pid_values]
        total_exp = sum(exp_vals)
        weights  = [e / total_exp for e in exp_vals]
        log.info("FedPID weights: %s", [f"{w:.4f}" for w in weights])

        self._save_pid_state({"trn_loss_history": trn_history})

        avg = {}
        for key in state_dicts[0]:
            avg[key] = sum(w * sd[key].float() for w, sd in zip(weights, state_dicts))
        return avg

    def _pid_state_path(self):
        agg_dir = os.path.dirname(self._agg_path(self.args.round))
        return os.path.join(agg_dir, "pid_state.json")

    def _load_pid_state(self):
        if self.args.round == 0:
            return {}
        prev_agg_dir = os.path.dirname(self._agg_path(self.args.round - 1))
        path = os.path.join(prev_agg_dir, "pid_state.json")
        if not os.path.exists(path):
            log.warning("FedPID — 이전 라운드 pid_state.json 없음, 초기화합니다: %s", path)
            return {}
        with open(path) as f:
            return json.load(f)

    def _save_pid_state(self, state):
        path = self._pid_state_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        log.info("FedPID state saved → %s", path)

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
    parser.add_argument("--selection",      default="random",                   help="subject 선택 방식 (random / entropy / anti_entropy)")
    parser.add_argument("--scoring",        default="bald",                     help="entropy 계열 선택 시 점수 계산 방식 (bald / texture)")
    parser.add_argument("--committee",            default="",    help="entropy 선택용 모델 경로 (.pt 또는 폴더, 콤마 구분 복수 경로)")
    parser.add_argument("--committee-job",        default="",    help="committee job 이름 — 체크포인트 구조에서 자동 경로 생성 시 사용")
    parser.add_argument("--committee-rounds",     type=int, default=1, help="committee job의 총 라운드 수")
    parser.add_argument("--committee-round",      type=int, default=0, help="committee로 사용할 라운드 번호 (0-indexed)")
    parser.add_argument("--committee-partitions", default="",    help="committee 기관 ID 목록 (콤마 구분, e.g. '1,2,5,7')")
    parser.add_argument("--committee-in-ch",      type=int, default=4,                  help="committee 모델 입력 채널 수")
    parser.add_argument("--committee-out-classes", type=int, default=1,                 help="committee 모델 출력 클래스 수")
    parser.add_argument("--committee-chan",        default="[t1,t1ce,t2,flair]",         help="committee 모델 입력 채널 (entropy 추론용)")
    parser.add_argument("--reuse-pool",            default="",                            help="pool 재사용 — 이전 실험 split.csv 경로 (pool 컬럼 필요, entropy inference 스킵)")
    parser.add_argument("-D", "--data",     default="/data/fets128/trainval",    help="데이터 경로 (entropy 선택 시 추론에 사용)")
    parser.add_argument("--chan",           default="[t1,t1ce,t2,flair]",        help="입력 채널 (entropy 추론용)")
    parser.add_argument("--algorithm",      default="fedavg",                   help="집계 알고리즘 (fedavg / fedwavg / fedpod / fedpid / fedbn)")
    parser.add_argument("--kp",             type=float, default=1.0,            help="FedPOD/FedPID — proportional gain")
    parser.add_argument("--ki",             type=float, default=0.1,            help="FedPOD/FedPID — integral gain")
    parser.add_argument("--kd",             type=float, default=0.0,            help="FedPOD/FedPID — derivative gain")
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
