# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FL(Federated Learning) client pod for 3D brain tumor segmentation (FeTS dataset). Trains a 3D Residual U-Net with BCE+SoftDice loss on NIfTI volumes. Supports multi-round FL with FedAvg aggregation, Poisson-based subject sampling, and Katib HPO.

## Running the Application

```bash
# dry-run: 초기 모델 + split CSV 생성
python3 scripts/agg.py --dry-run -J stage1 --ckpt-root ./checkpoints -c ./experiments/fets/partition2/fets_split.csv --sampling-rate 1.0 --sampling-mode static

# round 0 학습
python3 scripts/app.py -D ./data/fets128/trainval -c ./checkpoints/stage1/agg/init/split.csv -P 1 -J stage1 --ckpt-root ./checkpoints --rounds 1 --round 0 -E 3 --init-ckpt ./checkpoints/stage1/agg/init/agg.pt

# round 0 집계
python3 scripts/agg.py -J stage1 --ckpt-root ./checkpoints -c ./checkpoints/stage1/agg/init/split.csv --rounds 1 --round 0 --epochs 3 --partitions 1,2
```

## Container

NVIDIA CUDA 12.8.1 + cuDNN, Ubuntu 24.04, PyTorch with CUDA 12.8.

```bash
# Argo 이미지 빌드 (scripts/ 는 NFS 마운트, COPY 없음)
podman build -f argo/Containerfile -t argo-fedpod:v0.2 .

# Katib 이미지 빌드 (scripts/ COPY 포함, 프로젝트 루트에서 실행)
podman build -f katib/Containerfile -t simple-fedpod:katib .

# 로컬 실행 (Podman GPU)
sudo podman run \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  -v /lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:ro \
  -e LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  --shm-size=8g \
  -v ./scripts:/app:z -v ./data:/data:z \
  -v ./experiments:/experiments:z -v ./checkpoints:/checkpoints:z \
  192.168.0.80:30002/dwnkim/argo-fedpod:v0.2
```

## Architecture

```
scripts/
  app.py          # CLI entry point — args, App class, run() flow
                  #   · 라운드당 prv/pst val_loss + hard Dice 평가
                  #   · metrics.json 저장 (ckpt_dir/)
                  #   · TensorBoard 기록 → runs/{job}/inst{PP}/
  agg.py          # Aggregator — FedAvg, dry-run, Poisson split sampling
                  #   · 라운드 집계 후 inst_avg TensorBoard 기록
                  #     → runs/{job}/inst_avg/
  gen_eval.py     # Generalization eval — agg.pt 를 모든 기관 val에 평가
                  #   · train-job 과 병렬 실행 (stage2 DAG)
                  #   · val 대상: CSV Partition_ID notna() 전체 val subject
                  #   · TensorBoard 기록 → runs/{job}/gen_eval/
  dataset.py      # FeTSDataset, load_split (round-aware)
  trainer.py      # Trainer — BCE+SoftDice, Adam, checkpoint save/resume
                  #   · eval()      — val loss only (prv/pst 측정용)
                  #   · eval_dice() — hard Dice (sigmoid>0.5, 클래스별)
  models/
    unet3d.py     # 3D Residual U-Net (dynamic: channels, block, norm)
    loss.py       # SoftDiceBCEWithLogitsLoss
argo/
  Containerfile   # Argo image — scripts/ NFS 마운트, tensorboard 포함
  workflow.yaml   # single job test
  stage1.yaml     # stage1 FL DAG — 5 rounds, dynamic+random, inst best.pt
  stage1-test.yaml  # 단순 테스트용 (소규모)
  stage2.yaml     # stage2 FL DAG — 5 rounds, pool+entropy/random, FedAvg
                  #   · eval-job: 매 라운드 gen_eval 병렬 실행
                  #   · --selection / --agg-sampling-mode 파라미터화
katib/
  Containerfile   # katib image — COPYs scripts/; build from project root
  experiment.yaml # Katib HPO Experiment (lr, batch / multivariate-tpe)
analysis/
  split_analysis.ipynb  # split CSV 시각화
```

### FL Round Flow

```
agg.py --dry-run   → checkpoints/{job}/agg/init/agg.pt + split.csv (R00)
app.py --round 0   → checkpoints/{job}/inst{PP}/R{RR}r00/ (best.pt, latest.pt)
agg.py --round 0   → checkpoints/{job}/agg/R{RR}r00/agg.pt + split.csv (R01)
app.py --round 1   → checkpoints/{job}/inst{PP}/R{RR}r01/
...
```

### Checkpoint Path

`{ckpt_root}/{job}/inst{partition:02d}/R{rounds:02d}r{round:02d}/`

### Split CSV

원본에서 `Partition_ID`, `Subject_ID`, `TrainOrVal` 3컬럼만 유지. 라운드마다 `R{DD}` 컬럼 추가 (train 선택=1, 미선택=0, val=빈칸).

Poisson 샘플링: `k ~ Poisson(λ)`, `λ = mean(N_per_partition) × sampling_rate`  
- `static`: dry-run 1회 draw 후 고정  
- `dynamic`: 매 라운드 재추출  
- `pool`: **dry-run 전용**. entropy/random으로 top-k 선정 → R00에 직접 기록, 비선택 subject는 `Partition_ID=NA`로 마스킹.  
  이후 라운드는 `static` 또는 `dynamic`으로 지정해야 하며, `Partition_ID.notna()` 필터로 pool 내에서만 샘플링됨.  
  라운드에서 `--sampling-mode pool` 지정 시 에러.

### Subject 선택 방식 (`--selection`)

- `random`: 기관별 Poisson(λ) 무작위 샘플링  
- `entropy`: **dry-run 전용**. uncertainty 기반 global top-k 선택.  
  - **committee > 1**: **BALD** = `H[E_m[p]] - E_m[H[p_m]]` — 모델 간 불일치(inter-domain uncertainty) 측정  
  - committee 미지정 시 random으로 대체 (warning 출력)  
  - 라운드에서 지정 시 에러 (committee 미갱신으로 의미 없음)

### Data Pipeline (`App.run()`)

```
load_split(split_csv, partition_id, round_idx)  → train/val subject lists
FeTSDataset(data_dir, subjects, ...)
  {subject}_{ch}.nii.gz × C   → image (C, 128, 128, 128)
  {subject}_sub.nii.gz + lgrp → label (L, 128, 128, 128)
DataLoader → batches (B, C/L, 128, 128, 128)
```

- 레이블: `sub.nii.gz` subregion 값을 `lgrp`로 OR 조합해 binary mask
- `--epoch` (epoch offset): round 간 global epoch 번호 연속성 유지
- `--init-ckpt`: 이전 round agg.pt 경로 → 학습 전 모델 가중치 로드

### Metrics & TensorBoard

**metrics.json** (`{ckpt_dir}/metrics.json`):
```json
{
  "partition": 1, "round": 0, "n_train": 42,
  "avg_trn_loss": 0.5, "avg_val_loss": 0.6,
  "prv_val_loss": 0.7, "pst_val_loss": 0.55,
  "prv_dice": {"wt": 0.82, "tc": 0.71, "et": 0.65},
  "pst_dice": {"wt": 0.85, "tc": 0.74, "et": 0.68},
  "prv_dice_avg": 0.727, "pst_dice_avg": 0.757
}
```

**TensorBoard 태그 구조** `(ech|rnd)_(trn|val)_(loss|dice)_(avg|prv|pst|prvpst)/(avg|{name})`:
```
rnd_trn_loss_avg/avg                  ← 라운드 평균 train loss
rnd_val_loss_(avg|prv|pst|prvpst)/avg ← 라운드 단위 val loss
rnd_val_dice_(prv|pst|prvpst)/(avg|wt|tc|et)  ← 라운드 단위 val dice
ech_val_loss_(avg|prv|pst|prvpst)/avg ← epoch 단위 val loss
ech_val_dice_(prv|pst|prvpst)/(avg|wt|tc|et)  ← epoch 단위 val dice
```

**runs 디렉토리 구조**:
```
runs/{job}/
  inst{PP}/    ← 기관별 (app.py 기록)
  inst_avg/    ← 기관 평균 (agg.py 기록, 라운드 집계 후)
  gen_eval/    ← 일반화 평가 (gen_eval.py 기록, train과 병렬)
```

`inst_avg`와 `gen_eval`은 동일한 태그명을 사용하므로 TensorBoard에서 두 run을 선택하면 자동 오버레이 비교 가능.  
`gen_eval`은 `prv`/`prvpst` 패널만 기록 (fine-tuning 전 글로벌 모델 평가).

### stage2 DAG

```
dry-run ──┬── train-r0 (9 pods) ── agg-r0 ──┬── train-r1 ... ── agg-r4
          └── eval-r0 (gen_eval)             └── eval-r1 ...    eval-r4
```

Note: test framework, linter, dependency manifest 미설정.
