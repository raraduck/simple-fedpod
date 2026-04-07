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
python3 scripts/agg.py -J stage1 --ckpt-root ./checkpoints -c ./checkpoints/stage1/agg/init/split.csv --rounds 1 --round 0 --epochs 3 --num-partitions 2
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
  agg.py          # Aggregator — FedAvg, dry-run, Poisson split sampling
  dataset.py      # FeTSDataset, load_split (round-aware)
  trainer.py      # Trainer — BCE+SoftDice, Adam, checkpoint save/resume
  models/
    unet3d.py     # 3D Residual U-Net (dynamic: channels, block, norm)
    loss.py       # SoftDiceBCEWithLogitsLoss
argo/
  Containerfile   # Argo image — scripts/ mounted from NFS at runtime
  workflow.yaml   # single job test
  stage1.yaml     # parallel FL DAG (withItems)
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

Note: test framework, linter, dependency manifest 미설정.
