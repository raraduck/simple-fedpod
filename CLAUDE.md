# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FL(Federated Learning) client pod for 3D brain tumor segmentation (FeTS dataset). Trains a 3D Residual U-Net with BCE+SoftDice loss on NIfTI volumes, with checkpoint save/resume and Katib HPO support.

## Running the Application

```bash
# Run directly
python3 scripts/app.py

# Run with help
python3 scripts/app.py --help
```

## Container

The container uses NVIDIA CUDA 12.8.1 with cuDNN on Ubuntu 24.04, installs PyTorch with CUDA 12.8 support.

```bash
# Argo 이미지 빌드 (scripts/ 는 NFS 마운트, COPY 없음)
podman build -f argo/Containerfile -t argo-fedpod:v0.1 .

# Katib 이미지 빌드 (scripts/ COPY 포함, 프로젝트 루트에서 실행)
podman build -f katib/Containerfile -t simple-fedpod:katib .

# 로컬 실행 (Argo 이미지, scripts/ 바인드 마운트)
# --device /dev/nvidia* : GPU 디바이스 직접 지정 (Podman에서 --gpus 사용 불가)
# --shm-size=8g : DataLoader num_workers>0 사용 시 shared memory 필요
sudo podman run \
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-uvm \
  --shm-size=8g \
  -v ./scripts:/app:z \
  -v ./data:/data:z \
  -v ./experiments:/experiments:z \
  -v ./checkpoints:/checkpoints:z \
  192.168.0.80:30002/dwnkim/argo-fedpod:v0.1
```

| 이미지 | 코드 | 용도 |
|--------|------|------|
| `argo-fedpod:v0.1` | NFS `/app` 마운트 | Argo Workflow 학습 |
| `simple-fedpod:katib` | 이미지에 COPY | Katib HPO trial |

args 기본값은 컨테이너 내부 경로(`/data/...`, `/experiments/...`, `/checkpoints`) 기준입니다.

## Architecture

```
scripts/
  app.py          # CLI entry point — args, App class, run() flow
  dataset.py      # FeTSDataset, load_split
  trainer.py      # Trainer — BCE+SoftDice, Adam, checkpoint save/resume
  models/
    unet3d.py     # 3D Residual U-Net (dynamic: channels, block, norm)
    loss.py       # SoftDiceBCEWithLogitsLoss
argo/
  Containerfile   # Argo image — scripts/ mounted from NFS at runtime
  workflow.yaml   # Argo Workflow definition
katib/
  Containerfile   # katib image — COPYs scripts/ in; build from project root
  experiment.yaml # Katib HPO Experiment (lr, batch / multivariate-tpe)
```

### Data pipeline (`App.run()`)

```
load_split(split_csv, partition_id)   CSV → train/val subject lists
FeTSDataset(data_dir, subjects, ...)  subject list → Dataset
  __getitem__:
    {subject}_{ch}.nii.gz × C        → image tensor (C, 128, 128, 128)
    {subject}_sub.nii.gz + lgrp      → label tensor (L, 128, 128, 128)
DataLoader(dataset, batch_size, ...)  → batches (B, C/L, 128, 128, 128)
```

- 입력 채널(`-C`)은 t1/t1ce/t2/flair 외 seg도 추가 가능
- 레이블은 `sub.nii.gz`의 subregion 값을 `lgrp`로 OR 조합해 binary mask 생성
- `-P`(Partition ID)로 CSV에서 해당 FL 클라이언트의 subject를 필터링

Note: test framework, linter, dependency manifest 미설정.
