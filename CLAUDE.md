# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A skeleton Python application framework for a federated learning pod ("fedpod"), designed to run in GPU-enabled containers. The codebase is in early development — `App.run()` is currently a stub.

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
# Build — dev (scripts/ 는 런타임 마운트)
podman build -f builds/Containerfile -t simple-fedpod:dev .

# Build — katib (scripts/ 를 이미지에 COPY, 반드시 프로젝트 루트에서 실행)
podman build -f katib/Containerfile -t simple-fedpod:katib .

# Run
podman run --gpus 1 \
  -v ./scripts:/app:z \
  -v ./data:/data:z \
  -v ./experiments:/experiments:z \
  -v ./checkpoints:/checkpoints:z \
  simple-fedpod:dev
```

| 호스트 경로 | 컨테이너 경로 | 용도 |
|-------------|---------------|------|
| `./scripts` | `/app`        | 코드 (재빌드 없이 수정 반영) |
| `./data`    | `/data`       | 학습 데이터 (`-D`) |
| `./experiments` | `/experiments` | 분할 CSV (`-c`) |
| `./checkpoints` | `/checkpoints` | 체크포인트 저장/로드 (`-J`) |

args 기본값은 컨테이너 내부 경로(`/data/...`, `/experiments/...`) 기준입니다.

## Architecture

```
scripts/
  app.py          # CLI entry point — args, App class, run() flow
  dataset.py      # FeTSDataset, load_split
  trainer.py      # training loop (not yet implemented)
  models/
    unet3d.py     # 3D Residual U-Net (dynamic: channels, block, norm)
builds/
  Containerfile   # bind-mounts scripts/ at runtime, does not copy code
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

Note: model, training loop 미구현. test framework, linter, dependency manifest 미설정.
