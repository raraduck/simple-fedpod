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
# Build (once)
podman build -f builds/Containerfile -t simple-fedpod:dev

# Run — 세 디렉터리를 각각 마운트
podman run --gpus 1 \
  -v ./scripts:/app:z \
  -v ./data:/data:z \
  -v ./experiments:/experiments:z \
  simple-fedpod:dev
```

| 호스트 경로 | 컨테이너 경로 | 용도 |
|-------------|---------------|------|
| `./scripts` | `/app`        | 코드 (재빌드 없이 수정 반영) |
| `./data`    | `/data`       | 학습 데이터 (`-D`) |
| `./experiments` | `/experiments` | 분할 CSV (`-c`) |

args 기본값은 컨테이너 내부 경로(`/data/...`, `/experiments/...`) 기준입니다.

## Architecture

- `scripts/app.py` — CLI entry point (`main()`) that parses args and delegates to `App`.
- `builds/Containerfile` — Container definition; `scripts/` is bind-mounted at runtime into `/app/` (not copied into the image).

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
