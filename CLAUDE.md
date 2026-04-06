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
podman build -f builds/Containerfile -t simple-fedpod:latest

# Run — mounts scripts/ into /app so code changes apply without rebuilding
podman run --gpus all -v ./scripts:/app:z simple-fedpod:latest
```

`scripts/` 는 이미지에 복사되지 않으며, 런타임에 바인드 마운트됩니다. 코드 수정 후 재빌드 없이 재실행만 하면 됩니다.

## Architecture

- `scripts/app.py` — CLI entry point (`main()`) that parses args and delegates to `App`. The `App.run()` method is where fedpod logic will live.
- `builds/Containerfile` — Container definition; `scripts/` is bind-mounted at runtime into `/app/` (not copied into the image).

Note: there is no test framework, linter, or dependency manifest configured yet.
