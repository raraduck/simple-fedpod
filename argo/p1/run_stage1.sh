#!/bin/bash
set -euo pipefail

# stage1-p1: partition1 (23개 기관) committee 모델 생성
# 완료 후 생성되는 split.csv 경로:
#   /checkpoints/stage1-p1/agg/init/split.csv
# run_v1.sh 의 REUSE_POOL 을 해당 경로로 업데이트한 뒤 실행하세요.

echo "[$(date '+%Y-%m-%d %H:%M:%S')] stage1-p1 제출"
argo submit stage1.yaml -p job=stage1-p1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 제출 완료 — argo watch 로 진행 상황을 확인하세요"
echo "  완료 후 run_v1.sh 의 REUSE_POOL 을 아래 경로로 업데이트하세요:"
echo "  /checkpoints/stage1-p1/agg/init/split.csv"
