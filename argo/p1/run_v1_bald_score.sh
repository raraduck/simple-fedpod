#!/bin/bash
set -euo pipefail

# BALD score dry-run: partition1 committee 로 전체 train subject 점수 계산
# 출력: /checkpoints/stage2-p1-bald-score/agg/init/bald_scores.csv

argo submit stage2-bald-score.yaml \
    -p job=stage2-p1-bald-score \
    -p committee-job=stage1-p1 \
    -p committee-rounds=5 \
    -p committee-round=4 \
    -p committee-partitions="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23" \
    -p sampling-rate=1.0 \
    --watch

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 완료 — /checkpoints/stage2-p1-bald-score/agg/init/bald_scores.csv"
