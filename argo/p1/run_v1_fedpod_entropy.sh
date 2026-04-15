#!/bin/bash
set -euo pipefail

REUSE_POOL=/checkpoints/stage2-p1-v1-pool-anti-entropy-0/agg/init/split.csv
INTERVAL=$((2 * 60 * 60))  # 2시간

run() {
    local desc="$1"; shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 시작: $desc"
    argo submit "$@"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 제출 완료: $desc — ${INTERVAL}초 대기"
    sleep "$INTERVAL"
}

# warm-up(r0-r4): inst_01 단독 학습 → FL(r5-r9): 전체 23 기관
run "fedpod-primary-entropy" stage2-primary-entropy.yaml \
    -p job=stage2-p1-v1-fedpod-primary-entropy-0 -p selection=entropy \
    -p epochs=3 -p algorithm=fedpod -p sampling-rate=0.2 \
    -p kp=0.45 -p ki=0.1 -p kd=0.45 \
    -p primary-partition=1 \
    -p committee-job=stage1-p1 \
    -p lr-scheduler=linear \
    -p reuse-pool="$REUSE_POOL"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 전체 완료"
