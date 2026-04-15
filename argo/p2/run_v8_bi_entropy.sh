#!/bin/bash
set -euo pipefail

REUSE_POOL=/checkpoints/stage2-v6-fedpid-entropy-0/agg/init/split.csv
INTERVAL=$((2 * 60 * 60))  # 2시간

run() {
    local desc="$1"; shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 시작: $desc"
    argo submit "$@"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 제출 완료: $desc — ${INTERVAL}초 대기"
    sleep "$INTERVAL"
}

# r0~r4: anti-entropy pool(k/2), r5~r19: entropy pool(k/2)
run "fedpod-bi-entropy" stage2-bi-entropy.yaml \
    -p job=stage2-p2-v8-fedpod-bi-entropy-1 -p selection=bi_entropy \
    -p epochs=3 -p algorithm=fedpod -p sampling-rate=0.2 \
    -p kp=0.45 -p ki=0.1 -p kd=0.45 \
    -p committee-job=stage1-v2 \
    -p lr-scheduler=linear \
    -p reuse-pool="$REUSE_POOL"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 전체 완료"
