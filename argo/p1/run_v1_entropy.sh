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

run "fedwavg-entropy" stage2-entropy.yaml \
    -p job=stage2-p1-v1-fedwavg-entropy-0 -p selection=entropy \
    -p epochs=3 -p algorithm=fedwavg -p sampling-rate=0.2 \
    -p committee-job=stage1-p1 \
    -p lr-scheduler=linear \
    -p reuse-pool="$REUSE_POOL"

run "fedpod-entropy" stage2-entropy.yaml \
    -p job=stage2-p1-v1-fedpod-entropy-0 -p selection=entropy \
    -p epochs=3 -p algorithm=fedpod -p sampling-rate=0.2 \
    -p kp=0.45 -p ki=0.1 -p kd=0.45 \
    -p committee-job=stage1-p1 \
    -p lr-scheduler=linear \
    -p reuse-pool="$REUSE_POOL"

run "fedpid-entropy" stage2-entropy.yaml \
    -p job=stage2-p1-v1-fedpid-entropy-0 -p selection=entropy \
    -p epochs=3 -p algorithm=fedpid -p sampling-rate=0.2 \
    -p kp=0.45 -p ki=0.1 -p kd=0.45 \
    -p committee-job=stage1-p1 \
    -p lr-scheduler=linear \
    -p reuse-pool="$REUSE_POOL"

run "fedbn-entropy" stage2-entropy.yaml \
    -p job=stage2-p1-v1-fedbn-entropy-0 -p selection=entropy \
    -p epochs=3 -p algorithm=fedbn -p sampling-rate=0.2 \
    -p committee-job=stage1-p1 \
    -p lr-scheduler=linear \
    -p reuse-pool="$REUSE_POOL"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 전체 완료"
