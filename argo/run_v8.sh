#!/bin/bash
set -euo pipefail

REUSE_POOL=/checkpoints/stage2-v6-fedpid-entropy-0/agg/init/split.csv
INTERVAL=$((2 * 60 * 60))  # 3시간

run() {
    local desc="$1"; shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 시작: $desc"
    argo submit "$@"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 제출 완료: $desc — ${INTERVAL}초 대기"
    sleep "$INTERVAL"
}

run "fedwavg-random" stage2-random.yaml \
    -p job=stage2-v8-fedwavg-random-0 -p selection=random \
    -p epochs=3 -p algorithm=fedwavg -p sampling-rate=0.2 \
    -p lr-scheduler=cosine \
    -p reuse-pool="$REUSE_POOL"

run "fedpod-random" stage2-random.yaml \
    -p job=stage2-v8-fedpod-random-0 -p selection=random \
    -p epochs=3 -p algorithm=fedpod -p sampling-rate=0.2 \
    -p kp=0.45 -p ki=0.1 -p kd=0.45 \
    -p lr-scheduler=cosine \
    -p reuse-pool="$REUSE_POOL"

run "fedpid-random" stage2-random.yaml \
    -p job=stage2-v8-fedpid-random-0 -p selection=random \
    -p epochs=3 -p algorithm=fedpid -p sampling-rate=0.2 \
    -p kp=0.45 -p ki=0.1 -p kd=0.45 \
    -p lr-scheduler=cosine \
    -p reuse-pool="$REUSE_POOL"

run "fedwavg-anti-entropy" stage2-entropy.yaml \
    -p job=stage2-v8-fedwavg-anti-entropy-0 -p selection=anti_entropy \
    -p epochs=3 -p algorithm=fedwavg -p sampling-rate=0.2 \
    -p committee-job=stage1-v2 \
    -p lr-scheduler=cosine \
    -p reuse-pool="$REUSE_POOL"

run "fedpod-anti-entropy" stage2-entropy.yaml \
    -p job=stage2-v8-fedpod-anti-entropy-0 -p selection=anti_entropy \
    -p epochs=3 -p algorithm=fedpod -p sampling-rate=0.2 \
    -p kp=0.45 -p ki=0.1 -p kd=0.45 \
    -p committee-job=stage1-v2 \
    -p lr-scheduler=cosine \
    -p reuse-pool="$REUSE_POOL"

run "fedpid-anti-entropy" stage2-entropy.yaml \
    -p job=stage2-v8-fedpid-anti-entropy-0 -p selection=anti_entropy \
    -p epochs=3 -p algorithm=fedpid -p sampling-rate=0.2 \
    -p kp=0.45 -p ki=0.1 -p kd=0.45 \
    -p committee-job=stage1-v2 \
    -p lr-scheduler=cosine \
    -p reuse-pool="$REUSE_POOL"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 전체 완료"
