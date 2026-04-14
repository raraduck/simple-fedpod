#!/bin/bash
set -euo pipefail

# stage1-p2: partition2 (33개 기관) committee 모델 생성
# p2 는 이미 stage2-v6-fedpid-entropy-0 의 split.csv 를 reuse-pool 로 사용 중.
# 새로운 committee 모델이 필요한 경우에만 실행하세요.

echo "[$(date '+%Y-%m-%d %H:%M:%S')] stage1-p2 제출"
argo submit stage1.yaml -p job=stage1-p2

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 제출 완료 — argo watch 로 진행 상황을 확인하세요"
