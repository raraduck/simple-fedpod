# [2026-04-09] Stage2 Entropy vs. Random — Argo Workflow 비교 실험

**목적**: `stage2.yaml`의 `selection` 파라미터를 달리하여 두 실험을 동시에 제출하고,
pool+entropy vs. pool+random dry-run이 FL 학습 결과에 미치는 영향을 TensorBoard로 비교.

**전제 조건**: stage1이 `job=stage1`, rounds=5, epochs=4, partitions=1,2,3,10,22,24,25,26,28 로
완료되어 `checkpoints/stage1/inst{PP}/R05r04/best.pt` 가 존재해야 함.

**실험 구성**

| 실험 | job 이름 | dry-run selection | round agg sampling |
|------|----------|-------------------|--------------------|
| A | `stage2-entropy` | entropy (BALD) | dynamic |
| B | `stage2-random`  | random          | dynamic |

---

### Step 1 — 두 실험 동시 제출

```bash
argo submit argo/stage2.yaml \
  -p job=stage2-entropy \
  -p selection=entropy

argo submit argo/stage2.yaml \
  -p job=stage2-random \
  -p selection=random
```

각 워크플로우는 독립적으로 실행되며 서로 다른 체크포인트 경로를 사용한다:
- `checkpoints/stage2-entropy/`
- `checkpoints/stage2-random/`

기본값으로 사용되는 파라미터 (변경 불필요):
- `committee-job=stage1`, `committee-rounds=5`, `committee-round=4`
- `committee-partitions=1,2,3,10,22,24,25,26,28`
- `agg-sampling-mode=dynamic`

---

### Step 2 — 워크플로우 상태 확인

```bash
# 전체 목록
argo list -n dwnkim

# 개별 상세
argo get -n dwnkim <workflow-name>

# 실시간 로그 (dry-run 단계)
argo logs -n dwnkim <workflow-name> -f
```

DAG 실행 순서 (두 실험 동일):
```
dry-run
  └─ train-r0 (9개 기관 병렬)
       └─ agg-r0
            └─ train-r1 (9개 기관 병렬)
                 └─ agg-r1
                      └─ ...
                           └─ train-r4 → agg-r4
```

---

### Step 3 — 체크포인트 출력 구조

```
checkpoints/
  stage2-entropy/
    agg/init/agg.pt          ← dry-run: entropy pool 초기 모델
    agg/init/split.csv       ← pool=1 대상 subject (entropy top-k)
    agg/R05r00/agg.pt        ← round 0 FedAvg 결과
    agg/R05r00/split.csv
    ...
    agg/R05r04/agg.pt        ← 최종 집계 모델
    inst{PP}/R05r{rr}/best.pt
    inst{PP}/R05r{rr}/metrics.json

  stage2-random/
    agg/init/agg.pt          ← dry-run: random pool 초기 모델
    agg/init/split.csv       ← pool=1 대상 subject (random top-k)
    ...
```

---

### Step 4 — TensorBoard 비교

**runs 디렉토리 구조** (실험당):
```
runs/stage2-entropy/
  inst{PP}/    ← 기관별 학습 기록 (app.py)
  inst_avg/    ← 기관 평균 (agg.py, 라운드 집계 후)
  gen_eval/    ← 일반화 평가 (gen_eval.py, train과 병렬)
```

```bash
# 두 실험 오버레이 비교
tensorboard --logdir_spec \
  entropy:runs/stage2-entropy,\
  random:runs/stage2-random
```

**태그 구조**: `(ech|rnd)_(trn|val)_(loss|dice)_(avg|prv|pst|prvpst)/(avg|{name})`

**확인 포인트**

| 패널 | entropy | random | gen_eval 오버레이 |
|------|---------|--------|------------------|
| `rnd_val_loss_prvpst/avg` | prv→pst loss | prv→pst loss | prv 위치 (글로벌 모델) |
| `rnd_val_dice_prvpst/avg` | prv→pst dice | prv→pst dice | prv 위치 |
| `rnd_val_dice_pst/wt` | wt Dice | wt Dice | — |
| `rnd_val_dice_pst/tc` | tc Dice | tc Dice | — |
| `rnd_val_dice_pst/et` | et Dice | et Dice | — |
| `ech_val_loss_prvpst/avg` | epoch 축 loss | epoch 축 loss | epoch 축 prv |

```bash
# inst_avg + gen_eval 오버레이 (단일 실험 내 일반화 비교)
tensorboard --logdir runs/stage2-entropy
# → inst_avg 와 gen_eval 을 함께 선택하면 fine-tuning 전후 비교 가능
```

---

### Step 5 — 특정 파라미터 재지정 예시

```bash
# committee round를 다르게 지정할 경우
argo submit argo/stage2.yaml \
  -p job=stage2-entropy-r3 \
  -p selection=entropy \
  -p committee-round=3

# agg sampling mode를 static으로 변경
argo submit argo/stage2.yaml \
  -p job=stage2-entropy-static \
  -p selection=entropy \
  -p agg-sampling-mode=static
```
