# [2026-04-07] Stage1 → Stage2 entropy selection 동작 검증

**목적**: stage1에서 3 rounds × 5 epochs × dynamic random 샘플링으로 2개 기관 지역모델 학습 후,
stage2 dry-run에서 해당 지역모델을 committee로 삼아 entropy selection이 정상 동작하는지 확인.

**stage1 핵심**: FedAvg 없이 각 기관이 지역모델만 독립 학습. `agg.py`는 다음 라운드 split CSV
생성 목적으로만 실행하며, 생성된 `agg.pt`는 stage1 학습에 사용하지 않음.

---

### Step 1 — Stage1 dry-run (split CSV 초기화)

```bash
python3 scripts/agg.py --dry-run -J stage1 --ckpt-root checkpoints \
  -c experiments/fets/partition2/fets_split.csv \
  --sampling-rate 1.0 --sampling-mode dynamic --selection random
```

출력: `checkpoints/stage1/agg/init/agg.pt`, `checkpoints/stage1/agg/init/split.csv`

---

### Step 2 — Stage1 Round 0 학습 (P1, P2 병렬 가능)

random init에서 시작 (`--init-ckpt` 없음). `CUDA_VISIBLE_DEVICES`로 GPU 분리.

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/app.py -D data/fets128/trainval \
  -c checkpoints/stage1/agg/init/split.csv \
  -P 1 -J stage1 --ckpt-root checkpoints \
  --rounds 3 --round 0 -E 5 --epoch 0
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/app.py -D data/fets128/trainval \
  -c checkpoints/stage1/agg/init/split.csv \
  -P 2 -J stage1 --ckpt-root checkpoints \
  --rounds 3 --round 0 -E 5 --epoch 0
```

출력: `checkpoints/stage1/inst01/R03r00/best.pt`, `checkpoints/stage1/inst02/R03r00/best.pt`

---

### Step 3 — Stage1 Round 0 집계 (split CSV 갱신 목적)

```bash
python3 scripts/agg.py -J stage1 --ckpt-root checkpoints \
  -c checkpoints/stage1/agg/init/split.csv \
  --sampling-rate 1.0 --sampling-mode dynamic --selection random \
  --rounds 3 --round 0 --epochs 5 --partitions 1,2
```

출력: `checkpoints/stage1/agg/R03r00/split.csv` (R01 컬럼 추가) — `agg.pt`는 미사용

---

### Step 4 — Stage1 Round 1 학습

각 기관이 자신의 이전 라운드 지역모델에서 이어서 학습.

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/app.py -D data/fets128/trainval \
  -c checkpoints/stage1/agg/R03r00/split.csv \
  -P 1 -J stage1 --ckpt-root checkpoints \
  --rounds 3 --round 1 -E 5 --epoch 5 \
  --init-ckpt checkpoints/stage1/inst01/R03r00/best.pt
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/app.py -D data/fets128/trainval \
  -c checkpoints/stage1/agg/R03r00/split.csv \
  -P 2 -J stage1 --ckpt-root checkpoints \
  --rounds 3 --round 1 -E 5 --epoch 5 \
  --init-ckpt checkpoints/stage1/inst02/R03r00/best.pt
```

출력: `checkpoints/stage1/inst01/R03r01/best.pt`, `checkpoints/stage1/inst02/R03r01/best.pt`

---

### Step 5 — Stage1 Round 1 집계 (split CSV 갱신 목적)

```bash
python3 scripts/agg.py -J stage1 --ckpt-root checkpoints \
  -c checkpoints/stage1/agg/R03r00/split.csv \
  --sampling-rate 1.0 --sampling-mode dynamic --selection random \
  --rounds 3 --round 1 --epochs 5 --partitions 1,2
```

출력: `checkpoints/stage1/agg/R03r01/split.csv` (R02 컬럼 추가) — `agg.pt`는 미사용

---

### Step 6 — Stage1 Round 2 학습

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/app.py -D data/fets128/trainval \
  -c checkpoints/stage1/agg/R03r01/split.csv \
  -P 1 -J stage1 --ckpt-root checkpoints \
  --rounds 3 --round 2 -E 5 --epoch 10 \
  --init-ckpt checkpoints/stage1/inst01/R03r01/best.pt
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/app.py -D data/fets128/trainval \
  -c checkpoints/stage1/agg/R03r01/split.csv \
  -P 2 -J stage1 --ckpt-root checkpoints \
  --rounds 3 --round 2 -E 5 --epoch 10 \
  --init-ckpt checkpoints/stage1/inst02/R03r01/best.pt
```

출력: `checkpoints/stage1/inst01/R03r02/best.pt`, `checkpoints/stage1/inst02/R03r02/best.pt`

---

### Step 7 — Stage1 Round 2 집계 (split CSV 갱신 목적)

```bash
python3 scripts/agg.py -J stage1 --ckpt-root checkpoints \
  -c checkpoints/stage1/agg/R03r01/split.csv \
  --sampling-rate 1.0 --sampling-mode dynamic --selection random \
  --rounds 3 --round 2 --epochs 5 --partitions 1,2
```

출력: `checkpoints/stage1/agg/R03r02/split.csv` (R03 컬럼 추가) — `agg.pt`는 미사용

---

### Step 8 — Stage2 dry-run (entropy selection, committee = stage1 최종 지역모델)

stage2는 입력 채널에 `seg`가 추가되고 레이블이 wt/tc/et 3개로 확장된다.
committee(stage1 모델)는 4채널/1클래스 아키텍처이므로 `--committee-*` 인자로 별도 지정한다.

```bash
python3 scripts/agg.py --dry-run -J stage2 --ckpt-root checkpoints \
  -c experiments/fets/partition2/fets_split.csv \
  --sampling-rate 1.0 --sampling-mode pool --selection entropy \
  --committee-job stage1 --committee-rounds 3 --committee-round 2 \
  --committee-partitions 1,2 \
  --committee-in-ch 4 --committee-out-classes 1 \
  --committee-chan [t1,t1ce,t2,flair] \
  --in-ch 5 --out-classes 3 \
  -D data/fets128/trainval --gpu 1
```

출력: `checkpoints/stage2/agg/init/agg.pt` (5ch/3class), `checkpoints/stage2/agg/init/split.csv` (R00 = entropy top-k)

**확인 포인트**
- `split.csv`의 R00 컬럼에 선택된 subject(1)와 미선택(0)이 기록되었는지
- 로그에 `Entropy range — top / k-th / bottom` 값이 출력되는지
- 두 기관 간 선택 분포가 random과 다른지 비교

---

### Step 9 — Stage2 Round 0 학습 (P1, P2 병렬 가능)

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/app.py -D data/fets128/trainval \
  -c checkpoints/stage2/agg/init/split.csv \
  -P 1 -J stage2 --ckpt-root checkpoints \
  --rounds 3 --round 0 -E 3 --epoch 0 \
  --init-ckpt checkpoints/stage2/agg/init/agg.pt \
  -C [t1,t1ce,t2,flair,seg] \
  -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/app.py -D data/fets128/trainval \
  -c checkpoints/stage2/agg/init/split.csv \
  -P 2 -J stage2 --ckpt-root checkpoints \
  --rounds 3 --round 0 -E 3 --epoch 0 \
  --init-ckpt checkpoints/stage2/agg/init/agg.pt \
  -C [t1,t1ce,t2,flair,seg] \
  -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
```

출력: `checkpoints/stage2/inst01/R03r00/best.pt`, `checkpoints/stage2/inst02/R03r00/best.pt`

---

### Step 10 — Stage2 Round 0 집계

라운드에서는 `--selection random`만 허용. pool(Partition_ID notna) 내에서 sampling_rate로 random 샘플링해 R01 컬럼을 갱신한다.

```bash
python3 scripts/agg.py -J stage2 --ckpt-root checkpoints \
  -c checkpoints/stage2/agg/init/split.csv \
  --sampling-rate 1.0 --sampling-mode dynamic \
  --rounds 3 --round 0 --epochs 3 --partitions 1,2
```

출력: `checkpoints/stage2/agg/R03r00/agg.pt`, `checkpoints/stage2/agg/R03r00/split.csv` (R01 컬럼 추가)

**확인 포인트**
- R01 컬럼이 R00 pool(Partition_ID notna) 내에서만 선택되었는지
- pool 밖 subjects(Partition_ID=NA)가 R01에서 0인지 확인

---

### Step 11 — Stage2 Round 1 학습 (P1, P2 병렬 가능)

각 기관이 직전 라운드 집계 모델(`agg.pt`)에서 시작 — FL 규칙.

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/app.py -D data/fets128/trainval \
  -c checkpoints/stage2/agg/R03r00/split.csv \
  -P 1 -J stage2 --ckpt-root checkpoints \
  --rounds 3 --round 1 -E 3 --epoch 3 \
  --init-ckpt checkpoints/stage2/agg/R03r00/agg.pt \
  -C [t1,t1ce,t2,flair,seg] \
  -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/app.py -D data/fets128/trainval \
  -c checkpoints/stage2/agg/R03r00/split.csv \
  -P 2 -J stage2 --ckpt-root checkpoints \
  --rounds 3 --round 1 -E 3 --epoch 3 \
  --init-ckpt checkpoints/stage2/agg/R03r00/agg.pt \
  -C [t1,t1ce,t2,flair,seg] \
  -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
```

출력: `checkpoints/stage2/inst01/R03r01/best.pt`, `checkpoints/stage2/inst02/R03r01/best.pt`

---

### Step 12 — Stage2 Round 1 집계

```bash
python3 scripts/agg.py -J stage2 --ckpt-root checkpoints \
  -c checkpoints/stage2/agg/R03r00/split.csv \
  --sampling-rate 1.0 --sampling-mode dynamic \
  --rounds 3 --round 1 --epochs 3 --partitions 1,2
```

출력: `checkpoints/stage2/agg/R03r01/agg.pt`, `checkpoints/stage2/agg/R03r01/split.csv` (R02 컬럼 추가)

---

### Step 13 — Stage2 Round 2 학습 (P1, P2 병렬 가능)

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/app.py -D data/fets128/trainval \
  -c checkpoints/stage2/agg/R03r01/split.csv \
  -P 1 -J stage2 --ckpt-root checkpoints \
  --rounds 3 --round 2 -E 3 --epoch 6 \
  --init-ckpt checkpoints/stage2/agg/R03r01/agg.pt \
  -C [t1,t1ce,t2,flair,seg] \
  -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/app.py -D data/fets128/trainval \
  -c checkpoints/stage2/agg/R03r01/split.csv \
  -P 2 -J stage2 --ckpt-root checkpoints \
  --rounds 3 --round 2 -E 3 --epoch 6 \
  --init-ckpt checkpoints/stage2/agg/R03r01/agg.pt \
  -C [t1,t1ce,t2,flair,seg] \
  -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
```

출력: `checkpoints/stage2/inst01/R03r02/best.pt`, `checkpoints/stage2/inst02/R03r02/best.pt`

---

### Step 14 — Stage2 Round 2 집계

```bash
python3 scripts/agg.py -J stage2 --ckpt-root checkpoints \
  -c checkpoints/stage2/agg/R03r01/split.csv \
  --sampling-rate 1.0 --sampling-mode dynamic \
  --rounds 3 --round 2 --epochs 3 --partitions 1,2
```

출력: `checkpoints/stage2/agg/R03r02/agg.pt`, `checkpoints/stage2/agg/R03r02/split.csv` (R03 컬럼 추가)
