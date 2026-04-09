# [2026-04-08] Stage1 → Stage2 TensorBoard 기록 검증

**목적**: stage1에서 3 rounds × 4 epochs × 9개 기관 dynamic random 샘플링으로 지역모델 학습 후,
stage2 dry-run에서 stage1 최종 지역모델을 committee로 삼아 entropy selection + 3 rounds FL 학습.
각 라운드 완료 시 `runs/{job}/inst{PP}/` 에 `trn_loss` / `val_loss` 가 TensorBoard에 기록되는지 검증.

**stage1 핵심**: FedAvg 없이 각 기관이 지역모델만 독립 학습. `agg.py`는 다음 라운드 split CSV
생성 목적으로만 실행하며, 생성된 `agg.pt`는 stage1 학습에 사용하지 않음.

**stage2 핵심**: FL 규칙 — 매 라운드 직전 집계 모델 `agg/R03r{rr}/agg.pt` 를 `--init-ckpt` 로 사용.

```
runs/
  stage1/inst01/ … inst28/   ← round 0~2 trn_loss / val_loss
  stage2/inst01/ … inst28/   ← round 0~2 trn_loss / val_loss
```

---

### Step 1 — Stage1 dry-run (split CSV 초기화)

```bash
python3 scripts/agg.py --dry-run -J stage1 --ckpt-root checkpoints \
  -c experiments/fets/partition2/fets_split.csv \
  --sampling-rate 1.0 --sampling-mode dynamic --selection random
```

출력: `checkpoints/stage1/agg/init/agg.pt`, `checkpoints/stage1/agg/init/split.csv`

---

### Step 2 — Stage1 Round 0 학습 (9개 기관, GPU 분리)

random init에서 시작 (`--init-ckpt` 없음). GPU 0 / GPU 1 로 분리하여 병렬 실행.

```bash
# GPU 0
for P in 1 3 10 25 28; do
  CUDA_VISIBLE_DEVICES=0 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage1/agg/init/split.csv \
    -P $P -J stage1 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 0 --epochs 4 --epoch 0
done
```

```bash
# GPU 1
for P in 2 22 24 26; do
  CUDA_VISIBLE_DEVICES=1 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage1/agg/init/split.csv \
    -P $P -J stage1 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 0 --epochs 4 --epoch 0
done
```

출력: `checkpoints/stage1/inst{PP}/R03r00/best.pt` (9개 기관)

**TensorBoard 확인 포인트**
- `runs/stage1/inst{PP}/` 에 이벤트 파일 생성 확인
- `tensorboard --logdir runs/stage1` 실행 후 round=0 데이터 포인트 확인

---

### Step 3 — Stage1 Round 0 집계 (split CSV 갱신 목적)

```bash
python3 scripts/agg.py -J stage1 --ckpt-root checkpoints \
  -c checkpoints/stage1/agg/init/split.csv \
  --sampling-rate 1.0 --sampling-mode dynamic --selection random \
  --rounds 3 --round 0 --epochs 4 --partitions 1,2,3,10,22,24,25,26,28
```

출력: `checkpoints/stage1/agg/R03r00/split.csv` (R01 컬럼 추가) — `agg.pt`는 미사용

---

### Step 4 — Stage1 Round 1 학습

각 기관이 자신의 이전 라운드 지역모델에서 이어서 학습.

```bash
# GPU 0
for P in 1 3 10 25 28; do
  INST=$(printf "inst%02d" $P)
  CUDA_VISIBLE_DEVICES=0 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage1/agg/R03r00/split.csv \
    -P $P -J stage1 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 1 --epochs 4 --epoch 4 \
    --init-ckpt checkpoints/stage1/$INST/R03r00/best.pt
done
```

```bash
# GPU 1
for P in 2 22 24 26; do
  INST=$(printf "inst%02d" $P)
  CUDA_VISIBLE_DEVICES=1 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage1/agg/R03r00/split.csv \
    -P $P -J stage1 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 1 --epochs 4 --epoch 4 \
    --init-ckpt checkpoints/stage1/$INST/R03r00/best.pt
done
```

출력: `checkpoints/stage1/inst{PP}/R03r01/best.pt`

---

### Step 5 — Stage1 Round 1 집계

```bash
python3 scripts/agg.py -J stage1 --ckpt-root checkpoints \
  -c checkpoints/stage1/agg/R03r00/split.csv \
  --sampling-rate 1.0 --sampling-mode dynamic --selection random \
  --rounds 3 --round 1 --epochs 4 --partitions 1,2,3,10,22,24,25,26,28
```

출력: `checkpoints/stage1/agg/R03r01/split.csv` (R02 컬럼 추가)

---

### Step 6 — Stage1 Round 2 학습

```bash
# GPU 0
for P in 1 3 10 25 28; do
  INST=$(printf "inst%02d" $P)
  CUDA_VISIBLE_DEVICES=0 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage1/agg/R03r01/split.csv \
    -P $P -J stage1 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 2 --epochs 4 --epoch 8 \
    --init-ckpt checkpoints/stage1/$INST/R03r01/best.pt
done
```

```bash
# GPU 1
for P in 2 22 24 26; do
  INST=$(printf "inst%02d" $P)
  CUDA_VISIBLE_DEVICES=1 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage1/agg/R03r01/split.csv \
    -P $P -J stage1 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 2 --epochs 4 --epoch 8 \
    --init-ckpt checkpoints/stage1/$INST/R03r01/best.pt
done
```

출력: `checkpoints/stage1/inst{PP}/R03r02/best.pt`

---

### Step 7 — Stage1 Round 2 집계

```bash
python3 scripts/agg.py -J stage1 --ckpt-root checkpoints \
  -c checkpoints/stage1/agg/R03r01/split.csv \
  --sampling-rate 1.0 --sampling-mode dynamic --selection random \
  --rounds 3 --round 2 --epochs 4 --partitions 1,2,3,10,22,24,25,26,28
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
  --committee-partitions 1,2,3,10,22,24,25,26,28 \
  --committee-in-ch 4 --committee-out-classes 1 \
  --committee-chan [t1,t1ce,t2,flair] \
  --in-ch 5 --out-classes 3 \
  -D data/fets128/trainval --gpu 1
```

출력: `checkpoints/stage2/agg/init/agg.pt` (5ch/3class), `checkpoints/stage2/agg/init/split.csv` (R00 = entropy pool)

**확인 포인트**
- `split.csv` R00 컬럼에 pool=1/0 분포 확인
- 로그에 `BALD scoring` 진행률 출력 확인

---

### Step 9 — Stage2 Round 0 학습

FL 규칙: `agg/init/agg.pt` 를 `--init-ckpt` 로 사용.

```bash
# GPU 0
for P in 1 3 10; do
  CUDA_VISIBLE_DEVICES=0 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage2/agg/init/split.csv \
    -P $P -J stage2 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 0 --epochs 3 --epoch 0 \
    --init-ckpt checkpoints/stage2/agg/init/agg.pt \
    -C [t1,t1ce,t2,flair,seg] -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
done
```

```bash
# GPU 1
for P in 2 22; do
  CUDA_VISIBLE_DEVICES=1 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage2/agg/init/split.csv \
    -P $P -J stage2 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 0 --epochs 3 --epoch 0 \
    --init-ckpt checkpoints/stage2/agg/init/agg.pt \
    -C [t1,t1ce,t2,flair,seg] -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
done
```

```bash
# GPU 2
for P in 25 28; do
  CUDA_VISIBLE_DEVICES=2 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage2/agg/init/split.csv \
    -P $P -J stage2 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 0 --epochs 3 --epoch 0 \
    --init-ckpt checkpoints/stage2/agg/init/agg.pt \
    -C [t1,t1ce,t2,flair,seg] -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
done
```

```bash
# GPU 3
for P in 24 26; do
  CUDA_VISIBLE_DEVICES=3 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage2/agg/init/split.csv \
    -P $P -J stage2 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 0 --epochs 3 --epoch 0 \
    --init-ckpt checkpoints/stage2/agg/init/agg.pt \
    -C [t1,t1ce,t2,flair,seg] -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
done
```
출력: `checkpoints/stage2/inst{PP}/R03r00/best.pt`

---

### Step 10 — Stage2 Round 0 집계

```bash
python3 scripts/agg.py -J stage2 --ckpt-root checkpoints \
  -c checkpoints/stage2/agg/init/split.csv \
  --sampling-rate 1.0 --sampling-mode dynamic \
  --rounds 3 --round 0 --epochs 3 --partitions 1,2,3,10,22,24,25,26,28
```

출력: `checkpoints/stage2/agg/R03r00/agg.pt`, `checkpoints/stage2/agg/R03r00/split.csv` (R01 컬럼 추가)

---

### Step 11 — Stage2 Round 1 학습

FL 규칙: 직전 집계 모델 `agg/R03r00/agg.pt` 사용.

```bash
# GPU 0
for P in 1 3 10; do
  CUDA_VISIBLE_DEVICES=0 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage2/agg/R03r00/split.csv \
    -P $P -J stage2 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 1 --epochs 3 --epoch 3 \
    --init-ckpt checkpoints/stage2/agg/R03r00/agg.pt \
    -C [t1,t1ce,t2,flair,seg] -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
done
```

```bash
# GPU 1
for P in 2 22; do
  CUDA_VISIBLE_DEVICES=1 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage2/agg/R03r00/split.csv \
    -P $P -J stage2 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 1 --epochs 3 --epoch 3 \
    --init-ckpt checkpoints/stage2/agg/R03r00/agg.pt \
    -C [t1,t1ce,t2,flair,seg] -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
done
```

```bash
# GPU 2
for P in 25 28; do
  CUDA_VISIBLE_DEVICES=2 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage2/agg/R03r00/split.csv \
    -P $P -J stage2 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 1 --epochs 3 --epoch 3 \
    --init-ckpt checkpoints/stage2/agg/R03r00/agg.pt \
    -C [t1,t1ce,t2,flair,seg] -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
done
```

```bash
# GPU 3
for P in 24 26; do
  CUDA_VISIBLE_DEVICES=3 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage2/agg/R03r00/split.csv \
    -P $P -J stage2 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 1 --epochs 3 --epoch 3 \
    --init-ckpt checkpoints/stage2/agg/R03r00/agg.pt \
    -C [t1,t1ce,t2,flair,seg] -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
done
```

출력: `checkpoints/stage2/inst{PP}/R03r01/best.pt`

---

### Step 12 — Stage2 Round 1 집계

```bash
python3 scripts/agg.py -J stage2 --ckpt-root checkpoints \
  -c checkpoints/stage2/agg/R03r00/split.csv \
  --sampling-rate 1.0 --sampling-mode dynamic \
  --rounds 3 --round 1 --epochs 3 --partitions 1,2,3,10,22,24,25,26,28
```

출력: `checkpoints/stage2/agg/R03r01/agg.pt`, `checkpoints/stage2/agg/R03r01/split.csv` (R02 컬럼 추가)

---

### Step 13 — Stage2 Round 2 학습

FL 규칙: 직전 집계 모델 `agg/R03r01/agg.pt` 사용.

```bash
# GPU 0
for P in 1 3 10; do
  CUDA_VISIBLE_DEVICES=0 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage2/agg/R03r01/split.csv \
    -P $P -J stage2 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 2 --epochs 3 --epoch 6 \
    --init-ckpt checkpoints/stage2/agg/R03r01/agg.pt \
    -C [t1,t1ce,t2,flair,seg] -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
done
```

```bash
# GPU 1
for P in 2 22; do
  CUDA_VISIBLE_DEVICES=1 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage2/agg/R03r01/split.csv \
    -P $P -J stage2 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 2 --epochs 3 --epoch 6 \
    --init-ckpt checkpoints/stage2/agg/R03r01/agg.pt \
    -C [t1,t1ce,t2,flair,seg] -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
done
```

```bash
# GPU 2
for P in 25 28; do
  CUDA_VISIBLE_DEVICES=2 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage2/agg/R03r01/split.csv \
    -P $P -J stage2 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 2 --epochs 3 --epoch 6 \
    --init-ckpt checkpoints/stage2/agg/R03r01/agg.pt \
    -C [t1,t1ce,t2,flair,seg] -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
done
```

```bash
# GPU 3
for P in 24 26; do
  CUDA_VISIBLE_DEVICES=3 python3 scripts/app.py -D data/fets128/trainval \
    -c checkpoints/stage2/agg/R03r01/split.csv \
    -P $P -J stage2 --ckpt-root checkpoints --runs-root runs \
    --rounds 3 --round 2 --epochs 3 --epoch 6 \
    --init-ckpt checkpoints/stage2/agg/R03r01/agg.pt \
    -C [t1,t1ce,t2,flair,seg] -G [[1,2,4],[1,4],[4]] -N [wt,tc,et] -I [1,2,3]
done
```

출력: `checkpoints/stage2/inst{PP}/R03r02/best.pt`

---

### Step 14 — Stage2 Round 2 집계

```bash
python3 scripts/agg.py -J stage2 --ckpt-root checkpoints \
  -c checkpoints/stage2/agg/R03r01/split.csv \
  --sampling-rate 1.0 --sampling-mode dynamic \
  --rounds 3 --round 2 --epochs 3 --partitions 1,2,3,10,22,24,25,26,28
```

출력: `checkpoints/stage2/agg/R03r02/agg.pt`, `checkpoints/stage2/agg/R03r02/split.csv` (R03 컬럼 추가)

---

### Step 15 — TensorBoard 검증

```bash
tensorboard --logdir runs
```

**확인 포인트**

| 항목 | 기대값 |
|------|--------|
| `runs/stage1/inst{PP}` 이벤트 파일 존재 | 9개 기관 × round 0~2 |
| `runs/stage2/inst{PP}` 이벤트 파일 존재 | 9개 기관 × round 0~2 |
| `trn_loss` curve | round 0→1→2 로 3개 데이터 포인트, 기관별 색상 구분 |
| `val_loss` curve | 동일 |
| stage1 vs stage2 비교 | `--logdir runs/stage1:stage1,runs/stage2:stage2` 로 오버레이 가능 |

```bash
# stage1 / stage2 비교 오버레이
tensorboard --logdir_spec stage1:runs/stage1,stage2:runs/stage2
```

**resume 후 기록 없음 케이스 확인**
- 이미 완료된 라운드를 재실행하면 epoch 루프가 비어 TensorBoard 기록이 생략되는지 확인
- 로그에 `TensorBoard —` 줄이 없으면 정상
