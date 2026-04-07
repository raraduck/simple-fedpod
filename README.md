# simple-fedpod

FL(Federated Learning) 클라이언트로 동작하는 컨테이너 기반 학습 pod입니다.

## 코드 구조

```
scripts/
  app.py          # 진입점 — args 파싱, App.run() 흐름
  agg.py          # 집계 — FedAvg, dry-run(모델/split 초기화), split CSV 업데이트
  dataset.py      # FeTSDataset, load_split
  trainer.py      # Trainer — BCE+SoftDice loss, Adam, 체크포인트 저장/로드
  models/
    unet3d.py     # 3D Residual U-Net
    loss.py       # SoftDiceBCEWithLogitsLoss
argo/
  Containerfile   # Argo 이미지 — scripts/ 는 NFS 마운트 (COPY 없음)
  workflow.yaml   # Argo Workflow 정의
  stage1.yaml     # 병렬 학습 DAG (withItems)
katib/
  Containerfile   # Katib 이미지 — scripts/ 를 이미지에 COPY
  experiment.yaml # HPO Experiment — lr, batch
analysis/
  split_analysis.ipynb  # split CSV 시각화 (R별 기관 선택 수 plot)
```

## FL 라운드 흐름

```
agg.py --dry-run          초기 모델 생성 + R00 split CSV 준비
        ↓
app.py --round 0          각 inst 병렬 학습 (init 모델 로드, R00 subjects)
        ↓
agg.py --round 0          FedAvg 집계 + R01 split CSV 업데이트
        ↓
app.py --round 1          각 inst 병렬 학습 (agg 모델 로드, R01 subjects)
        ↓
      ...
```

### 체크포인트 경로 구조

```
checkpoints/{job}/
  inst{PP}/
    R{RR}r{NN}/       ← app.py 저장 (PP=partition, RR=rounds, NN=round)
      latest.pt
      best.pt
  agg/
    init/             ← dry-run 결과
      agg.pt
      split.csv
    R{RR}r{NN}/       ← 라운드별 집계 결과
      agg.pt
      split.csv
```

예: `-J stage1 -P 1 --rounds 1 --round 0` → `checkpoints/stage1/inst01/R01r00/`

## Split CSV

원본 `fets_split.csv`에서 `Partition_ID`, `Subject_ID`, `TrainOrVal` 3컬럼만 유지하고, 라운드마다 `R{DD}` 컬럼을 추가합니다.

| Partition_ID | Subject_ID | TrainOrVal | R00 | R01 | ... |
|---|---|---|---|---|---|
| 1 | subj001 | train | 1 | 1 | |
| 1 | subj002 | val | | | |
| 2 | subj003 | train | 0 | 1 | |

- train 선택: `1`, 미선택: `0`, val: 빈칸
- `load_split(split_csv, partition_id, round_idx)` — `R{round_idx:02d}` 컬럼으로 train 필터링

### Poisson 샘플링

각 라운드에서 학습할 train subjects를 **기관 간 균등 λ**로 Poisson 샘플링합니다.

$$k \sim \text{Poisson}(\lambda), \quad \lambda = \bar{N} \times r$$

- $\bar{N}$: 전체 기관의 평균 train subject 수
- $r$: `--sampling-rate` (0.0~1.0)
- $k$: 실제 선택 수 (1 이상, 기관 보유 수 이하로 clip)

| 모드 | 동작 |
|------|------|
| `static` | dry-run 시 1회 Poisson draw → 이후 라운드 고정 |
| `dynamic` | 매 라운드 Poisson draw → 라운드마다 선택 변동 |

## 실행 명령

### 1. dry-run (초기 모델 + split CSV 생성)

```bash
python3 scripts/agg.py --dry-run -J stage1 --ckpt-root ./checkpoints \
  -c ./experiments/fets/partition2/fets_split.csv \
  --sampling-rate 1.0 --sampling-mode static
```

### 2. round 0 학습

```bash
python3 scripts/app.py -D ./data/fets128/trainval \
  -c ./checkpoints/stage1/agg/init/split.csv \
  -P 1 -J stage1 --ckpt-root ./checkpoints \
  --rounds 1 --round 0 -E 3 \
  --init-ckpt ./checkpoints/stage1/agg/init/agg.pt
```

### 3. round 0 집계

```bash
python3 scripts/agg.py -J stage1 --ckpt-root ./checkpoints \
  -c ./checkpoints/stage1/agg/init/split.csv \
  --rounds 1 --round 0 --epochs 3 --num-partitions 2
```

### 4. round 1 학습

```bash
python3 scripts/app.py -D ./data/fets128/trainval \
  -c ./checkpoints/stage1/agg/R01r00/split.csv \
  -P 1 -J stage1 --ckpt-root ./checkpoints \
  --rounds 1 --round 1 -E 3 \
  --init-ckpt ./checkpoints/stage1/agg/R01r00/agg.pt
```

## app.py Args

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `-D` | `/data/fets128/trainval` | 데이터 경로 |
| `-c` | `/experiments/fets/partition2/fets_split.csv` | split CSV |
| `-P` | `1` | Partition ID |
| `-J` | `test_run` | job 이름 |
| `--ckpt-root` | `/checkpoints` | 체크포인트 루트 |
| `--init-ckpt` | `` | 초기 모델 경로 (agg.pt) |
| `--rounds` | `1` | 총 FL 라운드 수 |
| `--round` | `0` | 현재 라운드 (0-indexed) |
| `-E` | `30` | 라운드당 에폭 수 |
| `--epoch` | `0` | 에폭 오프셋 (resume) |
| `--batch` | `2` | 배치 크기 |
| `--lr` | `5e-3` | 학습률 |
| `--block` | `residual` | 블록 타입 (residual / plain) |
| `--channels` | `[32,64,128,256]` | encoder 채널 수 |
| `--norm` | `instance` | 정규화 (instance / batch) |

## agg.py Args

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `-J` | `stage1` | job 이름 |
| `--ckpt-root` | `/checkpoints` | 체크포인트 루트 |
| `-c` | `/experiments/.../fets_split.csv` | split CSV |
| `--rounds` | `1` | 총 FL 라운드 수 |
| `--round` | `0` | 현재 라운드 |
| `--epochs` | `30` | 라운드당 에폭 수 |
| `--num-partitions` | `2` | 집계할 partition 수 |
| `--algorithm` | `fedavg` | 집계 알고리즘 |
| `--dry-run` | `False` | 초기 모델/split 생성만 수행 |
| `--sampling-rate` | `1.0` | Poisson 샘플링 비율 |
| `--sampling-mode` | `static` | static / dynamic |

## 이미지

| 이미지 | Containerfile | 코드 포함 방식 | 용도 |
|--------|--------------|----------------|------|
| `argo-fedpod:v0.2` | `argo/Containerfile` | NFS 마운트 (`/app`) | Argo Workflow 학습 |
| `simple-fedpod:katib` | `katib/Containerfile` | 이미지에 COPY | Katib HPO trial |

```bash
# Argo 이미지 빌드
podman build -f argo/Containerfile -t argo-fedpod:v0.2 .

# Katib 이미지 빌드 (프로젝트 루트에서 실행)
podman build -f katib/Containerfile -t simple-fedpod:katib .

# 로컬 실행 (Podman GPU)
sudo podman run \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  -v /lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:ro \
  -e LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  --shm-size=8g \
  -v ./scripts:/app:z -v ./data:/data:z \
  -v ./experiments:/experiments:z -v ./checkpoints:/checkpoints:z \
  192.168.0.80:30002/dwnkim/argo-fedpod:v0.2
```
