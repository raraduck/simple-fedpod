# simple-fedpod

FL(Federated Learning) 클라이언트로 동작하는 컨테이너 기반 학습 pod입니다.

## 코드 구조

```
scripts/
  app.py          # 진입점 — args 파싱, App.run() 흐름
  dataset.py      # FeTSDataset, load_split
  trainer.py      # Trainer — Dice loss, Adam, 체크포인트 저장/로드
  models/
    unet3d.py     # 3D Residual U-Net
builds/
  Containerfile   # 컨테이너 정의
```

### `App` 클래스 (`scripts/app.py`)

`main()`에서 args를 파싱한 뒤 `App(args).run()`을 호출합니다.

```
main()
 ├── argparse로 설정 파싱
 └── App(args).run()
      ├── load_split()       — CSV에서 Partition_ID로 train/val subject 목록 분리
      ├── FeTSDataset()      — subject 목록으로 Dataset 구성 (train / val)
      ├── DataLoader()       — 배치 구성 (train: shuffle, val: fixed)
      ├── UNet()             — args로 모델 초기화, GPU 배치
      └── Trainer.run()      — epoch 루프, 체크포인트 저장/로드
```

### `load_split(split_csv, partition_id)`

CSV에서 `Partition_ID`로 해당 클라이언트의 subject를 필터링해 train/val 리스트를 반환합니다.

### `FeTSDataset`

| 항목 | 내용 |
|------|------|
| 입력 (`-C`) | `{subject}_{ch}.nii.gz` — t1, t1ce, t2, flair, seg 등 조합 가능 |
| 레이블 소스 | `{subject}_sub.nii.gz` — subregion 값 포함 |
| 레이블 생성 | `lgrp`의 값들을 OR 조합해 binary mask 생성, `lnam`/`lidx`로 이름/인덱스 부여 |
| `__getitem__` 출력 | `image (C, H, W, D)`, `label (L, H, W, D)` float32 텐서 |

예: `lgrp=[[1,2,4]], lnam=[wt]` → sub에서 값 1,2,4를 합쳐 wt(whole tumor) binary mask 생성

### Args

| 플래그 | 이름 | 기본값 | 설명 |
|--------|------|--------|------|
| `-D` | `--data`     | `/data/fets128/trainval` | 데이터 경로 |
| `-d` | `--dset`     | `fets` | 데이터셋 종류 |
| `-c` | `--split`    | `/experiments/fets/partition2/fets_split.csv` | train/val 분할 CSV |
| `-C` | `--chan`     | `[t1,t1ce,t2,flair]` | 입력 채널 |
| `-P` | `--partition` | `1` | FL 클라이언트 Partition ID |
| `-G` | `--lgrp`     | `[[1,2,4]]` | 레이블 그룹 |
| `-N` | `--lnam`     | `[wt]` | 레이블 이름 |
| `-I` | `--lidx`     | `[1]` | 레이블 인덱스 |
| | `--block`    | `residual` | 블록 타입 |
| | `--channels` | `[32,64,128,256]` | 채널 수 |
| | `--norm`     | `instance` | 정규화 |
| `-E` | `--epochs`   | `30` | 에폭 수 |
| | `--batch`    | `2` | 배치 크기 |
| | `--lr`       | `5e-3` | 학습률 |
| | `--gpu`      | `1` | GPU 사용 여부 (1/0) |
| `-J` | `--job`      | `test_run` | 실험 이름 |
| | `--ckpt-root` | `/checkpoints` | 체크포인트 루트 경로 |

args 기본값은 컨테이너 내부 마운트 경로(`/data`, `/experiments`, `/checkpoints`) 기준입니다.

### 컨테이너 마운트 구조

```
호스트                  컨테이너
./scripts      →  /app           (코드)
./data         →  /data          (-D 경로)
./experiments  →  /experiments   (-c 경로)
./checkpoints  →  /checkpoints   (--ckpt-root 경로)
```

코드(`scripts/`)는 이미지에 포함되지 않고 런타임에 마운트되므로, 코드 수정 후 재빌드 없이 재실행만 하면 반영됩니다.

### `Trainer` (`scripts/trainer.py`)

- loss: Binary Dice loss
- optimizer: Adam
- 에폭마다 `{ckpt_root}/{job}/latest.pt` 저장
- val_loss 개선 시 `best.pt` 추가 저장
- 시작 시 `latest.pt` 존재하면 자동 resume

## 실행

### 로컬 (Python 직접 실행)

로컬에 PyTorch가 설치된 경우 사용합니다. 경로 기본값이 컨테이너 기준이므로 직접 지정이 필요합니다.

```bash
python3 scripts/app.py \
  -D ./data/fets128/trainval \
  -c ./experiments/fets/partition2/fets_split.csv \
  --ckpt-root ./checkpoints
```

### 컨테이너 (Podman)

```bash
# 빌드 (최초 1회)
podman build -f builds/Containerfile -t simple-fedpod:dev

# 실행
podman run --gpus 1 \
  -v ./scripts:/app:z \
  -v ./data:/data:z \
  -v ./experiments:/experiments:z \
  -v ./checkpoints:/checkpoints:z \
  simple-fedpod:dev
```
