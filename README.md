# simple-fedpod

FL(Federated Learning) 클라이언트로 동작하는 컨테이너 기반 학습 pod입니다.

## 코드 구조

```
scripts/
  app.py          # 진입점 — args 파싱, App.run() 흐름
  dataset.py      # FeTSDataset, load_split
  trainer.py      # Trainer — BCE+SoftDice loss, Adam, 체크포인트 저장/로드
  models/
    unet3d.py     # 3D Residual U-Net
    loss.py       # SoftDiceBCEWithLogitsLoss
argo/
  Containerfile   # Argo 이미지 — scripts/ 는 NFS 마운트 (COPY 없음)
  workflow.yaml   # Argo Workflow 정의
katib/
  Containerfile   # Katib 이미지 — scripts/ 를 이미지에 COPY
  experiment.yaml # HPO Experiment — lr, batch
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

### 이미지 구분

| 이미지 | Containerfile | 코드 포함 방식 | 용도 |
|--------|--------------|----------------|------|
| `argo-fedpod:v0.2` | `argo/Containerfile` | NFS 마운트 (`/app`) | Argo Workflow 학습 |
| `simple-fedpod:katib` | `katib/Containerfile` | 이미지에 COPY | Katib HPO trial |

Argo 이미지는 코드를 포함하지 않으므로 `scripts/`를 NFS에 올려두면 이미지 재빌드 없이 코드 변경이 반영됩니다.

### `Trainer` (`scripts/trainer.py`)

- loss: `SoftDiceBCEWithLogitsLoss` (BCE + SoftDice, logits 직접 입력)
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

### 컨테이너 (Podman — 로컬 테스트)

```bash
# Argo 이미지 빌드
podman build -f argo/Containerfile -t argo-fedpod:v0.2 .

# 실행 (scripts/ 바인드 마운트)
# --device /dev/nvidia* : GPU 디바이스 직접 지정 (Podman에서 --gpus 사용 불가)
# --shm-size=8g : DataLoader num_workers>0 사용 시 shared memory 필요
sudo podman run \
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-uvm \
  --shm-size=8g \
  -v ./scripts:/app:z \
  -v ./data:/data:z \
  -v ./experiments:/experiments:z \
  -v ./checkpoints:/checkpoints:z \
  192.168.0.80:30002/dwnkim/argo-fedpod:v0.2

# Katib 이미지 빌드 (scripts/ COPY 포함)
podman build -f katib/Containerfile -t simple-fedpod:katib .
```
