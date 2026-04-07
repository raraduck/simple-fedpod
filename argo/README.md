# Argo Workflow

## 파일 구성

```
argo/
  Containerfile   # 이미지 빌드 정의
  workflow.yaml   # 단일 job 테스트용
  stage1.yaml     # 병렬 학습 DAG (withItems 패턴)
```

## 이미지

- 이미지명: `192.168.0.80:30002/dwnkim/argo-fedpod:v0.2`
- 코드(`scripts/`)는 이미지에 포함되지 않으며 NFS에서 `/app`으로 마운트됨
- NFS 파일 수정 시 이미지 재빌드 없이 반영됨

```bash
# 빌드 (프로젝트 루트에서 실행)
podman build -f argo/Containerfile -t 192.168.0.80:30002/dwnkim/argo-fedpod:v0.2 .
podman push 192.168.0.80:30002/dwnkim/argo-fedpod:v0.2
```

## NFS 마운트 구조

| NFS 경로 | 컨테이너 경로 | 용도 |
|----------|---------------|------|
| `.../simple-fedpod/scripts` | `/app` | 코드 |
| `.../data` | `/data` | 학습 데이터 (`-D`) |
| `.../experiments` | `/experiments` | 분할 CSV (`-c`) |
| `.../simple-fedpod/checkpoints` | `/checkpoints` | 체크포인트 저장/로드 |

`dshm` (`emptyDir: Memory, 8Gi`) — DataLoader shared memory용

## Workflow 실행

### workflow.yaml — 단일 job 테스트

```bash
argo submit argo/workflow.yaml -n dwnkim --watch
```

### stage1.yaml — 병렬 학습 DAG

`withItems`로 partition 목록을 정의하면 자동으로 병렬 실행됩니다.

```bash
argo submit argo/stage1.yaml -n dwnkim --watch
```

파라미터 오버라이드:
```bash
argo submit argo/stage1.yaml -n dwnkim \
  -p epochs=10 \
  -p image=192.168.0.80:30002/dwnkim/argo-fedpod:v0.3
```

#### DAG 구조

```
stage1 (DAG)
 ├── train (partition=1, job=stage1-p01)  ─┐ 병렬
 └── train (partition=2, job=stage1-p02)  ─┘
```

job 추가 시 `withItems`에 한 줄 추가:
```yaml
withItems:
- { partition: "1",  job-name: "stage1-p01" }
- { partition: "2",  job-name: "stage1-p02" }
- { partition: "3",  job-name: "stage1-p03" }  # 추가
```

#### 공통 파라미터 (`workflow.parameters`)

| 파라미터 | 기본값 |
|----------|--------|
| `image` | `argo-fedpod:v0.2` |
| `data-path` | `/data/fets128/trainval` |
| `split-csv` | `/experiments/fets/partition2/fets_split.csv` |
| `ckpt-root` | `/checkpoints` |
| `epochs` | `3` |

체크포인트는 `{ckpt-root}/{job-name}/` 에 저장되어 partition별로 분리됩니다.

## 로컬 테스트 (Podman)

Podman은 `--gpus` 미지원. 호스트 nvidia 드라이버 라이브러리를 직접 마운트해야 GPU 인식:

```bash
sudo podman run \
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-uvm \
  -v /lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:ro \
  -e LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  --shm-size=8g \
  -v ./scripts:/app:z \
  -v ./data:/data:z \
  -v ./experiments:/experiments:z \
  -v ./checkpoints:/checkpoints:z \
  192.168.0.80:30002/dwnkim/argo-fedpod:v0.2
```
