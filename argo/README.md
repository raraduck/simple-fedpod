# Argo Workflow

## 파일 구성

```
argo/
  Containerfile   # 이미지 빌드 정의
  workflow.yaml   # Argo Workflow 실행 정의
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
| `.../checkpoints` | `/checkpoints` | 체크포인트 저장/로드 |

`dshm` (`emptyDir: Memory, 8Gi`) — DataLoader shared memory용

## Workflow 실행

```bash
# 제출
kubectl apply -f argo/workflow.yaml

# 또는 argo CLI
argo submit argo/workflow.yaml -n dwnkim
argo submit argo/workflow.yaml -n dwnkim --watch
```

## 현재 학습 설정

| 항목 | 값 |
|------|----|
| 이미지 | `argo-fedpod:v0.2` |
| GPU | rtx3090 × 1 |
| epochs | 3 |
| 데이터 | `/data/fets128/trainval` |
| split | `/experiments/fets/partition2/fets_split.csv` |
| checkpoint | `/checkpoints` |

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
