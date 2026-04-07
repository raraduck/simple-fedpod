# Katib HPO

## 파일 구성

```
katib/
  Containerfile   # 이미지 빌드 정의 (scripts/ COPY 포함)
  experiment.yaml # Katib Experiment 정의
```

## 이미지

- 이미지명: `192.168.0.80:30002/dwnkim/katib-fedpod:v0.1`
- 코드(`scripts/`)가 이미지에 포함됨 — 코드 변경 시 재빌드 필요
- Argo 이미지와 달리 볼륨 마운트 없이 K8s Pod에서 독립 실행

```bash
# 빌드 (프로젝트 루트에서 실행)
podman build -f katib/Containerfile -t 192.168.0.80:30002/dwnkim/katib-fedpod:v0.1 .
podman push 192.168.0.80:30002/dwnkim/katib-fedpod:v0.1
```

## Experiment 설정

| 항목 | 값 |
|------|----|
| objective | minimize `val_loss` |
| goal | 0.20 |
| algorithm | multivariate-tpe |
| parallelTrialCount | 2 |
| maxTrialCount | 4 |
| maxFailedTrialCount | 3 |
| GPU | rtx3090 × 1 |
| epochs per trial | 10 |

## HPO 파라미터

| 파라미터 | 타입 | 범위 |
|----------|------|------|
| `lr` | double | 0.0001 ~ 0.01 |
| `batch` | int | 1 ~ 3 |

## 메트릭 수집

Katib StdOut collector가 아래 포맷을 파싱:

```
{metricName: val_loss, metricValue: 0.2741}
```

`trainer.py`의 `val_epoch()`에서 매 epoch 출력됨.

## Experiment 실행

```bash
kubectl apply -f katib/experiment.yaml
```

## NFS 마운트 구조

| NFS 경로 | 컨테이너 경로 | 용도 |
|----------|---------------|------|
| `.../data` | `/data` | 학습 데이터 (`-D`) |
| `.../experiments` | `/experiments` | 분할 CSV (`-c`) |

체크포인트는 `/tmp/checkpoints` (Pod 임시 저장소, 종료 시 삭제됨)
