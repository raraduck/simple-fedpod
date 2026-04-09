# TensorBoard 태그 구조 참조

## 태그 명명 규칙

```
(ech|rnd)_(trn|val)_(loss|dice)_(avg|prv|pst|prvpst) / (avg|{name})
│          │         │            │                      └─ 서브태그
│          │         │            └─ 측정 시점
│          │         └─ 지표 종류
│          └─ train / val
└─ epoch 축 / round 축
```

## 전체 태그 목록

### round 축 (`x = round 번호`)

| 태그 | 값 | 기록 위치 |
|------|----|-----------|
| `rnd_trn_loss_avg/avg` | 라운드 평균 train loss | inst{PP}, inst_avg |
| `rnd_val_loss_avg/avg` | 라운드 평균 val loss | inst{PP}, inst_avg |
| `rnd_val_loss_prv/avg` | 학습 전 val loss | inst{PP}, inst_avg, gen_eval |
| `rnd_val_loss_pst/avg` | 학습 후 val loss | inst{PP}, inst_avg |
| `rnd_val_loss_prvpst/avg` | prv→pst 오버레이 | inst{PP}, inst_avg, gen_eval(prv) |
| `rnd_val_dice_prv/avg` | 학습 전 평균 Dice | inst{PP}, inst_avg, gen_eval |
| `rnd_val_dice_pst/avg` | 학습 후 평균 Dice | inst{PP}, inst_avg |
| `rnd_val_dice_prvpst/avg` | prv→pst 오버레이 | inst{PP}, inst_avg, gen_eval(prv) |
| `rnd_val_dice_prv/{name}` | 학습 전 클래스별 Dice | inst{PP}, inst_avg, gen_eval |
| `rnd_val_dice_pst/{name}` | 학습 후 클래스별 Dice | inst{PP}, inst_avg |
| `rnd_val_dice_prvpst/{name}` | prv→pst 오버레이 | inst{PP}, inst_avg, gen_eval(prv) |

### epoch 축 (`x = global epoch 번호`)

| 태그 | 값 | x 위치 |
|------|----|--------|
| `ech_val_loss_avg/avg` | 라운드 평균 val loss | epoch_end |
| `ech_val_loss_prv/avg` | 학습 전 val loss | epoch_start |
| `ech_val_loss_pst/avg` | 학습 후 val loss | epoch_end |
| `ech_val_loss_prvpst/avg` | prv→pst 오버레이 | start→end |
| `ech_val_dice_prv/(avg\|{name})` | 학습 전 Dice | epoch_start |
| `ech_val_dice_pst/(avg\|{name})` | 학습 후 Dice | epoch_end |
| `ech_val_dice_prvpst/(avg\|{name})` | prv→pst 오버레이 | start→end |

epoch_start = `round × epochs`, epoch_end = `(round + 1) × epochs`

## runs 디렉토리 구조

```
runs/{job}/
  inst{PP}/    ← 기관별 (app.py — 학습 완료 후)
  inst_avg/    ← 기관 평균 (agg.py — 라운드 집계 완료 후)
  gen_eval/    ← 일반화 평가 (gen_eval.py — train-job 과 병렬)
```

## 기록 주체

| 스크립트 | 기록 대상 | 시점 |
|----------|-----------|------|
| `app.py` | `inst{PP}/` | 라운드 학습 완료 후 |
| `agg.py` | `inst_avg/` | FedAvg 집계 완료 후 |
| `gen_eval.py` | `gen_eval/` | train-job 과 동시 (병렬 pod) |

## Dice 계산 방식

- **hard Dice**: `sigmoid(logit) > 0.5` binary prediction 기준
- **집계**: val set 전체 TP/FP/FN 합산 후 `2TP / (2TP + FP + FN)`
- **클래스**: `--lnam` 인자로 지정 (stage2: `wt, tc, et`)

## gen_eval 특이사항

- val 대상: split CSV에서 `Partition_ID.notna()` + `TrainOrVal == "val"` 전체
  (`--partitions` 인자로 필터링하지 않음)
- `prv` 태그만 기록 (fine-tuning 전 글로벌 모델 평가)
- `prvpst` 패널에도 prv 값을 기록 → inst_avg 의 prv→pst 커브와 오버레이 가능

## TensorBoard 비교 명령 예시

```bash
# 단일 실험 전체 보기
tensorboard --logdir runs/stage2-entropy

# 두 실험 비교
tensorboard --logdir_spec \
  entropy:runs/stage2-entropy,\
  random:runs/stage2-random

# stage1 vs stage2
tensorboard --logdir_spec \
  stage1:runs/stage1,\
  stage2:runs/stage2-entropy
```
