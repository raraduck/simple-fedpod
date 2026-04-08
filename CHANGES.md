# CHANGES.md

## 2026-04-08 (2)

### scripts/agg.py
- `--committee-job`, `--committee-rounds`, `--committee-round`, `--committee-partitions` 인자 추가
  - 체크포인트 구조 기반으로 committee 경로 자동 생성
  - `--committee-partitions` 콤마 구분 ID 목록으로 기관 선택 지정 (e.g. `1,2,5,7`)
  - 기존 `--committee` 직접 경로 방식과 병행 지원

### tests/
- `2026-04-07_stage1-stage2-entropy-selection.md` Step 7 커맨드를 `--committee-job` 방식으로 업데이트

---

## 2026-04-08 (1)

### scripts/agg.py
- `--selection random|entropy` 인자 추가 — subject 선택 방식
- `--committee` 인자 추가 — entropy 추론용 사전 모델 경로 (콤마 구분 복수 경로/폴더 지원)
- `--gpu`, `-D/--data`, `--chan` 인자 추가 — entropy 추론 설정
- `_build_committee()`, `_sample_train_entropy()`, `_build_model()` 메서드 추가
- dry-run에서 `--committee` 제공 시 entropy selection 가능 (stage2 지원)
- **버그 수정**: CSV 재읽기 시 `Int64` 컬럼이 `float`으로 변환되는 문제
- **버그 수정**: entropy global k 공식 오류 (`round(lam * P)` → `sum(min(round(lam), n_i))`) — rate=1.0에서 전체 subjects가 선택되던 문제

### scripts/app.py
- **버그 수정**: epoch 루프 상한 오류 (`epochs+1` → `epoch+epochs+1`) — round 1 이상에서 학습이 즉시 종료되던 문제

### TEST.md / tests/
- 테스트 파일을 `tests/{날짜}_{테스트명}.md` 단위로 분리 관리
- `tests/2026-04-07_stage1-stage2-entropy-selection.md` 추가
