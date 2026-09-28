# Battery RUL 데이터 (실습용 가공본)

원본: HNEI(Hawaii Natural Energy Institute) 유래 파생본, Kaggle `ignaciovinuales/battery-remaining-useful-life-rul`
라이선스: CC0 (파생본 표기 기준) — 상세·인용 서지는 상위 폴더 `DATASET_CARDS.md` 참조

## 파일 구성

| 파일 | 규모 | 설명 |
|---|---|---|
| `battery_rul_original.csv` | 15,064 × 9 | 다운로드 원본(무가공). 재현용 보관본 |
| `battery_rul_cells.csv` | 15,064 × 18 | **주 실습 파일.** 원본 + `cell_id`(1~14 복원) + 이상값 플래그 7종 + `is_valid` |
| `battery_rul_clean.csv` | 14,996 × 10 | 물리적 불가 68행 제거본(hard 규칙 4종 합집합) |
| `cell_summary.csv` | 14 × 12 | 셀별 사이클 수·RUL 범위·이상값 건수 요약 |
| `data_dictionary.csv` | — | 컬럼 사전(한글 병기) |
| `prepare.py` | — | 재현 스크립트 |

## 재현 방법

```bash
python prepare.py --src battery_rul_original.csv --out .
```

## 교육 포인트 (실측 근거)

- **셀 식별자가 없다.** `Cycle_Index`가 리셋되는 지점으로 14개 셀을 복원해야 GroupShuffleSplit이 가능하다. 무작위 분할하면 같은 셀의 인접 사이클이 학습·평가에 나뉘어 누수가 발생한다.
- **물리적으로 불가능한 값**: `Decrement 3.6-3.4V (s)` 음수 24건(최소 −397,645초), `Time at 4.15V (s)` 음수 9건. 시간 구간 길이는 음수일 수 없다 — 도메인 지식으로만 잡히는 이상치.
- **부분이 전체보다 큰 모순**: `Time at 4.15V > Charging time` 28건, `|Decrement| > Discharge Time` 34건.
- **타깃 누수 후보**: `Cycle_Index`와 `RUL`의 관계를 확인할 것(누수 시연 실습 소재).
- **문서-데이터 불일치**: 원 설명문의 `Total time (s)` 컬럼이 실제 CSV에 없다.
- `cell_code`(a~t)는 원저자 스크립트의 concat 순서에서 **추정한 값**이며 CSV만으로는 검증 불가. `cell_id`(1~14)는 결정론적으로 복원되므로 신뢰 가능.
