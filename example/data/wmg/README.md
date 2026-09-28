# WMG NMC622 캘린더링 DoE 데이터셋 (키: `wmg`)

『2차전지 AI 실무』 교재 실습용 가공본. 원본 엑셀 3개를 `pandas.read_csv()` 한 줄로 읽히는
tidy CSV로 변환한 것이다. 모든 수치는 원본 파일을 직접 열어 실측한 값이다.

---

## 1. 출처와 라이선스

| 항목 | 내용 |
|---|---|
| 데이터셋 | Characteristics of Electrodes and Lithium-ion Cells at Pilot-Plant Manufacturing Scale |
| 제작 | Faraji-Niri, M. 외 — WMG, University of Warwick / Faraday Institution |
| 저장소 | Mendeley Data, V1 |
| DOI | 10.17632/wwhm2frfmy.1 |
| 라이선스 | **CC0 1.0 (퍼블릭 도메인 기여)** — 저작권 제한 없이 복제·수정·배포·상업적 이용 가능 |
| 원본 아카이브 | ZIP 38,550,667 bytes, 압축해제 153,752,392 bytes (177개 파일) |
| SHA-256 | `bf93b0196d861838a7eeee8267d66a38cfda23306db76e7af6a10e79f077e9b3` (Mendeley API 게시 해시와 일치 확인) |

CC0이므로 교재 수록에 별도 허가가 필요 없다. 다만 학술 관례상 출처 표기를 권장한다.

> Faraji-Niri, M. et al. (2023). *Characteristics of Electrodes and Lithium-ion Cells at
> Pilot-Plant Manufacturing Scale* [Data set]. Mendeley Data, V1.
> https://doi.org/10.17632/wwhm2frfmy.1 (CC0 1.0)

### 원본 아카이브 전체 구성 (실측)

| 폴더 | 파일 수 | 용량 | 내용 |
|---|---:|---:|---|
| `1- Tables/` | 3 | 546,704 B (0.52 MiB) | **본 가공본의 원천** — 엑셀 요약표 |
| `2- Megtec and Mesys/` | 8 | 6,607,375 B (6.30 MiB) | 코터/건조기 시계열 로그 (xlsx) |
| `3- Biologic/` | 165 | 135,722,442 B (129.44 MiB) | Biologic 충방전기 원시 파일 (.mpt), 셀 55종 × 3파일 |
| `4- Images/` | 1 | 10,875,871 B (10.37 MiB) | SEM/EDS 이미지 모음 (.docx) |

실습에 필요한 것은 `1- Tables/` 3개뿐이므로 나머지는 배포하지 않는다.
원본 3개는 `raw/` 폴더에 그대로 보관했다(합계 534 KB).

---

## 2. 실험 설계 (DoE)

NMC622(BASF) 양극을 파일럿 라인에서 코팅한 뒤 **캘린더링 조건을 3인자 완전요인**으로 바꿨다.

| 인자 | 수준 | 컬럼 |
|---|---|---|
| 코팅중량 | 2수준: L=122.48, H=182.73 g/m² | `coat_weight_level`, `target_coat_weight_gsm` |
| 롤 온도 | 3수준: 85, 120, 145 °C | `roll_temp_c` |
| 목표 밀도 | 3수준: P=2.7, M=2.95, D=3.2 g/cm³ | `density_level`, `target_density_g_cm3` |

→ 2 × 3 × 3 = **18조건**, 조건당 하프셀 **3개 반복** = **54셀**. 결측 조건 없이 완전 균형.
(대극 Li 금속, 전해액 1M LiPF6 EC/EMC 3/7 + 1 wt% VC, 세퍼레이터 H1609)

---

## 3. 파일 구성 (실측 행수·용량)

| 파일 | 행 | 열 | 용량 | 단위(1행) |
|---|---:|---:|---:|---|
| `wmg_cells_54.csv` | 54 | 58 | 35,021 B | **하프셀 1개** ← 주력 모델링 테이블 |
| `wmg_conditions_18.csv` | 18 | 51 | 7,335 B | 캘린더링 조건 1개 (전극 물성 + 표준편차) |
| `wmg_rate_long.csv` | 648 | 13 | 90,726 B | 셀 × 레이트시험 스텝 (54 × 12) |
| `wmg_asi_long.csv` | 594 | 8 | 36,694 B | 셀 × ASI 측정점 (54 × 11) |
| `wmg_cycling_long.csv` | 2,808 | 8 | 229,896 B | 셀 × 사이클 (54 × 52) |
| `data_dictionary.csv` | 138 | 6 | 19,807 B | 위 5개 파일의 **전 컬럼** 한글 설명 |
| `raw/*.xlsx` | 3개 | | 546,704 B | 원본 엑셀 (무변경) |
| `prepare.py` | | | | 원본 → CSV 재현 스크립트 |

인코딩은 전부 UTF-8(BOM). 엑셀에서 더블클릭해도 한글이 깨지지 않는다.

```python
import pandas as pd
df = pd.read_csv('wmg_cells_54.csv')          # 54 x 58, 전부 숫자/문자 정상 파싱
```

---

## 4. 원본 구조와 가공 내용

원본 엑셀은 **그대로는 모델링에 못 쓴다.** 실측한 구조는 다음과 같다.

**`Intermediate measurements during calendering.xlsx`** (37,862 B)
- 시트 2개: `Cathode`(41행 × 36열), `Cathode-Intermediate`(41행 × 33열)
- 1~2행이 **병합된 2단 헤더**(`Cathode` 시트 병합영역 14개). 3~20행 = 평균값 블록,
  22~23행에 헤더가 **다시 나오고** 24~41행 = 표준편차 블록. 즉 한 시트에 표 2개가 세로로 쌓여 있다.
- `Cathode-Intermediate`는 패스(pass) 1·2·3을 **가로로** 반복 배치한 전형적 wide 레이아웃.

**`Half-cell (Cathode) Electrochemical Performance.xlsx`** (310,789 B)
- 시트 19개: `Table`(범례) + `Group1`~`Group18`
- `Table` 시트는 A1:DD20으로 잡히지만 **실제 값은 A~E열뿐**, F~DD(103열)는 서식만 남은 빈 열이다.
- `GroupN` 시트는 **완전 전치(transposed)** 구조 — 행이 변수, 열이 셀.
  A열=섹션(세로 병합 11개), B열=변수명, C·D·E열=반복 셀 3개, F열=Mean, G열=Standard deviation.
- 행 라벨이 **중복**된다. `At C/5`가 한 섹션에 5번, ASI의 `0.5`/`0.2`가 2번씩. 라벨만으로는 못 찾는다.
- `Group14`만 상단에 4행이 더 있어 행 번호가 8칸 밀린다(159행). **행 번호 하드코딩 불가.**

**`Half-cell Electrochemical Performance Cycling Performance.xlsx`** (198,053 B)
- 시트 1개(217행 × 56열). 열이 54개 셀, 행이 사이클. 세로로 블록 4개가 쌓여 있다.
  방전용량(4~55행, cycle 0~51) / 충전용량(59~110행, 0~51) / 용량유지율(114~163행, 1~50) /
  쿨롱효율(168~217행, 1~50).

**가공 방침**: 위 구조를 (섹션, 라벨, 등장순번) 키로 파싱해 셀 단위로 전치를 풀고,
캘린더링 조건표를 `group`으로 조인했다. 원본 값은 반올림 없이 그대로 옮겼다
(원본 셀과 1e-9 이내 일치를 스팟체크로 확인).

---

## 5. 결측·이상값 (전량 실측)

### 결측 — 원인까지 특정 완료

| 위치 | 건수 | 내용 |
|---|---:|---|
| `wmg_cells_54.csv` `asi_*` 6열 | 6 | **DD020** 한 셀의 ASI 22개 값이 원본에서 통째로 빈칸 |
| `wmg_cells_54.csv` `cyc_*` 4열 | 4 | **DD059**의 사이클 요약 4행이 원본에서 빈칸 (단, `wmg_cycling_long.csv`에는 데이터 존재) |
| `wmg_cells_54.csv` `first_cycle_loss_pct` | 1 | **DD001** (C/20 충전용량이 없어 계산 불가) |
| `wmg_cells_54.csv` `v_after_assembly_v` | 3 | DD034, DD035, DD046 |
| `wmg_cycling_long.csv` `cap_chg_mah` | 54 | **cycle 0 전 셀.** 원본 라벨: "C/10- not accurate due to possibly incomplete discharge during previous cycle" |
| `wmg_cycling_long.csv` `retention_pct`, `coulombic_eff_pct` | 108씩 | cycle 0·51에는 해당 블록이 없음 (원본이 1~50만 제공) |
| `wmg_conditions_18.csv` | 0 | 결측 없음 |

### 이상값

1. **10C 시험 실패 7셀** — `grav_dis_10c_mahg`가 0.0009 mAh/g 수준(실질 0):
   DD027, DD028, DD059(조건10) / DD051(조건13) / DD029, DD030, DD031(조건16).
   **7셀 전부 `density_level == 'P'`(고공극)이고 전부 H(고코팅중량)**이다. NaN이 아니라
   "0에 가까운 숫자"로 들어 있어 `dropna()`로 안 걸러진다.
2. **DD020** — `grav_dis_c20_mahg` = 226.8 mAh/g. NMC622의 4.2V 실용 용량(약 165~180)을
   크게 넘는 물리적으로 불가능한 값. 이 셀은 ASI도 전부 결측이다. 제외 검토 대상.
3. **DD036, DD039** — `first_cycle_loss_pct` 26.2%, 26.0% (나머지 51셀 중앙값 약 9.1%).
4. **용량유지율 100% 초과** — 54셀 중 31셀에서 발생(총 315행). 초기 젖음/활성화로 실제
   나타나는 현상이지만, **DD021**은 101.5%까지 튀고 사이클 간 진동이 심하다(쿨롱효율도 66~91%로 요동).
5. **원본 Mean 열 오염(중요)** — DD001의 C/20 충전용량은 빈칸인데, 같은 셀의
   그램당·부피당 충전용량 칸에는 **0이 적혀 있다.** 그 결과 원본 F열 Mean이
   (0 + 182.91 + 183.09)/3 = **122.00**으로 계산돼 있다. 참값은 약 183이다.
   → 본 가공본은 이 0을 결측으로 되돌렸고(2건), **원본의 Mean·SD 열은 일절 사용하지 않았다.**

### 단위

`data_dictionary.csv`의 `unit` 열에 전 컬럼 명시. 주요 단위:
mAh(용량), mAh/g(그램당), mAh/cm³(부피당), Ohm·cm²(ASI), g/m²(코팅중량), g/cm³(밀도),
µm(두께·롤갭), kPa(인장강도), °C(롤 온도), %(공극률·유지율·손실률).
공극률은 진밀도 4.458 g/cm³ 기준으로 계산된 값이다.

---

## 6. 이 데이터로 가르칠 수 있는 것

**Day 1 — 데이터 읽기·정리**
- 병합헤더·전치·중복라벨·한 시트 안 두 개 표 → "현장 엑셀이 왜 그대로는 안 되는가"의 교과서적 실례.
  `prepare.py`를 그대로 읽히며 해체 과정을 설명할 수 있다.
- `df.isna().sum()`으로 결측을 세고, **위 5번(원본 Mean 오염)** 을 직접 재현시키면
  "평균을 남이 계산해 준 값을 믿지 마라"가 숫자로 증명된다.

**Day 2 — 탐색적 분석·시각화**
- 목표 공극률 대비 실측 공극률 오차(조건 평균 −0.41 ~ +1.56 %p). 목표를 못 맞추는 조건이 어디인지.
- 완전균형 2×3×3 설계라 `groupby` 3단, `pivot_table`, 상자그림, 교호작용 도표가 전부 깔끔히 나온다.
- **레이트 특성의 주효과가 매우 뚜렷하다**: 5C 그램당 방전용량 평균이
  L(얇은 전극) 118~128 mAh/g vs H(두꺼운 전극) 30~72 mAh/g.
  두께가 늘면 물질전달이 율속이 된다는 전기화학 상식이 데이터로 바로 보인다.

**Day 3 — 모델링**
- X = 설계인자 8개(+실측 전극물성), y = `grav_dis_5c_mahg` / `retention_50cyc_pct` /
  `asi_dis_soc50_ohmcm2` 등. 회귀·트리 모델 모두 적용 가능.
- 표본이 54개뿐이라 **train/test 분할을 셀 단위로 하면 안 된다**(같은 조건 3셀이 양쪽에 섞이면 누수).
  `group`(조건) 단위 GroupKFold를 가르치기에 이상적인 크기다.
- 조건 내 반복 3셀의 산포 = 실험 재현성의 하한선. 실측한 조건내 SD 중앙값 vs 전체 SD:
  `grav_dis_c20_mahg` 0.84 vs 8.90 / `retention_50cyc_pct` 1.50 vs 7.74 /
  `asi_dis_soc50_ohmcm2` 5.16 vs 34.05. → "모델이 이 이하로 맞추면 과적합"이라는 기준선을 준다.

### 반드시 짚어야 할 함정

- **이상치가 상관계수 부호를 뒤집는다.** 공극률 vs 10C 그램당용량 피어슨 상관은
  전체 54셀 **−0.115**, 위 10C 실패 7셀을 빼면 **+0.081**. 부호가 바뀐다.
  "상관계수 하나 뽑아 결론 내지 마라"를 실측으로 보여줄 수 있는 사례.
- **캘린더링 날짜와 롤 온도가 완전 교락(confounded)** 되어 있다.
  85°C=2021-10-15, 120°C=10-18, 145°C=10-27로 1:1 대응(교차표 확인 완료).
  온도 효과와 날짜(재료 로트·습도 등) 효과를 이 데이터만으로는 분리할 수 없다.
- **캘린더링 전 SEM/EDS·인장강도는 코팅중량 수준당 값이 딱 1개다.**
  `precal_grad_carbon`(H=0.01758, L=0.00654), `precal_moran_carbon`(H=0.646, L=0.605),
  `precal_tensile_strength_kpa`(H=683.25, L=728.62) 등은 18조건에 걸쳐 고유값이 2개뿐이다.
  → `coat_weight_level`과 완전 공선(collinear). 예측변수로 넣으면 안 된다. 좋은 공선성 실습 재료.
- **충방전기 장비가 조건과 부분 교락**: BCS7은 11셀 전부 L, BCS8은 12셀 중 11셀이 H.
  장비 효과와 코팅중량 효과가 섞여 있다. `test_equipment`를 잡음인자로 넣어 확인시킬 것.
- **`n_passes`는 설계값이 아니라 결과값**이다(목표 밀도에 도달할 때까지 반복 통과, 1~3회).
  X에 넣을 때 인과 해석 주의.
- **ASI의 `0.5`, `0.2` 라벨이 두 번씩 등장**하는데 두 번째 것이 무엇인지 **원본에 설명이 없다.**
  값도 다르다(예 DD001: 0.5 → 52.6 vs 43.6). `wmg_asi_long.csv`의 `occurrence` 열로 구분해
  두었으니 1번만 쓰는 것을 권장한다. 의미를 아는 척하지 말 것.
- 레이트 시험에서 **충전은 항상 C/5 고정**, 방전만 C/2→10C로 올린다. 원본 충전용량 행 라벨이
  전부 `At C/5`라 방전 레이트와 헷갈리기 쉽다(`wmg_rate_long.csv`에서 분리해 둠).

---

## 7. 재현

```bash
cd <이 폴더>
python prepare.py      # raw/*.xlsx -> CSV 6개 재생성, 정합성 검증까지 수행
```

`prepare.py`는 `raw/` 폴더만 읽고 같은 폴더에 CSV를 쓴다. 필요 패키지는 `openpyxl`뿐이다
(검증 단계에서만 pandas 사용). 마지막에 셀 수 54, 셀ID 중복 없음, 메인/사이클 테이블 셀ID 일치를
확인하고 `PASS`를 출력한다. 컬럼 사전은 5개 CSV의 실제 헤더와 대조해 **누락·잉여가 있으면
`AssertionError`로 중단**하도록 되어 있다.
