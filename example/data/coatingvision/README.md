# CoatingVision 전극 코팅 결함 데이터 (교재 실습용 가공본)

## 출처·라이선스
- Sampath, V. **CoatingVision: A Defect Dataset for Coating Manufacturing**. figshare (2025-06-06).
  DOI: [10.6084/m9.figshare.29260121](https://doi.org/10.6084/m9.figshare.29260121)
- 라이선스: **CC BY 4.0** — 출처 표기 시 상업적 이용·재배포·변형 모두 허용. 교재 수록 가능.
- 원본 아카이브: `CoatingVision.zip`, 94,296,797 bytes (실측), 압축해제 98.3 MB / 9,490개 파일.
- 이 폴더는 원본을 가공한 **실습용 축약본(1.65 MB)** 이다. 원본 전체는 위 DOI에서 받는다.

## 폴더 구성
| 경로 | 내용 |
|---|---|
| `coating_labels_tidy.csv` | 2,227행 x 24열. 공정변수·그룹·마스크면적을 붙인 메인 실습 표 |
| `labels_original.csv` | 원본 `classification/labels.csv` 그대로 (2,227행 x 6열) |
| `images_curated/` | 큐레이션 이미지 54장 (JPG 640x480, 원본 파일명 유지) |
| `masks_curated/` | 위 54장 대응 세그멘테이션 마스크 54장 (PNG, 0/255 이진) |
| `detection_labels_curated/` | 위 54장 중 YOLO 박스가 있는 30개 `.txt` |
| `curated_index.csv` | 큐레이션 54장 색인 + 선별 사유 |
| `prepare.py` | 원본 ZIP → 이 폴더를 재생성하는 스크립트 (난수 없음, 결정론적) |

재생성: `python prepare.py --zip CoatingVision.zip --out .`

## 원본 구조 (실측)
```
CoatingVision.zip
├─ classification/  labels.csv + images/ 2,227장
├─ detection/       images/ 2,227장 + labels/ 581개 (.txt, YOLO)
└─ segmentation/    images/ 2,227장 + masks/ 2,227장 (.png)
```
**주의: 세 폴더의 `images/`는 완전히 동일한 파일이다**(샘플 40장 MD5 전수 일치).
같은 이미지 2,227장이 3벌 중복 저장돼 94 MB가 된 것으로, 실제 고유 이미지는 2,227장뿐이다.

## coating_labels_tidy.csv 컬럼 사전
| 컬럼 | 설명 |
|---|---|
| `file_name` | 이미지 파일명 `image_N.jpg`. 이미지·마스크·박스 조인 키 |
| `original_file_name` | 원본 파일명. 공정정보가 인코딩돼 있음 |
| `run` | 코팅 런 번호 (1 또는 7) |
| `coating_gap_um` | **코팅 갭(µm)** — 600/700/800/900/1000/1100 |
| `position` | 촬영 위치 (`top-to-bottom-center`, `top-to-bottom-center-1`, `middle`) |
| `video_id` | 촬영 영상 단위 (런+갭+위치). 8개 |
| `frame` | 영상 내 프레임 번호 (0~673) |
| `patch` | 프레임을 자른 패치 번호 (0~8, 최대 9장) |
| `group_frame` | `video_id + frame`. 같은 프레임에서 나온 패치 묶음. 367개 |
| `Surface_Crack` `Delamination` `Pinhole` `unclassified` | 멀티라벨 0/1 |
| `n_defect_types` | 결함 종류 수(unclassified 제외, 0~3) |
| `any_defect` | 결함 1개 이상이면 1 |
| `label_combo` | 라벨조합 문자열 `'1-0-1-0'` (앞자리 0 보존 위해 하이픈 연결) |
| `defect_set` | 사람이 읽는 조합명 `'Surface_Crack+Pinhole'`, 없으면 `'none'` |
| `is_unlabeled` | 4개 라벨이 모두 0인 행이면 1 (44행) |
| `has_detection_label` `n_boxes` | YOLO 박스 유무 / 개수 |
| `det_class_ids` `det_class_names` | 박스 클래스. 없으면 `'none'` |
| `mask_pixels` `mask_area_frac` | 마스크 결함 화소수 / 전체(640x480) 대비 비율 |

결측치 0개. `pandas.read_csv('coating_labels_tidy.csv')` 한 줄로 바로 사용 가능.

## 파일명 문법 (2,227건 전부 일치, 예외 0건)
```
R{run}-{gap}um-{position}_frame_{frame}_patch_{patch}.png
예) R1-1100um-top-to-bottom-center-1_frame_220_patch_7.png
```

## 라벨 분포 (실측)
멀티라벨이므로 합이 2,227을 넘는다.

| 클래스 | 양성 수 | 비율 |
|---|---:|---:|
| Surface_Crack | 1,947 | 87.4% |
| Pinhole | 519 | 23.3% |
| Delamination | 203 | 9.1% |
| unclassified | 78 | 3.5% |
| (라벨 전무) | 44 | 2.0% |

## 코팅갭 x 결함률 교차표 (실측, %)
| gap(µm) | 패치수 | Surface_Crack | Delamination | Pinhole | unclassified |
|---:|---:|---:|---:|---:|---:|
| 600 | 306 | 59.2 | 0.0 | 38.2 | 10.1 |
| 700 | 407 | 82.8 | 10.8 | 24.3 | 4.4 |
| 800 | 303 | 81.5 | 16.5 | 28.4 | 2.6 |
| 900 | 304 | 95.7 | 1.6 | 12.8 | 4.3 |
| 1000 | 230 | 98.3 | 9.1 | 20.9 | 0.9 |
| 1100 | 677 | 98.2 | 12.3 | 19.2 | 0.9 |

갭이 커질수록 Surface_Crack은 오르고(59→98%) Pinhole은 내려간다(38→19%).
**단, 아래 "함정 2"를 반드시 같이 가르칠 것.**

## 수업에서 반드시 짚어야 할 함정 (모두 실측 확인)

**함정 1 — 데이터 누수는 프레임이 아니라 "영상" 단계에서 터진다.**
32x32 그레이스케일 + RandomForest 로 측정한 5-fold AUC:

| 대상 | 랜덤 KFold | GroupKFold(frame) | GroupKFold(video) |
|---|---:|---:|---:|
| Surface_Crack | 0.896 | 0.907 | **0.702** |
| Pinhole | 0.701 | 0.712 | **0.487** |

프레임 단위로 묶어도 성능이 안 떨어진다(패치가 3x3로 공간 분리돼 있어서다).
그러나 **영상 단위로 묶으면 Pinhole은 AUC 0.49로 무작위 수준까지 붕괴**한다.
모델이 결함이 아니라 "그 영상의 조명·배경"을 외우고 있었다는 뜻이다.
→ 실습에서는 `groups=video_id` 로 나눠야 현장에서 통하는 숫자가 나온다.

**함정 2 — 코팅갭 효과와 영상이 완전히 교락(confounding)돼 있다.**
갭 1수준당 영상이 1~2개뿐이라, "갭의 효과"와 "그 영상의 효과"를 분리할 수 없다.
같은 700µm인데 런이 다르면 Delamination이 0.4%(R1) vs 32.6%(R7)로 갈린다.
갭-결함률 표를 인과관계로 읽으면 안 된다.

**함정 3 — 라벨이 하나도 없는 44장.**
4개 라벨이 전부 0인 행이 44건이며, 이들의 마스크는 전부 빈 마스크다(교차 확인).
"정상"인지 "미라벨"인지 원본에 명시가 없다. 이진분류 타깃을 만들 때
이 44장을 음성으로 쓸지 버릴지 학생에게 직접 결정시키면 좋은 토론거리가 된다.

**함정 4 — 세 과제의 대상이 다르다.**
디텍션 박스는 Pinhole·unclassified 에만 있고 Surface_Crack·Delamination 에는 없다.
세그멘테이션 마스크는 클래스 구분 없는 **이진 마스크**(0/255)다.

## 검출(detection) 라벨 주의
- YOLO 형식 5토큰(`class cx cy w h`), 정규화 범위 이탈 0건, 총 649개 박스.
- 581개 파일에만 존재. 나머지 1,646장은 파일 자체가 없다(YOLO 관례상 "박스 없음").
- **원본에 `data.yaml`·`classes.txt`가 없어 클래스 이름이 명시되지 않았다.**
  본 가공본의 `id 0 = Pinhole`, `id 1 = unclassified` 매핑은 실측 교차검증으로
  **추론**한 값이다(id 0 → Pinhole 양성 519장과 정확히 1:1, id 1 → unclassified
  양성 78장과 정확히 1:1, 불일치 0건). 원저자 공식 정의는 아니다.

## 큐레이션 54장 선별 기준
라벨조합(9종, 각 20장 이상 확보된 것)마다 6장씩, 코팅갭이 골고루 섞이도록
라운드로빈으로 선택. 같은 프레임 재사용 금지(54장 중 53개 프레임이 서로 다름).
전 과정 결정론적이라 몇 번 돌려도 동일한 54장이 나온다. 상세는 `prepare.py` 주석 참조.
