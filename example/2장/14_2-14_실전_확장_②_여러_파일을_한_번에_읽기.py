# -*- coding: utf-8 -*-
"""2장 예제 — 2.14 실전 확장 ② 여러 파일을 한 번에 읽기 (20분) `선택 학습(자율 복습)`

교재 출처 : manuscript/21_ch2_pandas_입출력.md
실행 방법 : example 폴더에서  python "2장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 11_2-11_SQLite_데이터베이스_연결.py, 12_2-12_연습문제.py, 13_2-13_실전_확장_①_엑셀_파일_읽기.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd

# --------------------------------------------------------------------
# [glob — 파일 목록을 코드로 얻는다 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
import glob

paths = sorted(glob.glob("data/*/*.csv"))
print("찾은 CSV 파일:", len(paths), "개")
for p in paths[:4]:
    print("  ", p)

# 교재 실행 결과 ------------------------------------------------------
#   찾은 CSV 파일: 22 개
#      data\coatingvision\coating_labels_tidy.csv
#      data\coatingvision\curated_index.csv
#      data\coatingvision\labels_original.csv
#      data\eis\eis_data_dictionary.csv

# --------------------------------------------------------------------
# [파일 인벤토리 만들기 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
import os

records = []
for p in paths:
    try:
        d = pd.read_csv(p)
    except Exception as e:
        print("읽기 실패:", p, e)
        continue
    records.append({"폴더": os.path.basename(os.path.dirname(p)),
                    "파일": os.path.basename(p),
                    "행": len(d),
                    "열": d.shape[1],
                    "KB": round(os.path.getsize(p) / 1024)})

inv = pd.DataFrame(records)
print(inv.to_string(index=False))
print("합계:", inv["행"].sum(), "행 /", inv["KB"].sum(), "KB")

# 교재 실행 결과 ------------------------------------------------------
#              폴더                       파일     행   열   KB
#   coatingvision  coating_labels_tidy.csv  2227  24  512
#   coatingvision        curated_index.csv    54  20   16
#   coatingvision      labels_original.csv  2227   6  161
#             eis  eis_data_dictionary.csv    44   5    4
#             eis             eis_long.csv  3360  11  301
#             eis             eis_wide.csv   240  33  103
#             eis          frequencies.csv    14   2    0
#             eis        impedance_raw.csv  3360   5  183
#             rul    battery_rul_cells.csv 15064  18 1798
#             rul    battery_rul_clean.csv 14996  10 1160
#             rul battery_rul_original.csv 15064   9 1116
#             rul         cell_summary.csv    14  12    1
#             rul      data_dictionary.csv    18   4    2
#           secom          secom_clean.csv  1567 449 4697
#           secom  secom_column_report.csv   590   6   34
#           secom         secom_merged.csv  1567 593 5185
#             wmg      data_dictionary.csv   138   6   19
#             wmg         wmg_asi_long.csv   594   8   36
#             wmg         wmg_cells_54.csv    54  58   34

# --------------------------------------------------------------------
# [이어붙이기 전에 스키마부터 비교하라 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
wmg_paths = sorted(glob.glob("data/wmg/wmg_*.csv"))
tables = {}
for p in wmg_paths:
    tables[os.path.basename(p)] = pd.read_csv(p)

for name, d in tables.items():
    print(f"{name:24s} {d.shape}")

common = set(tables["wmg_cells_54.csv"].columns)
for d in tables.values():
    common = common & set(d.columns)
print("5개 파일 모두에 있는 열:", sorted(common))

# 교재 실행 결과 ------------------------------------------------------
#   wmg_asi_long.csv         (594, 8)
#   wmg_cells_54.csv         (54, 58)
#   wmg_conditions_18.csv    (18, 51)
#   wmg_cycling_long.csv     (2808, 8)
#   wmg_rate_long.csv        (648, 13)
#   5개 파일 모두에 있는 열: ['group']

# --------------------------------------------------------------------
# [이어붙이기 전에 스키마부터 비교하라 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
big = pd.concat(tables.values(), ignore_index=True)
print("무작정 이어붙인 결과:", big.shape)
print("전체 칸 중 결측 비율: %.1f %%"
      % (big.isna().sum().sum() / big.size * 100))

# 교재 실행 결과 ------------------------------------------------------
#   무작정 이어붙인 결과: (4122, 109)
#   전체 칸 중 결측 비율: 91.2 %

# --------------------------------------------------------------------
# [이어붙이기 전에 스키마부터 비교하라 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
cells = tables["wmg_cells_54.csv"]
cyc = tables["wmg_cycling_long.csv"]
merged = pd.merge(cyc, cells[["cell_id", "coat_weight_level", "density_level"]],
                  on="cell_id", how="left")
print(merged.shape)
print(merged.groupby("coat_weight_level")["cap_dis_mah"].mean().round(3))

# 교재 실행 결과 ------------------------------------------------------
#   (2808, 10)
#   coat_weight_level
#   H    4.502
#   L    3.069
#   Name: cap_dis_mah, dtype: float64

# --------------------------------------------------------------------
# [파일명만으로는 구분되지 않는다 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
print(inv["파일"].value_counts().head(3))
print(inv.loc[inv["파일"] == "data_dictionary.csv", ["폴더", "행", "열"]])

# 교재 실행 결과 ------------------------------------------------------
#   파일
#   data_dictionary.csv        2
#   coating_labels_tidy.csv    1
#   curated_index.csv          1
#   Name: count, dtype: int64
#        폴더    행  열
#   12  rul   18  4
#   16  wmg  138  6

