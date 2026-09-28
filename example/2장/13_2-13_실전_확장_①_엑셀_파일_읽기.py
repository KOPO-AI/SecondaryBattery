# -*- coding: utf-8 -*-
"""2장 예제 — 2.13 실전 확장 ① 엑셀 파일 읽기 — 현장 데이터의 실제 모습 (25분) `선택 학습(자율 복습)`

교재 출처 : manuscript/21_ch2_pandas_입출력.md
실행 방법 : example 폴더에서  python "2장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 10_2-10_pandas_기초.py, 11_2-11_SQLite_데이터베이스_연결.py, 12_2-12_연습문제.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --------------------------------------------------------------------
# [read_excel과 openpyxl `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
import pandas as pd

XLSX = "data/wmg/raw/Intermediate measurements during calendering.xlsx"

xls = pd.ExcelFile(XLSX)
print("시트 개수:", len(xls.sheet_names))
print(xls.sheet_names)

# 교재 실행 결과 ------------------------------------------------------
#   시트 개수: 2
#   ['Cathode', 'Cathode-Intermediate']

# --------------------------------------------------------------------
# [첫 줄이 열 이름이 아니다 — header 인자 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
bad = pd.read_excel(XLSX, sheet_name="Cathode")
print(bad.shape)
print(list(bad.columns)[:6])

# 교재 실행 결과 ------------------------------------------------------
#   (40, 36)
#   ['Unnamed: 0', 'Mean values', 'Calendering conditions', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5']

# --------------------------------------------------------------------
# [첫 줄이 열 이름이 아니다 — header 인자 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
grid = pd.read_excel(XLSX, sheet_name="Cathode", header=None)
print("엑셀 격자 그대로:", grid.shape)
print(list(grid.iloc[0, 0:6]))
print(list(grid.iloc[1, 2:6]))

# 교재 실행 결과 ------------------------------------------------------
#   엑셀 격자 그대로: (41, 36)
#   [nan, 'Mean values', 'Calendering conditions', nan, nan, nan]
#   ['Target coating weight (GSM)', 'Roll temperature (oC)', 'Target density (g/cm3)', 'Calculated target porosity (%)']

# --------------------------------------------------------------------
# [첫 줄이 열 이름이 아니다 — header 인자 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
cal = pd.read_excel(XLSX, sheet_name="Cathode", header=1, nrows=18)
print(cal.shape)
print(cal[["No", "Roll temperature (oC)", "Target density (g/cm3)"]].head(3))

# 교재 실행 결과 ------------------------------------------------------
#   (18, 36)
#      No  Roll temperature (oC)  Target density (g/cm3)
#   0   1                     85                    2.70
#   1   2                     85                    2.95
#   2   3                     85                    3.20

# --------------------------------------------------------------------
# [한 시트에 표가 두 개 — nrows 인자 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
print(grid.iloc[18:24, [0, 1]])

# 교재 실행 결과 ------------------------------------------------------
#         0                                           1
#   18   17                         NEX_CAT_240_H_145_M
#   19   18                         NEX_CAT_240_H_145_D
#   20  NaN                                         NaN
#   21  NaN                         Standard Deviations
#   22   No  Electrode ID (P-porous, M-medium, D-dense)
#   23    1                          NEX_CAT_240_L_85_P

# --------------------------------------------------------------------
# [한 시트에 표가 두 개 — nrows 인자 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
print(cal["Roll temperature (oC)"].value_counts().sort_index())
print(cal["Target coating weight (GSM)"].value_counts())

# 교재 실행 결과 ------------------------------------------------------
#   Roll temperature (oC)
#   85     6
#   120    6
#   145    6
#   Name: count, dtype: int64
#   Target coating weight (GSM)
#   122.48    9
#   182.73    9
#   Name: count, dtype: int64

# --------------------------------------------------------------------
# [시트가 19장인 워크북 — 행 번호를 하드코딩하지 마라 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
XL2 = "data/wmg/raw/Half-cell (Cathode) Electrochemical Performance.xlsx"
book = pd.ExcelFile(XL2)
print("시트 개수:", len(book.sheet_names))
print(book.sheet_names[:5], "...")
for name in ["Group1", "Group13", "Group14"]:
    sh = pd.read_excel(XL2, sheet_name=name, header=None)
    print(name, sh.shape)

# 교재 실행 결과 ------------------------------------------------------
#   시트 개수: 19
#   ['Table', 'Group1', 'Group2', 'Group3', 'Group4'] ...
#   Group1 (151, 7)
#   Group13 (151, 7)
#   Group14 (159, 7)

# --------------------------------------------------------------------
# [시트가 19장인 워크북 — 행 번호를 하드코딩하지 마라 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
for name in ["Group1", "Group14"]:
    sh = pd.read_excel(XL2, sheet_name=name, header=None)
    print(name, "44행 →", sh.iloc[44, 1], "/ 첫 셀 값", round(sh.iloc[44, 2], 3))

# 교재 실행 결과 ------------------------------------------------------
#   Group1 44행 → At C/20 / 첫 셀 값 3.266
#   Group14 44행 → At C/5- 4 / 첫 셀 값 4.849

# --------------------------------------------------------------------
# [남이 계산해 준 평균을 믿지 마라 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
g1 = pd.read_excel(XL2, sheet_name="Group1", header=None)
print(g1.iloc[0, 1:5].tolist())
row = g1.iloc[80]
print(row.iloc[0], "|", row.iloc[1])
print("DD001:", row.iloc[2], "/ DD002:", round(row.iloc[3], 2),
      "/ DD056:", round(row.iloc[4], 2))
print("엑셀 Mean:", round(row.iloc[5], 2), "/ 엑셀 SD:", round(row.iloc[6], 2))

# 교재 실행 결과 ------------------------------------------------------
#   ['Cathode cell ID', 'DD001', 'DD002', 'DD056']
#   Gravimetric Charge Capacity (mAh/g) | At C/20
#   DD001: 0 / DD002: 182.91 / DD056: 183.09
#   엑셀 Mean: 122.0 / 엑셀 SD: 86.27

# --------------------------------------------------------------------
# [남이 계산해 준 평균을 믿지 마라 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
print("빈칸 대신 0이 들어간 DD001을 빼고 다시 평균:",
      round((row.iloc[3] + row.iloc[4]) / 2, 2))

# 교재 실행 결과 ------------------------------------------------------
#   빈칸 대신 0이 들어간 DD001을 빼고 다시 평균: 183.0

