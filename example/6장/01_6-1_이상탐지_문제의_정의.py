# -*- coding: utf-8 -*-
"""6장 예제 — 6.1 이상탐지 문제의 정의 `필수`

교재 출처 : manuscript/60_ch6_이상탐지.md
실행 방법 : example 폴더에서  python "6장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --------------------------------------------------------------------
# [실습 데이터 준비 — 표준 정제 규약 + 이 장만의 추가 조치]
# --------------------------------------------------------------------
import pandas as pd
import numpy as np


def clean_data(path):
    """배터리 공정 데이터 정제 파이프라인 (3.7.1절과 동일)"""
    df = pd.read_csv(path)
    # 1) 에러 코드 -> NaN
    df['건조로_1구간_온도'] = df['건조로_1구간_온도'].replace(9999, np.nan)
    # 2) 물리 범위 위반 -> NaN
    rules = {'믹싱_온도': (10, 50), '건조로_1구간_온도': (80, 160),
             '프레스_압력': (20, 80)}
    for col, (lo, hi) in rules.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    # 3) 결측치 -> 중앙값 대체
    num_cols = df.select_dtypes('number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # 4) 중복 제거
    df = df.drop_duplicates(subset='Lot_ID')
    return df


raw = pd.read_csv("data/battery_process_data.csv", encoding="utf-8-sig")
err_lots = raw.loc[raw["건조로_1구간_온도"] >= 9999, "Lot_ID"].tolist()
print("센서 오류(9999) Lot:", err_lots)

df = clean_data("data/battery_process_data.csv")          # 표준 정제 규약
print("clean_data() 직후:", df.shape)

# 이 장만의 추가 조치: 온도가 대체값인 Lot을 이상탐지 입력에서 제외
df = df[~df["Lot_ID"].isin(err_lots)].reset_index(drop=True)
print("센서 오류 Lot 제외 후:", df.shape, "/ 불량 Lot 수:", df["불량_여부"].sum())

# 교재 실행 결과 ------------------------------------------------------
#   센서 오류(9999) Lot: ['L25155', 'L25161', 'L25230', 'L25654', 'L25798']
#   clean_data() 직후: (1000, 10)
#   센서 오류 Lot 제외 후: (995, 10) / 불량 Lot 수: 19

