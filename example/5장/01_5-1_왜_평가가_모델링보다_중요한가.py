# -*- coding: utf-8 -*-
"""5장 예제 — 5.1 왜 평가가 모델링보다 중요한가 `필수`

교재 출처 : manuscript/50_ch5_평가.md
실행 방법 : example 폴더에서  python "5장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --------------------------------------------------------------------
# [실습 환경 준비 — 3.7.1절 `clean_data()` 규약 적용]
# --------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트 (macOS는 'AppleGothic')
plt.rcParams['axes.unicode_minus'] = False      # 마이너스 부호 깨짐 방지


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


df = clean_data('data/battery_process_data.csv')
print("정제 후:", df.shape, "| 결측 셀:", df.isna().sum().sum())
print("불량 분포:", df['불량_여부'].value_counts().to_dict())

FEATS = ['믹싱_RPM', '믹싱_온도', '코팅_토출압력',
         '건조로_1구간_온도', '프레스_압력', '프레스_Gap']
X = df[FEATS]
y_reg = df['최종_용량_mAh']   # 회귀 타깃
y_clf = df['불량_여부']        # 분류 타깃

# 교재 실행 결과 ------------------------------------------------------
#   정제 후: (1000, 10) | 결측 셀: 0
#   불량 분포: {0: 981, 1: 19}

