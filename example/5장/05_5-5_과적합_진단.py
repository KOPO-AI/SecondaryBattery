# -*- coding: utf-8 -*-
"""5장 예제 — 5.5 과적합 진단 — 학습곡선 읽는 법 `필수`

교재 출처 : manuscript/50_ch5_평가.md
실행 방법 : example 폴더에서  python "5장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 02_5-2_회귀_평가지표.py, 03_5-3_분류_평가지표.py, 04_5-4_검증_전략.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import numpy as np

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 5.1절·5.2절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
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


df = clean_data('data/battery_process_data.csv')

FEATS = ['믹싱_RPM', '믹싱_온도', '코팅_토출압력',
         '건조로_1구간_온도', '프레스_압력', '프레스_Gap']
X = df[FEATS]
y_reg = df['최종_용량_mAh']   # 회귀 타깃

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [실습 — 세 모델의 학습곡선 `필수`]
# --------------------------------------------------------------------
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeRegressor

for name, est in [('깊은 결정트리', DecisionTreeRegressor(random_state=42)),
                  ('랜덤 포레스트', RandomForestRegressor(n_estimators=200,
                                                          random_state=42)),
                  ('선형회귀',      LinearRegression())]:
    sizes, tr_sc, va_sc = learning_curve(
        est, X, y_reg, cv=5, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=True, random_state=42)
    print(name)
    print("  훈련량   :", sizes.tolist())
    print("  훈련 R2  :", np.round(tr_sc.mean(axis=1), 4).tolist())
    print("  검증 R2  :", np.round(va_sc.mean(axis=1), 4).tolist())

# 교재 실행 결과 ------------------------------------------------------
#   깊은 결정트리
#     훈련량   : [80, 260, 440, 620, 800]
#     훈련 R2  : [1.0, 1.0, 1.0, 1.0, 1.0]
#     검증 R2  : [0.4278, 0.5746, 0.5911, 0.5818, 0.5637]
#   랜덤 포레스트
#     훈련량   : [80, 260, 440, 620, 800]
#     훈련 R2  : [0.9581, 0.9675, 0.9697, 0.9698, 0.9701]
#     검증 R2  : [0.7037, 0.7613, 0.7637, 0.7716, 0.7747]
#   선형회귀
#     훈련량   : [80, 260, 440, 620, 800]
#     훈련 R2  : [0.8178, 0.8203, 0.8185, 0.8147, 0.8125]
#     검증 R2  : [0.7985, 0.8071, 0.8077, 0.8081, 0.8075]

