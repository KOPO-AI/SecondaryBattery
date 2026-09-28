# -*- coding: utf-8 -*-
"""5장 예제 — 5.2 회귀 평가지표 — 예측이 얼마나 빗나갔는가 `필수`

교재 출처 : manuscript/50_ch5_평가.md
실행 방법 : example 폴더에서  python "5장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 01_5-1_왜_평가가_모델링보다_중요한가.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import numpy as np

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 5.1절에서 만든 객체를 이 파일만으로 재현한 것이다.
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
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [실습 — 4장 회귀 3모델 전수 평가 `필수`]
# --------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)

models = {
    '선형회귀':      LinearRegression(),
    '랜덤 포레스트': RandomForestRegressor(n_estimators=200, random_state=42),
    'XGBoost':      XGBRegressor(n_estimators=200, learning_rate=0.1,
                                 max_depth=4, random_state=42),
}

print(f"{'모델':<12}{'MSE':>10}{'RMSE':>8}{'MAE':>8}{'R2':>9}{'MAPE(%)':>9}")
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    mse = mean_squared_error(y_test, p)
    print(f"{name:<12}{mse:>10.2f}{np.sqrt(mse):>8.2f}"
          f"{mean_absolute_error(y_test, p):>8.2f}"
          f"{r2_score(y_test, p):>9.4f}"
          f"{mean_absolute_percentage_error(y_test, p)*100:>9.3f}")

# 바닥선: 학습셋 평균으로만 예측
p0 = np.full(len(y_test), y_train.mean())
print(f"평균 예측    RMSE={np.sqrt(mean_squared_error(y_test, p0)):.2f} "
      f"MAE={mean_absolute_error(y_test, p0):.2f} "
      f"R2={r2_score(y_test, p0):.4f}")

# 교재 실행 결과 ------------------------------------------------------
#   모델                 MSE    RMSE     MAE       R2  MAPE(%)
#   선형회귀            239.48   15.47   12.33   0.7692    0.342
#   랜덤 포레스트         255.68   15.99   12.87   0.7535    0.358
#   XGBoost         258.43   16.08   13.00   0.7509    0.361
#   평균 예측    RMSE=32.29 MAE=25.83 R2=-0.0052

