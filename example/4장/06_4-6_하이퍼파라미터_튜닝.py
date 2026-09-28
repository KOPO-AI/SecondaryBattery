# -*- coding: utf-8 -*-
"""4장 예제 — 4.6 하이퍼파라미터 튜닝 — 모델의 손잡이를 체계적으로 돌린다

교재 출처 : manuscript/40_ch4_회귀.md
실행 방법 : example 폴더에서  python "4장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 03_4-3_선형_회귀.py, 04_4-4_트리_기반_회귀.py, 05_4-5_교차검증.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 4.1~4.4절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
# ── 4.1.2절: 정제 규약 clean_data()와 X, y ──
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
y = df['최종_용량_mAh']

# ── 4.2.1절: 학습/테스트 분할 ──
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20%를 테스트용으로
    random_state=42     # 재현성을 위한 난수 시드 고정
)

# ── 4.3.2절·4.4.2절: 평가 지표와 모델 클래스 import ──
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [GridSearchCV 실습]
# --------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 3, 5],
}   # 3×3×3 = 27조합 × 5-fold = 135회 학습

grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid, cv=5, scoring='r2', n_jobs=-1
)
grid.fit(X_train, y_train)   # 학습 데이터만 사용!

print("최적 조합:", grid.best_params_)
print(f"교차검증 최고 R² = {grid.best_score_:.4f}")

best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test)
print(f"테스트 R² = {r2_score(y_test, y_pred_best):.4f}")

# --------------------------------------------------------------------
# [RandomizedSearchCV — 넓은 공간을 효율적으로 〔선택 학습(자율 복습)〕]
# --------------------------------------------------------------------
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 15),
    'min_samples_leaf': randint(1, 10),
}
rand = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_dist, n_iter=20, cv=5, scoring='r2',
    random_state=42, n_jobs=-1
)
rand.fit(X_train, y_train)
print("최적 조합:", rand.best_params_)
print(f"테스트 R² = {r2_score(y_test, rand.best_estimator_.predict(X_test)):.4f}")

