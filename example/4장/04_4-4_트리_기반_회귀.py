# -*- coding: utf-8 -*-
"""4장 예제 — 4.4 트리 기반 회귀 — 비선형을 잡는 세 가지 무기

교재 출처 : manuscript/40_ch4_회귀.md
실행 방법 : example 폴더에서  python "4장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 01_4-1_회귀_문제의_정의.py, 02_4-2_학습테스트_분할.py, 03_4-3_선형_회귀.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 4.1~4.3절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
# ── 4.1.2절: 정제 규약 clean_data()와 X, y ──
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

# ── 4.3.2절: 평가 지표 함수 ──
from sklearn.metrics import r2_score, mean_squared_error
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [결정 트리 — 스무고개식 분할]
# --------------------------------------------------------------------
from sklearn.tree import DecisionTreeRegressor

# 깊이 제한 없는 트리
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
print(f"학습 R² = {dt.score(X_train, y_train):.4f}")
print(f"테스트 R² = {r2_score(y_test, dt.predict(X_test)):.4f}")

# 깊이를 5로 제한한 트리
dt5 = DecisionTreeRegressor(max_depth=5, random_state=42)
dt5.fit(X_train, y_train)
print(f"깊이 5 테스트 R² = {r2_score(y_test, dt5.predict(X_test)):.4f}")

# --------------------------------------------------------------------
# [랜덤 포레스트 — 배깅으로 분산을 줄인다]
# --------------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,     # 트리 200그루
    oob_score=True,       # OOB 점수 계산
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print(f"테스트 R² = {r2_score(y_test, y_pred_rf):.4f}")
print(f"OOB R²   = {rf.oob_score_:.4f}")

# 특성 중요도
imp = pd.Series(rf.feature_importances_, index=FEATS).sort_values(ascending=False)
print(imp)

# --------------------------------------------------------------------
# [XGBoost — 부스팅으로 오차를 순차 보정한다]
# --------------------------------------------------------------------
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(f"테스트 R² = {r2_score(y_test, y_pred_xgb):.4f}")

