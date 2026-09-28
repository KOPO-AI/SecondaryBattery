# -*- coding: utf-8 -*-
"""4장 예제 — 4.7 종합: 회귀 모델 구축 파이프라인

교재 출처 : manuscript/40_ch4_회귀.md
실행 방법 : example 폴더에서  python "4장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 04_4-4_트리_기반_회귀.py, 05_4-5_교차검증.py, 06_4-6_하이퍼파라미터_튜닝.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 4.1절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
# ── 4.1.2절: 정제 규약 clean_data() ──
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
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [전체 코드 완성본 (배포 스크립트 실행·해설)]
# --------------------------------------------------------------------
"""4장 종합: 배터리 용량 예측 회귀 파이프라인"""
import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, KFold,
                                     cross_val_score, GridSearchCV)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

RANDOM_STATE = 42

# ── 1. 데이터 로드 및 정제 (3.7.1절 clean_data 규약) ─────
# clean_data() 정의는 4.1.2절과 완전히 같으므로 지면에서는 생략했다.
# 배포 스크립트에는 이 함수 정의가 파일 맨 위에 그대로 들어 있다.
df = clean_data('data/battery_process_data.csv')   # 정제 후 1,000행

FEATS = ['믹싱_RPM', '믹싱_온도', '코팅_토출압력',
         '건조로_1구간_온도', '프레스_압력', '프레스_Gap']
X, y = df[FEATS], df['최종_용량_mAh']

# ── 2. 학습/테스트 분할 ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)

# ── 3. 후보 모델 정의 ─────────────────────────────────
models = {
    '선형회귀': LinearRegression(),
    '랜덤포레스트': RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        random_state=RANDOM_STATE),
}

# ── 4. 교차검증으로 후보 비교 (학습 데이터 내부) ─────────
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    print(f"{name:8s} CV R² = {cv.mean():.4f} ± {cv.std():.4f}")

# ── 5. 유망 모델(RF) 하이퍼파라미터 튜닝 ────────────────
grid = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, None],
     'min_samples_leaf': [1, 3, 5]},
    cv=5, scoring='r2', n_jobs=-1).fit(X_train, y_train)
models['RF(튜닝)'] = grid.best_estimator_

# ── 6. 최종 평가: 테스트 데이터로 단 한 번 ──────────────
print("\n=== 최종 테스트 성능 ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name:8s} R² = {r2:.4f}, RMSE = {rmse:.2f} mAh")

# 교재 실행 결과 ------------------------------------------------------
#   선형회귀     CV R² = 0.8148 ± 0.0273
#   랜덤포레스트   CV R² = 0.7799 ± 0.0382
#   XGBoost  CV R² = 0.7738 ± 0.0333
#   === 최종 테스트 성능 ===
#   선형회귀     R² = 0.7692, RMSE = 15.47 mAh
#   랜덤포레스트   R² = 0.7535, RMSE = 15.99 mAh
#   XGBoost  R² = 0.7557, RMSE = 15.92 mAh
#   RF(튜닝)   R² = 0.7673, RMSE = 15.54 mAh

