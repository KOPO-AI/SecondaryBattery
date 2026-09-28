# -*- coding: utf-8 -*-
"""4장 예제 — 4.5 교차검증 — 단 한 번의 분할을 믿지 마라

교재 출처 : manuscript/40_ch4_회귀.md
실행 방법 : example 폴더에서  python "4장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 02_4-2_학습테스트_분할.py, 03_4-3_선형_회귀.py, 04_4-4_트리_기반_회귀.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 4.1·4.3·4.4절에서 만든 객체를 이 파일만으로 재현한 것이다.
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

# ── 4.3.2절·4.4.2절: 모델 클래스 import ──
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [cross_val_score 실습]
# --------------------------------------------------------------------
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_lr = cross_val_score(LinearRegression(), X, y, cv=kf, scoring='r2')
cv_rf = cross_val_score(
    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    X, y, cv=kf, scoring='r2')

print(f"선형회귀 fold별: {cv_lr.round(3)}")
print(f"선형회귀 평균 {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")
print(f"랜덤포레스트 평균 {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

