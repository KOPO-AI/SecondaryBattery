# -*- coding: utf-8 -*-
"""4장 예제 — 4.3 선형 회귀 — 가장 단순하고 가장 해석하기 좋은 모델

교재 출처 : manuscript/40_ch4_회귀.md
실행 방법 : example 폴더에서  python "4장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 01_4-1_회귀_문제의_정의.py, 02_4-2_학습테스트_분할.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 4.1~4.2절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
# ── 4.1.2절: 한글 폰트, 정제 규약 clean_data()와 X, y ──
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
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [sklearn 실습]
# --------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

lr = LinearRegression()
lr.fit(X_train, y_train)          # 학습: 계수 추정

y_pred = lr.predict(X_test)       # 테스트 데이터 예측

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"테스트 R² = {r2:.4f}, RMSE = {rmse:.2f} mAh")
print(f"학습   R² = {lr.score(X_train, y_train):.4f}")

# --------------------------------------------------------------------
# [계수 해석 — Gap 1 µm당 용량이 얼마나 변하는가]
# --------------------------------------------------------------------
coef_table = pd.DataFrame({
    '공정 인자': FEATS,
    '회귀계수': lr.coef_
})
print(f"절편: {lr.intercept_:.2f}")
print(coef_table)

# --------------------------------------------------------------------
# [잔차 분석 기초]
# --------------------------------------------------------------------
# 한글 폰트 설정은 4.1.2절 첫 셀에서 이미 잡아 두었다.
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# (1) 예측값 vs 잔차
axes[0].scatter(y_pred, residuals, alpha=0.5)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('예측 용량 (mAh)'); axes[0].set_ylabel('잔차 (mAh)')
# (2) 잔차 히스토그램
axes[1].hist(residuals, bins=30)
axes[1].set_xlabel('잔차 (mAh)'); axes[1].set_ylabel('빈도')
plt.tight_layout(); plt.show()

