# -*- coding: utf-8 -*-
"""4장 예제 — 4.13 확률 예측과 임계값 — 판정선은 비즈니스가 정한다

교재 출처 : manuscript/41_ch4_분류.md
실행 방법 : example 폴더에서  python "4장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 10_4-10_SVM_분류.py, 11_4-11_트리_기반_분류.py, 12_4-12_클래스_불균형_문제.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 4.1·4.8·4.9·4.12절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
# ── 4.1.2절: 정제 규약 clean_data() ──
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


# ── 4.8.3절: 층화 분할과 표준화 ──
df = clean_data("data/battery_process_data.csv")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 4.1.2절에서 정의한 것과 같은 6개 공정 변수 리스트, 같은 이름(FEATS)을 쓴다
FEATS = ["믹싱_RPM", "믹싱_온도", "코팅_토출압력",
         "건조로_1구간_온도", "프레스_압력", "프레스_Gap"]
X = df[FEATS]
y = df["불량_여부"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

# 로지스틱 회귀·SVM용 표준화 (트리 계열은 원본 사용)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 4.9.3절: 로지스틱 회귀·평가 지표 import ──
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── 4.12.2절: class_weight='balanced' 로지스틱 회귀 logreg_bal ──
logreg_bal = LogisticRegression(max_iter=1000, class_weight="balanced",
                                random_state=42)
logreg_bal.fit(X_train_s, y_train)
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [predict가 아니라 predict_proba를 보라]
# --------------------------------------------------------------------
proba = logreg_bal.predict_proba(X_test_s)[:, 1]
print(pd.Series(proba).describe().round(3))

# --------------------------------------------------------------------
# [임계값을 움직이면 무슨 일이 벌어지는가]
# --------------------------------------------------------------------
print(" 임계값  판정건수  정밀도  재현율    F1")
for t in [0.3, 0.5, 0.7, 0.8, 0.9]:
    pred_t = (proba >= t).astype(int)
    print(f"  {t:.1f}    {pred_t.sum():4d}   "
          f"{precision_score(y_test, pred_t, zero_division=0):.3f}  "
          f"{recall_score(y_test, pred_t):.3f}  "
          f"{f1_score(y_test, pred_t):.3f}")

