# -*- coding: utf-8 -*-
"""5장 예제 — 5.10 순열 중요도 `필수`

교재 출처 : manuscript/51_ch5_중요도.md
실행 방법 : example 폴더에서  python "5장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 06_5-6_모델_성능_비교_실험.py, 07_5-8_왜_설명가능성이_필요한가.py, 08_5-9_불순도_기반_중요도.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 5.8절에서 만든 객체를 이 파일만으로 재현한 것이다.
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


df = clean_data("data/battery_process_data.csv")

FEATS = ["믹싱_RPM", "믹싱_온도", "코팅_토출압력",
         "건조로_1구간_온도", "프레스_압력", "프레스_Gap"]
X = df[FEATS]
y_cap = df["최종_용량_mAh"]     # 회귀 타깃
y_def = df["불량_여부"]          # 분류 타깃 (불량률 1.9%, 19/1,000)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# 회귀: 최종 용량 예측
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cap, test_size=0.2, random_state=42)
reg = RandomForestRegressor(n_estimators=300, random_state=42)
reg.fit(X_train, y_train)

# 분류: 불량 예측 (불균형 데이터이므로 층화 분할 + class_weight)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_def, test_size=0.2, random_state=42, stratify=y_def)
clf = RandomForestClassifier(n_estimators=300, random_state=42,
                             class_weight="balanced")
clf.fit(X_train_clf, y_train_clf)
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [실습: permutation_importance `필수`]
# --------------------------------------------------------------------
from sklearn.inspection import permutation_importance

pi_reg = permutation_importance(reg, X_test, y_test, scoring="r2",
                                n_repeats=30, random_state=42)
pi_tab = pd.DataFrame({"중요도(R2 하락)": pi_reg.importances_mean,
                       "표준편차": pi_reg.importances_std}, index=FEATS)
print(pi_tab.sort_values("중요도(R2 하락)", ascending=False).round(4))

# 교재 실행 결과 ------------------------------------------------------
#               중요도(R2 하락)    표준편차
#   프레스_Gap         1.5090  0.1217
#   코팅_토출압력         0.1038  0.0197
#   믹싱_온도           0.0021  0.0041
#   건조로_1구간_온도      0.0008  0.0036
#   프레스_압력          0.0006  0.0041
#   믹싱_RPM         -0.0014  0.0040

# --------------------------------------------------------------------
# [실습: permutation_importance `필수`]
# --------------------------------------------------------------------
pi_clf = permutation_importance(clf, X_test_clf, y_test_clf, scoring="roc_auc",
                                n_repeats=30, random_state=42)
pi_tab2 = pd.DataFrame({"중요도(AUC 하락)": pi_clf.importances_mean,
                        "표준편차": pi_clf.importances_std}, index=FEATS)
print(pi_tab2.sort_values("중요도(AUC 하락)", ascending=False).round(4))

# 교재 실행 결과 ------------------------------------------------------
#               중요도(AUC 하락)    표준편차
#   프레스_Gap          0.5036  0.1455
#   건조로_1구간_온도       0.0056  0.0072
#   믹싱_온도            0.0036  0.0021
#   프레스_압력           0.0000  0.0033
#   코팅_토출압력         -0.0022  0.0034
#   믹싱_RPM          -0.0027  0.0022

