# -*- coding: utf-8 -*-
"""6장 예제 — 6.9 원인 추적(Root Cause) 기법 `필수`

교재 출처 : manuscript/61_ch6_최적화.md
실행 방법 : example 폴더에서  python "6장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 05_6-5_검출_결과의_검증.py, 06_6_예제.py, 07_6-8_불량_발생_패턴_분석.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd
import numpy as np

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 6장 후반부 도입 셀·6.8절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
# (6장 후반부 도입 셀, 6.1.4절과 동일) 표준 정제 규약 clean_data() + 센서 오류(9999) Lot 5건 제외 → df 995 Lot
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


raw = pd.read_csv("data/battery_process_data.csv", encoding="utf-8-sig")
err_lots = raw.loc[raw["건조로_1구간_온도"] >= 9999, "Lot_ID"].tolist()

df = clean_data("data/battery_process_data.csv")          # 표준 정제 규약

# 6장 공통의 추가 조치: 온도가 대체값인 Lot을 제외(6.1.4절과 동일)
df = df[~df["Lot_ID"].isin(err_lots)].reset_index(drop=True)

# (6.8.2절) 공정 변수 6개
FEATS = ["믹싱_RPM", "믹싱_온도", "코팅_토출압력",
         "건조로_1구간_온도", "프레스_압력", "프레스_Gap"]

# (6.8.2절) 정상·불량 그룹
normal = df[df["불량_여부"] == 0]
defect = df[df["불량_여부"] == 1]
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [변수별 z-score 기여도]
# --------------------------------------------------------------------
mu = normal[FEATS].mean()
sigma = normal[FEATS].std()
Z = (df[FEATS] - mu) / sigma

# 불량 Lot 전체의 변수별 평균 |z| — 집단 수준의 기여도
contrib = Z.loc[defect.index].abs().mean().sort_values(ascending=False)
print(contrib.round(2))

# 교재 실행 결과 ------------------------------------------------------
#   프레스_Gap       2.19
#   믹싱_RPM        1.11
#   코팅_토출압력       1.03
#   믹싱_온도         0.84
#   프레스_압력        0.82
#   건조로_1구간_온도    0.59
#   dtype: float64

# --------------------------------------------------------------------
# [이상 Lot 1건 심층 분석 워크스루]
# --------------------------------------------------------------------
z_defect = Z.loc[defect.index]
worst_idx = z_defect.abs().max(axis=1).idxmax()
print("심층 분석 대상:", df.loc[worst_idx, "Lot_ID"])
print(df.loc[worst_idx])
print("\n변수별 z-score:")
print(z_defect.loc[worst_idx].round(2).sort_values())

# 교재 실행 결과 ------------------------------------------------------
#   심층 분석 대상: L25949
#   Lot_ID            L25949
#   믹싱_RPM              1458
#   믹싱_온도               28.0
#   코팅_토출압력            129.4
#   건조로_1구간_온도         109.8
#   프레스_압력              43.4
#   프레스_Gap            114.2
#   최종_용량_mAh         3527.1
#   전극_면저항_mOhmcm2    1795.5
#   불량_여부                  1
#   Name: 944, dtype: object
#   변수별 z-score:
#   믹싱_RPM       -3.71
#   건조로_1구간_온도   -0.05
#   프레스_압력        0.88
#   코팅_토출압력       1.12
#   믹싱_온도         2.06
#   프레스_Gap       3.26
#   Name: 944, dtype: float64

# --------------------------------------------------------------------
# [SHAP 개별 설명으로 교차 검증 (5장 연계)]
# --------------------------------------------------------------------
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = df[FEATS], df["불량_여부"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
# roc_auc_score는 파이썬 float를 반환하므로 .round()가 아니라 내장 round()를 쓴다
print("검증 AUC:", round(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]), 3))

explainer = shap.TreeExplainer(clf)
sv = explainer.shap_values(df.loc[[worst_idx], FEATS])
shap_lot = pd.Series(np.array(sv[..., 1]).ravel(), index=FEATS)
print("기저 불량 확률:", explainer.expected_value[1].round(3))
print("예측 불량 확률:", clf.predict_proba(df.loc[[worst_idx], FEATS])[0, 1].round(3))
print(shap_lot.round(3).sort_values(ascending=False))

# 교재 실행 결과 ------------------------------------------------------
#   검증 AUC: 0.826
#   기저 불량 확률: 0.019
#   예측 불량 확률: 0.773
#   프레스_Gap       0.410
#   믹싱_RPM        0.221
#   프레스_압력        0.049
#   믹싱_온도         0.041
#   건조로_1구간_온도    0.018
#   코팅_토출압력       0.016
#   dtype: float64

