# -*- coding: utf-8 -*-
"""5장 예제 — 5.13 모델 저장과 활용 `필수`

교재 출처 : manuscript/51_ch5_중요도.md
실행 방법 : example 폴더에서  python "5장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 09_5-10_순열_중요도.py, 10_5-11_SHAP_기여도의_공정한_배분.py, 11_5-12_핵심_공정_인자_도출_실습.py
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
# [joblib으로 저장하고 불러오기]
# --------------------------------------------------------------------
import joblib

joblib.dump(reg, "rf_capacity.joblib")   # 용량 회귀 모델
joblib.dump(clf, "rf_defect.joblib")     # 불량 분류 모델

# 불러오기 — 학습 코드 없이 즉시 사용 가능
clf_loaded = joblib.load("rf_defect.joblib")

# --------------------------------------------------------------------
# [새 로트 예측 함수 만들기 `필수`]
# --------------------------------------------------------------------
# FEATS는 5.8.3절에서 정의한 것과 동일하다. 배포 스크립트에서는
# 함수와 함께 이 상수도 파일 안에 다시 명시해 두는 것이 안전하다.
FEATS = ["믹싱_RPM", "믹싱_온도", "코팅_토출압력",
         "건조로_1구간_온도", "프레스_압력", "프레스_Gap"]


def predict_new_lot(lot_dict, model_path="rf_defect.joblib",
                    threshold=0.5):
    """새 로트의 공정 조건 딕셔너리를 받아 불량 위험을 판정한다."""
    model = joblib.load(model_path)
    x = pd.DataFrame([lot_dict])[FEATS]      # 순서 강제 + 누락 시 KeyError
    p = model.predict_proba(x)[0, 1]
    return {"불량확률": round(float(p), 3),
            "판정": "고위험-검사강화" if p >= threshold else "정상"}

new_lot = {"믹싱_RPM": 1750, "믹싱_온도": 25.0, "코팅_토출압력": 121.0,
           "건조로_1구간_온도": 110.5, "프레스_압력": 40.0, "프레스_Gap": 95.0}
print(predict_new_lot(new_lot))

# 교재 실행 결과 ------------------------------------------------------
#   {'불량확률': 0.01, '판정': '정상'}

