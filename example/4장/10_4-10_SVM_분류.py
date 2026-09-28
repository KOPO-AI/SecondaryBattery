# -*- coding: utf-8 -*-
"""4장 예제 — 4.10 SVM 분류 — 여유 폭을 최대로 하는 경계선 〔선택 학습(자율 복습)〕

교재 출처 : manuscript/41_ch4_분류.md
실행 방법 : example 폴더에서  python "4장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 07_4-7_종합_회귀_모델_구축_파이프라인.py, 08_4-8_분류_문제의_정의.py, 09_4-9_로지스틱_회귀.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 4.1·4.8·4.9절에서 만든 객체를 이 파일만으로 재현한 것이다.
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

# ── 4.9.3절: 성능 출력 함수 report() ──
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def report(name, y_true, y_pred):
    print(f"{name:28s} 정확도={accuracy_score(y_true, y_pred):.3f} "
          f"정밀도={precision_score(y_true, y_pred, zero_division=0):.3f} "
          f"재현율={recall_score(y_true, y_pred):.3f} "
          f"F1={f1_score(y_true, y_pred):.3f}")
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [실습]
# --------------------------------------------------------------------
from sklearn.svm import SVC

svm_rbf = SVC(kernel="rbf", random_state=42)
svm_rbf.fit(X_train_s, y_train)
report("SVM RBF(기본)", y_test, svm_rbf.predict(X_test_s))

svm_bal = SVC(kernel="rbf", class_weight="balanced", random_state=42)
svm_bal.fit(X_train_s, y_train)
report("SVM RBF(balanced)", y_test, svm_bal.predict(X_test_s))

# 교재 실행 결과 ------------------------------------------------------
#   SVM RBF(기본)                  정확도=0.980 정밀도=0.000 재현율=0.000 F1=0.000
#   SVM RBF(balanced)            정확도=0.972 정밀도=0.375 재현율=0.600 F1=0.462

