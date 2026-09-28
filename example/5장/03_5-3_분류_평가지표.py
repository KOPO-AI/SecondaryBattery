# -*- coding: utf-8 -*-
"""5장 예제 — 5.3 분류 평가지표 — 불량 판정 모델의 성적표 `필수`

교재 출처 : manuscript/50_ch5_평가.md
실행 방법 : example 폴더에서  python "5장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 01_5-1_왜_평가가_모델링보다_중요한가.py, 02_5-2_회귀_평가지표.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 5.1절·5.2절에서 만든 객체를 이 파일만으로 재현한 것이다.
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


df = clean_data('data/battery_process_data.csv')

FEATS = ['믹싱_RPM', '믹싱_온도', '코팅_토출압력',
         '건조로_1구간_온도', '프레스_압력', '프레스_Gap']
X = df[FEATS]
y_clf = df['불량_여부']        # 분류 타깃

from sklearn.model_selection import train_test_split
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [정확도의 함정 — 5.1절 사고의 재현]
# --------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             average_precision_score)

# 회귀용 X_train/y_train과 겹치지 않도록 _clf 접미사를 붙인다
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
print("학습 불량:", y_train_clf.sum(), "/", len(y_train_clf),
      "| 테스트 불량:", y_test_clf.sum(), "/", len(y_test_clf))

clfs = {
    '로지스틱':      make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, random_state=42)),
    '랜덤 포레스트': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost':      XGBClassifier(n_estimators=200, learning_rate=0.1,
                        max_depth=4, random_state=42, eval_metric='logloss'),
}
for name, m in clfs.items():
    m.fit(X_train_clf, y_train_clf)
    p  = m.predict(X_test_clf)
    pr = m.predict_proba(X_test_clf)[:, 1]
    print(name, "혼동행렬:", confusion_matrix(y_test_clf, p).tolist(),
          f"Acc={accuracy_score(y_test_clf, p):.4f}",
          f"P={precision_score(y_test_clf, p, zero_division=0):.4f}",
          f"R={recall_score(y_test_clf, p, zero_division=0):.4f}",
          f"F1={f1_score(y_test_clf, p, zero_division=0):.4f}",
          f"ROC-AUC={roc_auc_score(y_test_clf, pr):.4f}",
          f"PR-AUC={average_precision_score(y_test_clf, pr):.4f}")

print("전부 양품 판정 시 Acc =", round((y_test_clf == 0).mean(), 4))

# 교재 실행 결과 ------------------------------------------------------
#   학습 불량: 15 / 800 | 테스트 불량: 4 / 200
#   로지스틱 혼동행렬: [[193, 3], [3, 1]] Acc=0.9700 P=0.2500 R=0.2500 F1=0.2500 ROC-AUC=0.9796 PR-AUC=0.5242
#   랜덤 포레스트 혼동행렬: [[193, 3], [3, 1]] Acc=0.9700 P=0.2500 R=0.2500 F1=0.2500 ROC-AUC=0.9707 PR-AUC=0.3681
#   XGBoost 혼동행렬: [[192, 4], [2, 2]] Acc=0.9700 P=0.3333 R=0.5000 F1=0.4000 ROC-AUC=0.9668 PR-AUC=0.3381
#   전부 양품 판정 시 Acc = 0.98

# --------------------------------------------------------------------
# [임계값 이동 — 같은 모델, 다른 성적 `필수`]
# --------------------------------------------------------------------
m = clfs['랜덤 포레스트']
pr = m.predict_proba(X_test_clf)[:, 1]
for t in (0.5, 0.3, 0.2):
    p = (pr >= t).astype(int)
    print(f"임계값 {t}: P={precision_score(y_test_clf, p, zero_division=0):.3f} "
          f"R={recall_score(y_test_clf, p, zero_division=0):.3f} "
          f"F1={f1_score(y_test_clf, p, zero_division=0):.3f} "
          f"CM={confusion_matrix(y_test_clf, p).tolist()}")

# 교재 실행 결과 ------------------------------------------------------
#   임계값 0.5: P=0.250 R=0.250 F1=0.250 CM=[[193, 3], [3, 1]]
#   임계값 0.3: P=0.333 R=0.500 F1=0.400 CM=[[192, 4], [2, 2]]
#   임계값 0.2: P=0.429 R=0.750 F1=0.545 CM=[[192, 4], [1, 3]]

# --------------------------------------------------------------------
# [임계값 이동 — 같은 모델, 다른 성적 `필수`]
# --------------------------------------------------------------------
logit_bal = make_pipeline(StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42,
                       class_weight='balanced'))
logit_bal.fit(X_train_clf, y_train_clf)
p = logit_bal.predict(X_test_clf)
print("혼동행렬:", confusion_matrix(y_test_clf, p).tolist(),
      f"P={precision_score(y_test_clf, p):.3f} "
      f"R={recall_score(y_test_clf, p):.3f} "
      f"F1={f1_score(y_test_clf, p):.3f}")

# 교재 실행 결과 ------------------------------------------------------
#   혼동행렬: [[187, 9], [1, 3]] P=0.250 R=0.750 F1=0.375

