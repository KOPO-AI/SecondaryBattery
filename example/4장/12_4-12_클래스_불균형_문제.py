# -*- coding: utf-8 -*-
"""4장 예제 — 4.12 클래스 불균형 문제 — 정확도 98.1%의 함정

교재 출처 : manuscript/41_ch4_분류.md
실행 방법 : example 폴더에서  python "4장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 09_4-9_로지스틱_회귀.py, 10_4-10_SVM_분류.py, 11_4-11_트리_기반_분류.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd
import numpy as np

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 4.1·4.8·4.9·4.11절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
# ── 4.1.2절: 정제 규약 clean_data() ──
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

# ── 4.9.3절: 로지스틱 회귀 import와 성능 출력 함수 report() ──
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def report(name, y_true, y_pred):
    print(f"{name:28s} 정확도={accuracy_score(y_true, y_pred):.3f} "
          f"정밀도={precision_score(y_true, y_pred, zero_division=0):.3f} "
          f"재현율={recall_score(y_true, y_pred):.3f} "
          f"F1={f1_score(y_true, y_pred):.3f}")

# ── 4.11.2절: 트리 계열 import와 기본 RandomForest rf ──
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)          # 트리 계열은 원본 스케일 그대로
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# ["전부 정상"이라고만 해도 정확도 98.1%]
# --------------------------------------------------------------------
# 무조건 '정상(0)'만 출력하는 더미 예측
pred_all_normal = np.zeros(len(y_test))
report("전부-정상 더미 모델", y_test, pred_all_normal)

# 교재 실행 결과 ------------------------------------------------------
#   전부-정상 더미 모델                  정확도=0.980 정밀도=0.000 재현율=0.000 F1=0.000

# --------------------------------------------------------------------
# [대응 ① 클래스 가중치 — class_weight='balanced']
# --------------------------------------------------------------------
logreg_bal = LogisticRegression(max_iter=1000, class_weight="balanced",
                                random_state=42)
logreg_bal.fit(X_train_s, y_train)
report("로지스틱(balanced)", y_test, logreg_bal.predict(X_test_s))

rf_bal = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=42)
rf_bal.fit(X_train, y_train)
report("RF(balanced)", y_test, rf_bal.predict(X_test))

# XGBoost는 scale_pos_weight = (음성 수 / 양성 수)
spw = (y_train == 0).sum() / (y_train == 1).sum()   # 52.6
xgb_w = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                      scale_pos_weight=spw,
                      eval_metric="logloss", random_state=42)
xgb_w.fit(X_train, y_train)
report("XGB(가중치)", y_test, xgb_w.predict(X_test))

# 교재 실행 결과 ------------------------------------------------------
#   로지스틱(balanced)               정확도=0.964 정밀도=0.333 재현율=0.800 F1=0.471
#   RF(balanced)                 정확도=0.972 정밀도=0.333 재현율=0.400 F1=0.364
#   XGB(가중치)                     정확도=0.976 정밀도=0.429 재현율=0.600 F1=0.500

# --------------------------------------------------------------------
# [대응 ② 오버샘플링 — SMOTE]
# --------------------------------------------------------------------
# 설치명은 imbalanced-learn, import명은 imblearn — 서로 다르다
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)               # k_neighbors 기본 5
X_train_sm, y_train_sm = smote.fit_resample(X_train_s, y_train)
print("SMOTE 후:", pd.Series(y_train_sm).value_counts().to_dict())

logreg_sm = LogisticRegression(max_iter=1000, random_state=42)
logreg_sm.fit(X_train_sm, y_train_sm)
report("로지스틱+SMOTE", y_test, logreg_sm.predict(X_test_s))

X_tr_sm2, y_tr_sm2 = SMOTE(random_state=42).fit_resample(X_train, y_train)
rf_sm = RandomForestClassifier(n_estimators=300, random_state=42)
rf_sm.fit(X_tr_sm2, y_tr_sm2)
report("RF+SMOTE", y_test, rf_sm.predict(X_test))

xgb_sm = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                       eval_metric="logloss", random_state=42)
xgb_sm.fit(X_tr_sm2, y_tr_sm2)
report("XGB+SMOTE", y_test, xgb_sm.predict(X_test))

# 교재 실행 결과 ------------------------------------------------------
#   SMOTE 후: {0: 736, 1: 736}
#   로지스틱+SMOTE                   정확도=0.960 정밀도=0.273 재현율=0.600 F1=0.375
#   RF+SMOTE                     정확도=0.968 정밀도=0.333 재현율=0.600 F1=0.429
#   XGB+SMOTE                    정확도=0.968 정밀도=0.333 재현율=0.600 F1=0.429

# --------------------------------------------------------------------
# [대응 ③ 임계값 조정 — 모델은 그대로, 판정선만 이동]
# --------------------------------------------------------------------
proba_rf = rf.predict_proba(X_test)[:, 1]     # 불량(클래스 1) 확률
for t in [0.5, 0.3, 0.1]:
    pred_t = (proba_rf >= t).astype(int)
    report(f"RF 임계값 {t}", y_test, pred_t)

