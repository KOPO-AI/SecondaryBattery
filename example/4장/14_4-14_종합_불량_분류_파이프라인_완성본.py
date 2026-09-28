# -*- coding: utf-8 -*-
"""4장 예제 — 4.14 종합: 불량 분류 파이프라인 완성본

교재 출처 : manuscript/41_ch4_분류.md
실행 방법 : example 폴더에서  python "4장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 11_4-11_트리_기반_분류.py, 12_4-12_클래스_불균형_문제.py, 13_4-13_확률_예측과_임계값.py
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
# [완성 코드 (배포 스크립트 실행·해설)]
# --------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""4장 후반 종합: 배터리 로트 불량 분류 파이프라인"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE   # 설치명: imbalanced-learn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# --- 1) 데이터 로드 및 정제 (3.7.1절 clean_data 규약) ---------------
# clean_data() 정의는 4.1.2절과 완전히 같으므로 지면에서는 생략했다.
# 배포 스크립트에는 이 함수 정의가 파일 맨 위에 그대로 들어 있다.
df = clean_data("data/battery_process_data.csv")   # 정제 후 1,000행

FEATS = ["믹싱_RPM", "믹싱_온도", "코팅_토출압력",
         "건조로_1구간_온도", "프레스_압력", "프레스_Gap"]
X, y = df[FEATS], df["불량_여부"]

# --- 2) 층화 분할 + 표준화 ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# --- 3) 모델 × 불균형 대응 조합 학습 --------------------------------
spw = (y_train == 0).sum() / (y_train == 1).sum()
X_sm_s, y_sm_s = SMOTE(random_state=42).fit_resample(X_train_s, y_train)
X_sm,   y_sm   = SMOTE(random_state=42).fit_resample(X_train,   y_train)

runs = [  # (이름, 모델, 학습X, 학습y, 평가X)
    ("로지스틱(기본)",      LogisticRegression(max_iter=1000, random_state=42),
     X_train_s, y_train, X_test_s),
    ("로지스틱(balanced)",  LogisticRegression(max_iter=1000, class_weight="balanced",
                                            random_state=42),
     X_train_s, y_train, X_test_s),
    ("로지스틱+SMOTE",      LogisticRegression(max_iter=1000, random_state=42),
     X_sm_s, y_sm_s, X_test_s),
    ("SVM RBF(기본)",       SVC(random_state=42), X_train_s, y_train, X_test_s),
    ("SVM RBF(balanced)",   SVC(class_weight="balanced", random_state=42),
     X_train_s, y_train, X_test_s),
    ("RF(기본)",            RandomForestClassifier(n_estimators=300, random_state=42),
     X_train, y_train, X_test),
    ("RF(balanced)",        RandomForestClassifier(n_estimators=300,
                                class_weight="balanced", random_state=42),
     X_train, y_train, X_test),
    ("RF+SMOTE",            RandomForestClassifier(n_estimators=300, random_state=42),
     X_sm, y_sm, X_test),
    ("XGB(기본)",           XGBClassifier(n_estimators=300, max_depth=4,
                                learning_rate=0.1, eval_metric="logloss",
                                random_state=42),
     X_train, y_train, X_test),
    ("XGB(가중치)",         XGBClassifier(n_estimators=300, max_depth=4,
                                learning_rate=0.1, scale_pos_weight=spw,
                                eval_metric="logloss", random_state=42),
     X_train, y_train, X_test),
    ("XGB+SMOTE",           XGBClassifier(n_estimators=300, max_depth=4,
                                learning_rate=0.1, eval_metric="logloss",
                                random_state=42),
     X_sm, y_sm, X_test),
]

rows = []
for name, model, X_fit, y_fit, X_eval in runs:
    model.fit(X_fit, y_fit)
    pred = model.predict(X_eval)
    rows.append([name,
                 accuracy_score(y_test, pred),
                 precision_score(y_test, pred, zero_division=0),
                 recall_score(y_test, pred),
                 f1_score(y_test, pred)])
result = pd.DataFrame(rows, columns=["모델", "정확도", "정밀도", "재현율", "F1"])
print(result.round(3).to_string(index=False))

# --- 4) 최종 후보 모델의 임계값 스윕 --------------------------------
final = LogisticRegression(max_iter=1000, class_weight="balanced",
                           random_state=42)
final.fit(X_train_s, y_train)
proba = final.predict_proba(X_test_s)[:, 1]
for t in [0.3, 0.5, 0.7]:
    pred_t = (proba >= t).astype(int)
    print(f"임계값 {t}: 정밀도={precision_score(y_test, pred_t, zero_division=0):.3f} "
          f"재현율={recall_score(y_test, pred_t):.3f} "
          f"F1={f1_score(y_test, pred_t):.3f}")

# 교재 실행 결과 ------------------------------------------------------
#                  모델   정확도   정밀도  재현율    F1
#            로지스틱(기본) 0.972 0.250  0.2 0.222
#      로지스틱(balanced) 0.964 0.333  0.8 0.471
#          로지스틱+SMOTE 0.960 0.273  0.6 0.375
#         SVM RBF(기본) 0.980 0.000  0.0 0.000
#   SVM RBF(balanced) 0.972 0.375  0.6 0.462
#              RF(기본) 0.972 0.250  0.2 0.222
#        RF(balanced) 0.972 0.333  0.4 0.364
#            RF+SMOTE 0.968 0.333  0.6 0.429
#             XGB(기본) 0.980 0.500  0.6 0.545
#            XGB(가중치) 0.976 0.429  0.6 0.500
#           XGB+SMOTE 0.968 0.333  0.6 0.429
#   임계값 0.3: 정밀도=0.200 재현율=0.800 F1=0.320
#   임계값 0.5: 정밀도=0.333 재현율=0.800 F1=0.471
#   임계값 0.7: 정밀도=0.333 재현율=0.600 F1=0.429

