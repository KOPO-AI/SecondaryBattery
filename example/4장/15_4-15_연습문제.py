# -*- coding: utf-8 -*-
"""4장 예제 — 4.15 연습문제 `선택 학습(자율 복습)`

교재 출처 : manuscript/41_ch4_분류.md
실행 방법 : example 폴더에서  python "4장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 12_4-12_클래스_불균형_문제.py, 13_4-13_확률_예측과_임계값.py, 14_4-14_종합_불량_분류_파이프라인_완성본.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd
import numpy as np
df = pd.read_csv("data/battery_process_data.csv")
df["건조로_1구간_온도"] = df["건조로_1구간_온도"].replace(9999, np.nan)
df = df.fillna(df.median(numeric_only=True))   # 3.7절 clean_data() 규약

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 4.1·4.8·4.9·4.11·4.12절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
# ── 4.1.2절: 한글 폰트, 정제 규약 clean_data() ──
import matplotlib.pyplot as plt

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

# ── 4.11.2절·4.12.3절: 트리 계열·SMOTE import ──
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [해답]
# --------------------------------------------------------------------
for rs in range(10):
    _, _, _, y_te = train_test_split(X, y, test_size=0.25, random_state=rs)
    print(f"random_state={rs}: 평가셋 불량 {int(y_te.sum())}건")

# 교재 실행 결과 ------------------------------------------------------
#   random_state=0: 평가셋 불량 5건
#   random_state=1: 평가셋 불량 2건
#   random_state=2: 평가셋 불량 7건
#   random_state=3: 평가셋 불량 4건
#   random_state=4: 평가셋 불량 4건
#   random_state=5: 평가셋 불량 6건
#   random_state=6: 평가셋 불량 5건
#   random_state=7: 평가셋 불량 6건
#   random_state=8: 평가셋 불량 3건
#   random_state=9: 평가셋 불량 4건

# --------------------------------------------------------------------
# [해답]
# --------------------------------------------------------------------
import matplotlib.pyplot as plt   # 한글 폰트 설정은 4.1.2절 첫 셀에서 이미 잡아 두었다

fig, ax = plt.subplots(figsize=(5, 4))
ax.boxplot([df.loc[df['불량_여부'] == 0, '코팅_토출압력'],
            df.loc[df['불량_여부'] == 1, '코팅_토출압력']],
           tick_labels=['정상', '불량'])       # labels= 는 제거된 인자
ax.set_ylabel('코팅_토출압력')
plt.tight_layout(); plt.show()

# --------------------------------------------------------------------
# [해답]
# --------------------------------------------------------------------
# 표준화본(로지스틱용)과 원본(트리용)을 각각 k_neighbors=3으로 다시 생성
X_s3, y_s3 = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_train_s, y_train)
X_r3, y_r3 = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_train,   y_train)

m1 = LogisticRegression(max_iter=1000, random_state=42).fit(X_s3, y_s3)
report("로지스틱+SMOTE(k=3)", y_test, m1.predict(X_test_s))

m2 = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_r3, y_r3)
report("RF+SMOTE(k=3)", y_test, m2.predict(X_test))

m3 = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                   eval_metric="logloss", random_state=42).fit(X_r3, y_r3)
report("XGB+SMOTE(k=3)", y_test, m3.predict(X_test))

# 교재 실행 결과 ------------------------------------------------------
#   로지스틱+SMOTE(k=3)              정확도=0.964 정밀도=0.300 재현율=0.600 F1=0.400
#   RF+SMOTE(k=3)                정확도=0.968 정밀도=0.333 재현율=0.600 F1=0.429
#   XGB+SMOTE(k=3)               정확도=0.972 정밀도=0.375 재현율=0.600 F1=0.462

