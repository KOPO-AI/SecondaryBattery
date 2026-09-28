# -*- coding: utf-8 -*-
"""4장 — 장 전체 예제 실행본

교재 4장의 파이썬 코드를 등장 순서대로 이어 붙였다.
절별 파일은 앞 절에서 만든 변수를 이어 쓰는 경우가 있어 단독 실행이 안 될 수 있는데,
이 파일은 처음부터 끝까지 순서대로 실행되므로 그대로 돌려 볼 수 있다.

실행 : example 폴더에서  python "4장/_장전체_실행본.py"
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""
import matplotlib
matplotlib.use("Agg")   # 창을 띄우지 않고 그림을 파일로만 처리


# ====================================================================
# 회귀 문제의 정의 — 공정 인자로 용량을 예측한다 / 데이터 준비 — 3.7.1절 `clean_data()` 규약 적용
# ====================================================================
import pandas as pd
import numpy as np
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


df = clean_data('data/battery_process_data.csv')

print(df.shape)
print(df.isna().sum().sum())   # 0 — 결측 없음 확인

FEATS = ['믹싱_RPM', '믹싱_온도', '코팅_토출압력',
         '건조로_1구간_온도', '프레스_압력', '프레스_Gap']
X = df[FEATS]
y = df['최종_용량_mAh']

print(y.describe())


# ====================================================================
# 학습/테스트 분할 — 모델의 실력을 공정하게 재는 법 / 왜 데이터를 나누는가
# ====================================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20%를 테스트용으로
    random_state=42     # 재현성을 위한 난수 시드 고정
)
print(X_train.shape, X_test.shape)  # (800, 6) (200, 6)


# ====================================================================
# 선형 회귀 — 가장 단순하고 가장 해석하기 좋은 모델 / sklearn 실습
# ====================================================================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

lr = LinearRegression()
lr.fit(X_train, y_train)          # 학습: 계수 추정

y_pred = lr.predict(X_test)       # 테스트 데이터 예측

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"테스트 R² = {r2:.4f}, RMSE = {rmse:.2f} mAh")
print(f"학습   R² = {lr.score(X_train, y_train):.4f}")


# ====================================================================
# 선형 회귀 — 가장 단순하고 가장 해석하기 좋은 모델 / 계수 해석 — Gap 1 µm당 용량이 얼마나 변하는가
# ====================================================================
coef_table = pd.DataFrame({
    '공정 인자': FEATS,
    '회귀계수': lr.coef_
})
print(f"절편: {lr.intercept_:.2f}")
print(coef_table)


# ====================================================================
# 선형 회귀 — 가장 단순하고 가장 해석하기 좋은 모델 / 잔차 분석 기초
# ====================================================================
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


# ====================================================================
# 트리 기반 회귀 — 비선형을 잡는 세 가지 무기 / 결정 트리 — 스무고개식 분할
# ====================================================================
from sklearn.tree import DecisionTreeRegressor

# 깊이 제한 없는 트리
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
print(f"학습 R² = {dt.score(X_train, y_train):.4f}")
print(f"테스트 R² = {r2_score(y_test, dt.predict(X_test)):.4f}")

# 깊이를 5로 제한한 트리
dt5 = DecisionTreeRegressor(max_depth=5, random_state=42)
dt5.fit(X_train, y_train)
print(f"깊이 5 테스트 R² = {r2_score(y_test, dt5.predict(X_test)):.4f}")


# ====================================================================
# 트리 기반 회귀 — 비선형을 잡는 세 가지 무기 / 랜덤 포레스트 — 배깅으로 분산을 줄인다
# ====================================================================
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,     # 트리 200그루
    oob_score=True,       # OOB 점수 계산
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print(f"테스트 R² = {r2_score(y_test, y_pred_rf):.4f}")
print(f"OOB R²   = {rf.oob_score_:.4f}")

# 특성 중요도
imp = pd.Series(rf.feature_importances_, index=FEATS).sort_values(ascending=False)
print(imp)


# ====================================================================
# 트리 기반 회귀 — 비선형을 잡는 세 가지 무기 / XGBoost — 부스팅으로 오차를 순차 보정한다
# ====================================================================
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(f"테스트 R² = {r2_score(y_test, y_pred_xgb):.4f}")


# ====================================================================
# 교차검증 — 단 한 번의 분할을 믿지 마라 / cross_val_score 실습
# ====================================================================
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_lr = cross_val_score(LinearRegression(), X, y, cv=kf, scoring='r2')
cv_rf = cross_val_score(
    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    X, y, cv=kf, scoring='r2')

print(f"선형회귀 fold별: {cv_lr.round(3)}")
print(f"선형회귀 평균 {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")
print(f"랜덤포레스트 평균 {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")


# ====================================================================
# 하이퍼파라미터 튜닝 — 모델의 손잡이를 체계적으로 돌린다 / GridSearchCV 실습
# ====================================================================
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 3, 5],
}   # 3×3×3 = 27조합 × 5-fold = 135회 학습

grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid, cv=5, scoring='r2', n_jobs=-1
)
grid.fit(X_train, y_train)   # 학습 데이터만 사용!

print("최적 조합:", grid.best_params_)
print(f"교차검증 최고 R² = {grid.best_score_:.4f}")

best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test)
print(f"테스트 R² = {r2_score(y_test, y_pred_best):.4f}")


# ====================================================================
# 하이퍼파라미터 튜닝 — 모델의 손잡이를 체계적으로 돌린다 / RandomizedSearchCV — 넓은 공간을 효율적으로 〔선택 학습(자율 복습)〕
# ====================================================================
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 15),
    'min_samples_leaf': randint(1, 10),
}
rand = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_dist, n_iter=20, cv=5, scoring='r2',
    random_state=42, n_jobs=-1
)
rand.fit(X_train, y_train)
print("최적 조합:", rand.best_params_)
print(f"테스트 R² = {r2_score(y_test, rand.best_estimator_.predict(X_test)):.4f}")


# ====================================================================
# 종합: 회귀 모델 구축 파이프라인 / 전체 코드 완성본 (배포 스크립트 실행·해설)
# ====================================================================
"""4장 종합: 배터리 용량 예측 회귀 파이프라인"""
import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, KFold,
                                     cross_val_score, GridSearchCV)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

RANDOM_STATE = 42

# ── 1. 데이터 로드 및 정제 (3.7.1절 clean_data 규약) ─────
# clean_data() 정의는 4.1.2절과 완전히 같으므로 지면에서는 생략했다.
# 배포 스크립트에는 이 함수 정의가 파일 맨 위에 그대로 들어 있다.
df = clean_data('data/battery_process_data.csv')   # 정제 후 1,000행

FEATS = ['믹싱_RPM', '믹싱_온도', '코팅_토출압력',
         '건조로_1구간_온도', '프레스_압력', '프레스_Gap']
X, y = df[FEATS], df['최종_용량_mAh']

# ── 2. 학습/테스트 분할 ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)

# ── 3. 후보 모델 정의 ─────────────────────────────────
models = {
    '선형회귀': LinearRegression(),
    '랜덤포레스트': RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        random_state=RANDOM_STATE),
}

# ── 4. 교차검증으로 후보 비교 (학습 데이터 내부) ─────────
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    print(f"{name:8s} CV R² = {cv.mean():.4f} ± {cv.std():.4f}")

# ── 5. 유망 모델(RF) 하이퍼파라미터 튜닝 ────────────────
grid = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, None],
     'min_samples_leaf': [1, 3, 5]},
    cv=5, scoring='r2', n_jobs=-1).fit(X_train, y_train)
models['RF(튜닝)'] = grid.best_estimator_

# ── 6. 최종 평가: 테스트 데이터로 단 한 번 ──────────────
print("\n=== 최종 테스트 성능 ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name:8s} R² = {r2:.4f}, RMSE = {rmse:.2f} mAh")


# ====================================================================
# 분류 문제의 정의 — "얼마나"에서 "무엇인가"로 / 실습 데이터 준비 — 4장 전반부와 같은 `clean_data()` 규약
# ====================================================================
import numpy as np
import pandas as pd

# clean_data()는 4.1.2절(= 3.7.1절)에서 정의한 함수를 그대로 쓴다.
# 새 노트북에서 시작했다면 4.1.2절의 정의 셀을 먼저 실행할 것.
df = clean_data("data/battery_process_data.csv")

print(df.shape)
print(df["불량_여부"].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 4.1.2절에서 정의한 것과 같은 6개 공정 변수 리스트, 같은 이름(FEATS)을 쓴다
FEATS = ["믹싱_RPM", "믹싱_온도", "코팅_토출압력",
         "건조로_1구간_온도", "프레스_압력", "프레스_Gap"]
X = df[FEATS]
y = df["불량_여부"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

print("학습:", y_train.value_counts().to_dict())
print("평가:", y_test.value_counts().to_dict())

# 로지스틱 회귀·SVM용 표준화 (트리 계열은 원본 사용)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)


# ====================================================================
# 로지스틱 회귀 — 확률을 예측하는 선형 모델 / sklearn 실습과 계수 해석
# ====================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def report(name, y_true, y_pred):
    print(f"{name:28s} 정확도={accuracy_score(y_true, y_pred):.3f} "
          f"정밀도={precision_score(y_true, y_pred, zero_division=0):.3f} "
          f"재현율={recall_score(y_true, y_pred):.3f} "
          f"F1={f1_score(y_true, y_pred):.3f}")

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_s, y_train)
report("로지스틱 회귀(기본)", y_test, logreg.predict(X_test_s))

# 계수와 오즈비 (표준화된 입력 기준: "1 표준편차 증가당" 효과)
coef_df = pd.DataFrame({
    "계수(w)": logreg.coef_[0].round(3),
    "오즈비(e^w)": np.exp(logreg.coef_[0]).round(3)
}, index=FEATS)
print(coef_df)


# ====================================================================
# SVM 분류 — 여유 폭을 최대로 하는 경계선 〔선택 학습(자율 복습)〕 / 실습
# ====================================================================
from sklearn.svm import SVC

svm_rbf = SVC(kernel="rbf", random_state=42)
svm_rbf.fit(X_train_s, y_train)
report("SVM RBF(기본)", y_test, svm_rbf.predict(X_test_s))

svm_bal = SVC(kernel="rbf", class_weight="balanced", random_state=42)
svm_bal.fit(X_train_s, y_train)
report("SVM RBF(balanced)", y_test, svm_bal.predict(X_test_s))


# ====================================================================
# 트리 기반 분류 — RandomForest와 XGBoost / 실습 — RandomForest와 XGBoost 분류기
# ====================================================================
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)          # 트리 계열은 원본 스케일 그대로
report("RandomForest(기본)", y_test, rf.predict(X_test))

xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                    eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)
report("XGBoost(기본)", y_test, xgb.predict(X_test))


# ====================================================================
# 클래스 불균형 문제 — 정확도 98.1%의 함정 / "전부 정상"이라고만 해도 정확도 98.1%
# ====================================================================
# 무조건 '정상(0)'만 출력하는 더미 예측
pred_all_normal = np.zeros(len(y_test))
report("전부-정상 더미 모델", y_test, pred_all_normal)


# ====================================================================
# 클래스 불균형 문제 — 정확도 98.1%의 함정 / 대응 ① 클래스 가중치 — class_weight='balanced'
# ====================================================================
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


# ====================================================================
# 클래스 불균형 문제 — 정확도 98.1%의 함정 / 대응 ② 오버샘플링 — SMOTE
# ====================================================================
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


# ====================================================================
# 클래스 불균형 문제 — 정확도 98.1%의 함정 / 대응 ③ 임계값 조정 — 모델은 그대로, 판정선만 이동
# ====================================================================
proba_rf = rf.predict_proba(X_test)[:, 1]     # 불량(클래스 1) 확률
for t in [0.5, 0.3, 0.1]:
    pred_t = (proba_rf >= t).astype(int)
    report(f"RF 임계값 {t}", y_test, pred_t)


# ====================================================================
# 확률 예측과 임계값 — 판정선은 비즈니스가 정한다 / predict가 아니라 predict_proba를 보라
# ====================================================================
proba = logreg_bal.predict_proba(X_test_s)[:, 1]
print(pd.Series(proba).describe().round(3))


# ====================================================================
# 확률 예측과 임계값 — 판정선은 비즈니스가 정한다 / 임계값을 움직이면 무슨 일이 벌어지는가
# ====================================================================
print(" 임계값  판정건수  정밀도  재현율    F1")
for t in [0.3, 0.5, 0.7, 0.8, 0.9]:
    pred_t = (proba >= t).astype(int)
    print(f"  {t:.1f}    {pred_t.sum():4d}   "
          f"{precision_score(y_test, pred_t, zero_division=0):.3f}  "
          f"{recall_score(y_test, pred_t):.3f}  "
          f"{f1_score(y_test, pred_t):.3f}")


# ====================================================================
# 종합: 불량 분류 파이프라인 완성본 / 완성 코드 (배포 스크립트 실행·해설)
# ====================================================================
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


# ====================================================================
# 연습문제 `선택 학습(자율 복습)` / 해답
# ====================================================================
for rs in range(10):
    _, _, _, y_te = train_test_split(X, y, test_size=0.25, random_state=rs)
    print(f"random_state={rs}: 평가셋 불량 {int(y_te.sum())}건")

import matplotlib.pyplot as plt   # 한글 폰트 설정은 4.1.2절 첫 셀에서 이미 잡아 두었다

fig, ax = plt.subplots(figsize=(5, 4))
ax.boxplot([df.loc[df['불량_여부'] == 0, '코팅_토출압력'],
            df.loc[df['불량_여부'] == 1, '코팅_토출압력']],
           tick_labels=['정상', '불량'])       # labels= 는 제거된 인자
ax.set_ylabel('코팅_토출압력')
plt.tight_layout(); plt.show()

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


# ====================================================================
# 실전 회귀 ① — 소표본 DoE 데이터 (WMG 캘린더링 54셀) `필수` 〔실습 45분〕 / 이번에 다루는 데이터
# ====================================================================
import pandas as pd
import numpy as np

wmg = pd.read_csv('data/wmg/wmg_cells_54.csv')

print(wmg.shape)
print(wmg[['cell_id', 'group', 'replicate', 'coat_weight_level',
           'density_level', 'grav_dis_5c_mahg']].head(6).to_string(index=False))


# ====================================================================
# 실전 회귀 ① — 소표본 DoE 데이터 (WMG 캘린더링 54셀) `필수` 〔실습 45분〕 / 표본 54개는 실험 54번이 아니다
# ====================================================================
# 조건(group)마다 셀이 몇 개인가
print(wmg.groupby('group').size().value_counts())

# 설계가 균형인지 교차표로 확인
print(wmg.pivot_table(index='coat_weight_level', columns='density_level',
                      values='grav_dis_5c_mahg', aggfunc='size'))

print(wmg['grav_dis_5c_mahg'].describe().round(2))


# ====================================================================
# 실전 회귀 ① — 소표본 DoE 데이터 (WMG 캘린더링 54셀) `필수` 〔실습 45분〕 / 4.2~4.4절 코드를 한 글자도 바꾸지 않고 이식한다
# ====================================================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

FEATS_W = ['target_coat_weight_gsm', 'roll_temp_c', 'target_density_g_cm3',
           'roll_gap_um', 'n_passes']
Xw = wmg[FEATS_W]
yw = wmg['grav_dis_5c_mahg']

Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(
    Xw, yw, test_size=0.2, random_state=42)      # 4.2절과 완전히 같은 줄
print(Xw_tr.shape, Xw_te.shape)

lr_w = LinearRegression().fit(Xw_tr, yw_tr)
rf_w = RandomForestRegressor(n_estimators=200, random_state=42,
                             n_jobs=-1).fit(Xw_tr, yw_tr)

for name, model in [('선형회귀', lr_w), ('랜덤포레스트', rf_w)]:
    pred = model.predict(Xw_te)
    rmse = np.sqrt(mean_squared_error(yw_te, pred))
    print(f"{name:6s} 테스트 R² = {r2_score(yw_te, pred):.4f}  "
          f"RMSE = {rmse:5.2f} mAh/g  학습 R² = {model.score(Xw_tr, yw_tr):.4f}")


# ====================================================================
# 실전 회귀 ① — 소표본 DoE 데이터 (WMG 캘린더링 54셀) `필수` 〔실습 45분〕 / 첫 번째 의심 — 시드를 열 번 바꿔 보라
# ====================================================================
for rs in range(10):
    a, b, c, d = train_test_split(Xw, yw, test_size=0.2, random_state=rs)
    s_lr = r2_score(d, LinearRegression().fit(a, c).predict(b))
    s_rf = r2_score(d, RandomForestRegressor(n_estimators=200, random_state=42,
                                             n_jobs=-1).fit(a, c).predict(b))
    print(f"random_state={rs}  선형 R²={s_lr:.4f}   랜덤포레스트 R²={s_rf:.4f}")


# ====================================================================
# 실전 회귀 ① — 소표본 DoE 데이터 (WMG 캘린더링 54셀) `필수` 〔실습 45분〕 / 두 번째 의심 — 테스트셋에 무엇이 들어 있었나
# ====================================================================
train_groups = set(wmg.loc[Xw_tr.index, 'group'])   # 학습셋이 담고 있는 조건 번호
test_groups = wmg.loc[Xw_te.index, 'group']         # 테스트 셀들의 조건 번호

print("학습셋이 담고 있는 조건 수 :", len(train_groups), "/ 18")
print("테스트 셀 11개의 조건 번호 :", sorted(test_groups.tolist()))
print("이 중 학습셋에도 있는 조건 :", int(test_groups.isin(train_groups).sum()), "개")

from sklearn.model_selection import KFold, GroupKFold, cross_val_score

kf  = KFold(n_splits=5, shuffle=True, random_state=42)   # 셀 단위 (4.5.2절과 동일)
gkf = GroupKFold(n_splits=6)                             # 조건 단위
groups = wmg['group']

for name, model in [('선형회귀', LinearRegression()),
                    ('랜덤포레스트', RandomForestRegressor(
                        n_estimators=200, random_state=42, n_jobs=-1))]:
    a = -cross_val_score(model, Xw, yw, cv=kf,
                         scoring='neg_root_mean_squared_error')
    b = -cross_val_score(model, Xw, yw, cv=gkf, groups=groups,
                         scoring='neg_root_mean_squared_error')
    print(f"{name:6s} 셀 단위 KFold(5)     RMSE = {a.mean():5.2f} ± {a.std():5.2f}")
    print(f"{name:6s} 조건 단위 GroupKFold RMSE = {b.mean():5.2f} ± {b.std():5.2f}")

b = -cross_val_score(RandomForestRegressor(n_estimators=200, random_state=42,
                                           n_jobs=-1),
                     Xw, yw, cv=gkf, groups=groups,
                     scoring='neg_root_mean_squared_error')
print("GroupKFold fold별 RMSE:", b.round(2))


# ====================================================================
# 실전 회귀 ① — 소표본 DoE 데이터 (WMG 캘린더링 54셀) `필수` 〔실습 45분〕 / 교호작용 — 두 인자가 서로의 효과를 바꾼다
# ====================================================================
print(wmg.pivot_table(index='coat_weight_level', columns='density_level',
                      values='grav_dis_5c_mahg',
                      aggfunc='mean')[['P', 'M', 'D']].round(1))


# ====================================================================
# 실전 회귀 ① — 소표본 DoE 데이터 (WMG 캘린더링 54셀) `필수` 〔실습 45분〕 / 소표본에서는 변수를 줄여야 이긴다
# ====================================================================
from sklearn.model_selection import GroupKFold, cross_val_score

후보 = {
    '① 설계인자 5개': ['target_coat_weight_gsm', 'roll_temp_c',
                  'target_density_g_cm3', 'roll_gap_um', 'n_passes'],
    '② 결과변수 제외': ['target_coat_weight_gsm', 'roll_temp_c',
                  'target_density_g_cm3'],
    '③ 핵심 2개':    ['target_coat_weight_gsm', 'target_density_g_cm3'],
    '④ 중량 1개':    ['target_coat_weight_gsm'],
}

gkf = GroupKFold(n_splits=6)
for 이름, cols in 후보.items():
    a = -cross_val_score(LinearRegression(), wmg[cols], yw, cv=gkf,
                         groups=groups, scoring='neg_root_mean_squared_error')
    b = -cross_val_score(RandomForestRegressor(n_estimators=200, random_state=42,
                                               n_jobs=-1),
                         wmg[cols], yw, cv=gkf, groups=groups,
                         scoring='neg_root_mean_squared_error')
    print(f"{이름:12s} 선형 {a.mean():5.2f} ± {a.std():5.2f}   "
          f"랜덤포레스트 {b.mean():5.2f} ± {b.std():5.2f}")


# ====================================================================
# 실전 회귀 ① — 소표본 DoE 데이터 (WMG 캘린더링 54셀) `필수` 〔실습 45분〕 / 여기가 한계다 — 재현성이 정하는 성능 상한
# ====================================================================
반복SD = wmg.groupby('group')['grav_dis_5c_mahg'].std()

print(f"조건 안 반복 3셀의 SD  중앙값 = {반복SD.median():5.2f} mAh/g")
print(f"                      최소 = {반복SD.min():5.2f} mAh/g")
print(f"                      최대 = {반복SD.max():5.2f} mAh/g")
print(f"54셀 전체 SD                = {yw.std():5.2f} mAh/g")

재현성 = pd.DataFrame({'조건내SD': 반복SD.round(2),
                    '중량': wmg.groupby('group')['coat_weight_level'].first(),
                    '밀도': wmg.groupby('group')['density_level'].first(),
                    '평균용량': wmg.groupby('group')['grav_dis_5c_mahg'].mean().round(1)})
print(재현성.sort_values('조건내SD').to_string())

print(재현성.groupby('중량')['조건내SD'].agg(['min', 'median', 'max']).round(2))

편차 = wmg['grav_dis_5c_mahg'] - wmg.groupby('group')['grav_dis_5c_mahg'].transform('mean')

풀드SD = np.sqrt((편차 ** 2).sum() / (len(wmg) - wmg['group'].nunique()))
설명비율 = 1 - (편차 ** 2).sum() / ((yw - yw.mean()) ** 2).sum()

print(f"풀드 조건내 표준편차     = {풀드SD:.2f} mAh/g   ← RMSE의 하한")
print(f"조건이 설명하는 분산 비율 = {설명비율*100:.1f}%")

from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()          # 18개 조건을 하나씩 빼며 18번 검증
X3 = wmg[['target_coat_weight_gsm', 'target_density_g_cm3']]

s = -cross_val_score(RandomForestRegressor(n_estimators=200, random_state=42,
                                           n_jobs=-1),
                     X3, yw, cv=logo, groups=groups,
                     scoring='neg_root_mean_squared_error')
print(f"LeaveOneGroupOut(18조건) RMSE = {s.mean():.2f} ± {s.std():.2f}")
print(f"  가장 잘 맞힌 조건 {s.min():.2f} / 가장 못 맞힌 조건 {s.max():.2f}")


# ====================================================================
# 실전 회귀 ② — 타깃 누수 2단 시연 (Battery RUL) `필수` 〔실습 30분〕 / 이번에 다루는 데이터
# ====================================================================
rul = pd.read_csv('data/rul/battery_rul_clean.csv')

print(rul.shape)
print(rul.columns.tolist())
print(rul.head(3).to_string(index=False))


# ====================================================================
# 실전 회귀 ② — 타깃 누수 2단 시연 (Battery RUL) `필수` 〔실습 30분〕 / 1단계 — 아무 생각 없이 전부 넣는다
# ====================================================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

X_all = rul.drop(columns=['cell_id', 'rul'])   # 식별자와 타깃만 제외
y = rul['rul']
print("입력 8개 :", X_all.columns.tolist())

Xtr, Xte, ytr, yte = train_test_split(X_all, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(Xtr, ytr)
pred = rf.predict(Xte)

print(f"테스트 R²  = {r2_score(yte, pred):.6f}")
print(f"테스트 MAE = {mean_absolute_error(yte, pred):.2f} 사이클")


# ====================================================================
# 실전 회귀 ② — 타깃 누수 2단 시연 (Battery RUL) `필수` 〔실습 30분〕 / 범인을 찾는다 — 특성 중요도부터
# ====================================================================
imp = pd.Series(rf.feature_importances_, index=X_all.columns)
print(imp.sort_values(ascending=False).round(4).to_string())

from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(Xtr[['cycle_index']], ytr)   # 변수 딱 하나
p1 = lr.predict(Xte[['cycle_index']])

print(f"cycle_index 하나만 쓴 선형회귀 R² = {r2_score(yte, p1):.6f}")
print(f"기울기 = {lr.coef_[0]:.4f}, 절편 = {lr.intercept_:.2f}")

합 = rul['rul'] + rul['cycle_index']
확인 = 합.groupby(rul['cell_id']).nunique()      # 셀마다 서로 다른 값이 몇 개인가

print(확인.to_string())
print("셀마다 값이 하나뿐인가?", bool((확인 == 1).all()))
print()
print(합.groupby(rul['cell_id']).first().astype(int).to_string())


# ====================================================================
# 실전 회귀 ② — 타깃 누수 2단 시연 (Battery RUL) `필수` 〔실습 30분〕 / 2단계 — 누수 컬럼을 빼고 다시 학습한다
# ====================================================================
X7 = X_all.drop(columns=['cycle_index'])          # 8개 → 7개

Xtr7, Xte7, ytr7, yte7 = train_test_split(X7, y, test_size=0.2, random_state=42)
rf7 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf7.fit(Xtr7, ytr7)
pred7 = rf7.predict(Xte7)

print(f"테스트 R²  = {r2_score(yte7, pred7):.6f}")
print(f"테스트 MAE = {mean_absolute_error(yte7, pred7):.2f} 사이클")


# ====================================================================
# 실전 회귀 ② — 타깃 누수 2단 시연 (Battery RUL) `필수` 〔실습 30분〕 / 3단계 — 셀 단위로 나누면 한 번 더 무너진다
# ====================================================================
from sklearn.model_selection import GroupKFold, cross_val_score

gkf = GroupKFold(n_splits=14)      # 셀이 14개 → 한 번에 한 셀씩 통째로 평가

for tag, X in [('cycle_index 포함(8특징)', X_all), ('cycle_index 제외(7특징)', X7)]:
    mae = -cross_val_score(RandomForestRegressor(n_estimators=200,
                                                 random_state=42, n_jobs=-1),
                           X, y, cv=gkf, groups=rul['cell_id'],
                           scoring='neg_mean_absolute_error')
    print(f"{tag:24s} MAE = {mae.mean():6.2f} ± {mae.std():5.2f} "
          f"(최악 셀 {mae.max():.2f})")


# ====================================================================
# 실전 분류 — 극단 불균형·고차원·시간 드리프트 (UCI SECOM) `선택 학습(자율 복습)` 〔실습 30분〕 / 첫 30초 — `dropna()`가 데이터를 전멸시킨다
# ====================================================================
sec = pd.read_csv('data/secom/secom_merged.csv', parse_dates=['timestamp'])

print(sec.shape)
print(sec['fail'].value_counts())
print(f"불량률 = {sec['fail'].mean()*100:.2f}%")

print("결측이 하나도 없는 행 =", int((sec.isna().sum(axis=1) == 0).sum()), "행")
print("dropna() 결과        =", sec.dropna().shape)


# ====================================================================
# 실전 분류 — 극단 불균형·고차원·시간 드리프트 (UCI SECOM) `선택 학습(자율 복습)` 〔실습 30분〕 / 590개 익명 센서 정리하기
# ====================================================================
센서 = [c for c in sec.columns if c.startswith('sensor_')]
print("센서 개수 :", len(센서))

분산0   = [c for c in 센서 if sec[c].nunique(dropna=True) <= 1]
결측많음 = [c for c in 센서 if sec[c].isna().mean() > 0.5]

print("분산이 0인 센서    :", len(분산0), "개")
print("결측 50% 초과 센서 :", len(결측많음), "개")

쓸센서 = [c for c in 센서 if c not in 분산0 and c not in 결측많음]
print("남은 센서          :", len(쓸센서), "개")


# ====================================================================
# 실전 분류 — 극단 불균형·고차원·시간 드리프트 (UCI SECOM) `선택 학습(자율 복습)` 〔실습 30분〕 / 정확도 93.4%의 함정, 그리고 class_weight가 듣지 않을 때
# ====================================================================
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score, confusion_matrix)

X, y = sec[쓸센서], sec['fail']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)
print("학습:", y_tr.value_counts().to_dict())
print("평가:", y_te.value_counts().to_dict())
print(f"전부 정상이라 답할 때의 정확도 = {1 - y_te.mean():.4f}")

def 만들기(분류기):
    return Pipeline([('결측대체', SimpleImputer(strategy='median')),
                     ('표준화',   StandardScaler()),
                     ('분류기',   분류기)])


def 성적표(이름, y_true, y_pred, 점수=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    pr = f"  PR-AUC={average_precision_score(y_true, 점수):.3f}" if 점수 is not None else ""
    print(f"{이름:22s} 정확도={accuracy_score(y_true, y_pred):.3f} "
          f"정밀도={precision_score(y_true, y_pred, zero_division=0):.3f} "
          f"재현율={recall_score(y_true, y_pred):.3f} "
          f"F1={f1_score(y_true, y_pred):.3f}  검출 {tp}/{tp+fn}건, 과검 {fp}건{pr}")


성적표('전부-정상 더미', y_te, np.zeros(len(y_te), dtype=int))

로짓 = 만들기(LogisticRegression(max_iter=2000, random_state=42)).fit(X_tr, y_tr)
성적표('로지스틱(기본)', y_te, 로짓.predict(X_te), 로짓.predict_proba(X_te)[:, 1])

로짓B = 만들기(LogisticRegression(max_iter=2000, class_weight='balanced',
                              random_state=42)).fit(X_tr, y_tr)
성적표('로지스틱(balanced)', y_te, 로짓B.predict(X_te), 로짓B.predict_proba(X_te)[:, 1])

숲 = 만들기(RandomForestClassifier(n_estimators=300, random_state=42,
                                n_jobs=-1)).fit(X_tr, y_tr)
성적표('RF(기본)', y_te, 숲.predict(X_te), 숲.predict_proba(X_te)[:, 1])

숲B = 만들기(RandomForestClassifier(n_estimators=300,
                                class_weight='balanced_subsample',
                                random_state=42, n_jobs=-1)).fit(X_tr, y_tr)
성적표('RF(balanced)', y_te, 숲B.predict(X_te), 숲B.predict_proba(X_te)[:, 1])

print(f"\n무작위로 찍었을 때의 PR-AUC 기대값 = {y_te.mean():.3f}")


# ====================================================================
# 실전 분류 — 극단 불균형·고차원·시간 드리프트 (UCI SECOM) `선택 학습(자율 복습)` 〔실습 30분〕 / 판정선을 옮긴다 — 여기서 유일하게 듣는 손잡이
# ====================================================================
확률 = 숲.predict_proba(X_te)[:, 1]        # RF(기본)의 불량 확률

print(" 임계값  판정건수  검출  과검  정밀도  재현율    F1")
for t in [0.50, 0.30, 0.20, 0.15, 0.10, 0.05]:
    판정 = (확률 >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, 판정).ravel()
    print(f"  {t:.2f}    {판정.sum():5d}   {tp:3d}  {fp:4d}   "
          f"{precision_score(y_te, 판정, zero_division=0):.3f}  "
          f"{recall_score(y_te, 판정):.3f}  {f1_score(y_te, 판정):.3f}")


# ====================================================================
# 실전 분류 — 극단 불균형·고차원·시간 드리프트 (UCI SECOM) `선택 학습(자율 복습)` 〔실습 30분〕 / 현장의 언어로 바꾸기 — "재검 예산이 하루 20건이라면"
# ====================================================================
기저 = y_te.mean()

print(" 재검 예산   검출 불량   적중률   무작위 대비")
for k in [10, 20, 40, 80]:
    상위 = np.argsort(-확률)[:k]              # 확률 내림차순 상위 K개의 위치
    검출 = int(y_te.values[상위].sum())
    print(f"  상위 {k:3d}건   {검출:5d}건   {검출/k*100:5.1f}%   {(검출/k)/기저:.1f}배")


# ====================================================================
# 실전 분류 — 극단 불균형·고차원·시간 드리프트 (UCI SECOM) `선택 학습(자율 복습)` 〔실습 30분〕 / 시간순으로 나누면 전부 무너진다
# ====================================================================
월별 = sec.groupby(sec['timestamp'].dt.to_period('M'))['fail'].agg(['size', 'sum', 'mean'])
월별['mean'] = (월별['mean'] * 100).round(2)
print(월별.to_string())

경계 = int(len(sec) * 0.75)          # 데이터는 시각 오름차순으로 정렬돼 있다
X_tr2, X_te2 = X.iloc[:경계], X.iloc[경계:]
y_tr2, y_te2 = y.iloc[:경계], y.iloc[경계:]

print(f"학습 구간: {sec['timestamp'].iloc[0]:%Y-%m-%d} ~ "
      f"{sec['timestamp'].iloc[경계-1]:%Y-%m-%d}  불량 {y_tr2.mean()*100:.2f}%")
print(f"평가 구간: {sec['timestamp'].iloc[경계]:%Y-%m-%d} ~ "
      f"{sec['timestamp'].iloc[-1]:%Y-%m-%d}  불량 {y_te2.mean()*100:.2f}%")

숲2 = 만들기(RandomForestClassifier(n_estimators=300, random_state=42,
                                n_jobs=-1)).fit(X_tr2, y_tr2)
확률2 = 숲2.predict_proba(X_te2)[:, 1]

print(f"시간순 분할 PR-AUC = {average_precision_score(y_te2, 확률2):.3f} "
      f"(기저 {y_te2.mean():.3f})")
for k in [10, 20, 40]:
    상위 = np.argsort(-확률2)[:k]
    검출 = int(y_te2.values[상위].sum())
    print(f"  상위 {k:3d}건   {검출:3d}건 검출   무작위 대비 {(검출/k)/y_te2.mean():.1f}배")

