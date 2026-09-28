# -*- coding: utf-8 -*-
"""5장 — 장 전체 예제 실행본

교재 5장의 파이썬 코드를 등장 순서대로 이어 붙였다.
절별 파일은 앞 절에서 만든 변수를 이어 쓰는 경우가 있어 단독 실행이 안 될 수 있는데,
이 파일은 처음부터 끝까지 순서대로 실행되므로 그대로 돌려 볼 수 있다.

실행 : example 폴더에서  python "5장/_장전체_실행본.py"
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""
import matplotlib
matplotlib.use("Agg")   # 창을 띄우지 않고 그림을 파일로만 처리


# ====================================================================
# 왜 평가가 모델링보다 중요한가 `필수` / 실습 환경 준비 — 3.7.1절 `clean_data()` 규약 적용
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
print("정제 후:", df.shape, "| 결측 셀:", df.isna().sum().sum())
print("불량 분포:", df['불량_여부'].value_counts().to_dict())

FEATS = ['믹싱_RPM', '믹싱_온도', '코팅_토출압력',
         '건조로_1구간_온도', '프레스_압력', '프레스_Gap']
X = df[FEATS]
y_reg = df['최종_용량_mAh']   # 회귀 타깃
y_clf = df['불량_여부']        # 분류 타깃


# ====================================================================
# 회귀 평가지표 — 예측이 얼마나 빗나갔는가 `필수` / 실습 — 4장 회귀 3모델 전수 평가 `필수`
# ====================================================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)

models = {
    '선형회귀':      LinearRegression(),
    '랜덤 포레스트': RandomForestRegressor(n_estimators=200, random_state=42),
    'XGBoost':      XGBRegressor(n_estimators=200, learning_rate=0.1,
                                 max_depth=4, random_state=42),
}

print(f"{'모델':<12}{'MSE':>10}{'RMSE':>8}{'MAE':>8}{'R2':>9}{'MAPE(%)':>9}")
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    mse = mean_squared_error(y_test, p)
    print(f"{name:<12}{mse:>10.2f}{np.sqrt(mse):>8.2f}"
          f"{mean_absolute_error(y_test, p):>8.2f}"
          f"{r2_score(y_test, p):>9.4f}"
          f"{mean_absolute_percentage_error(y_test, p)*100:>9.3f}")

# 바닥선: 학습셋 평균으로만 예측
p0 = np.full(len(y_test), y_train.mean())
print(f"평균 예측    RMSE={np.sqrt(mean_squared_error(y_test, p0)):.2f} "
      f"MAE={mean_absolute_error(y_test, p0):.2f} "
      f"R2={r2_score(y_test, p0):.4f}")


# ====================================================================
# 분류 평가지표 — 불량 판정 모델의 성적표 `필수` / 정확도의 함정 — 5.1절 사고의 재현
# ====================================================================
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


# ====================================================================
# 분류 평가지표 — 불량 판정 모델의 성적표 `필수` / 임계값 이동 — 같은 모델, 다른 성적 `필수`
# ====================================================================
m = clfs['랜덤 포레스트']
pr = m.predict_proba(X_test_clf)[:, 1]
for t in (0.5, 0.3, 0.2):
    p = (pr >= t).astype(int)
    print(f"임계값 {t}: P={precision_score(y_test_clf, p, zero_division=0):.3f} "
          f"R={recall_score(y_test_clf, p, zero_division=0):.3f} "
          f"F1={f1_score(y_test_clf, p, zero_division=0):.3f} "
          f"CM={confusion_matrix(y_test_clf, p).tolist()}")

logit_bal = make_pipeline(StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42,
                       class_weight='balanced'))
logit_bal.fit(X_train_clf, y_train_clf)
p = logit_bal.predict(X_test_clf)
print("혼동행렬:", confusion_matrix(y_test_clf, p).tolist(),
      f"P={precision_score(y_test_clf, p):.3f} "
      f"R={recall_score(y_test_clf, p):.3f} "
      f"F1={f1_score(y_test_clf, p):.3f}")


# ====================================================================
# 검증 전략 — 공정한 시험 설계 `필수` / Stratified K-fold — 불균형 분류의 기본값
# ====================================================================
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

kf  = KFold(n_splits=5)                                        # 무작위 셔플 없음
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("KFold fold별 불량 수:",
      [int(y_clf.iloc[te].sum()) for _, te in kf.split(X)])
print("Stratified fold별 불량 수:",
      [int(y_clf.iloc[te].sum()) for _, te in skf.split(X, y_clf)])

rf = RandomForestClassifier(n_estimators=200, random_state=42)
s = cross_val_score(rf, X, y_clf, cv=skf, scoring='f1')
print("Stratified 5-fold F1:", np.round(s, 4).tolist(),
      "| 평균", round(s.mean(), 4), "| 표준편차", round(s.std(), 4))


# ====================================================================
# 검증 전략 — 공정한 시험 설계 `필수` / 시계열 데이터 — 미래를 훔쳐보게 하지 마라
# ====================================================================
from sklearn.model_selection import TimeSeriesSplit
tss = TimeSeriesSplit(n_splits=5)
for i, (tr, te) in enumerate(tss.split(X), 1):
    print(f"fold {i}: 훈련 [{tr.min()}..{tr.max()}] → 테스트 [{te.min()}..{te.max()}]")


# ====================================================================
# 검증 전략 — 공정한 시험 설계 `필수` / 실제 데이터 ① — "GroupKFold를 썼다"는 답이 아니다 (EIS) `필수` · 실습 15분
# ====================================================================
import pandas as pd
import numpy as np

eis = pd.read_csv('data/eis/eis_wide.csv')
print("표본 수, 열 수:", eis.shape)
print("셀 목록:", sorted(eis['battery_id'].unique()))
print("셀별 표본 수:", eis['battery_id'].value_counts().sort_index().to_dict())
print("측정 회차(measure_id) 수:", eis['measure_id'].nunique())
print(eis[['sample_id', 'battery_id', 'repeat_id', 'soc',
           'Zre_0.05Hz', 'Zre_1000Hz']].head(3))

ID_COLS = ['sample_id', 'measure_id', 'battery_id', 'repeat_id', 'soc']
X_eis = eis.drop(columns=ID_COLS)   # 피처 28개 = 주파수 14점 x (실수부, 허수부)
y_eis = eis['soc']                  # 타깃: 충전 상태 10~100 %
print("X_eis:", X_eis.shape, "| y_eis 범위:", y_eis.min(), "~", y_eis.max())

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_score

rf_eis = RandomForestRegressor(n_estimators=300, random_state=42)


def mae_by_fold(cv, groups=None):
    """fold별 MAE 배열을 돌려준다 (부호를 뒤집어 양수로)."""
    return -cross_val_score(rf_eis, X_eis, y_eis, cv=cv, groups=groups,
                            scoring='neg_mean_absolute_error')


plans = [
    ('① 무작위 KFold(4)',       KFold(n_splits=4, shuffle=True, random_state=42), None),
    ('② GroupKFold(measure_id)', GroupKFold(n_splits=4), eis['measure_id']),
    ('③ GroupKFold(battery_id)', GroupKFold(n_splits=4), eis['battery_id']),
]
for name, cv, g in plans:
    m = mae_by_fold(cv, g)
    print(f"{name:26s} 평균 MAE={m.mean():5.2f} %SOC | "
          f"fold별 {np.round(m, 2).tolist()} | 표준편차 {m.std():.2f}")

print(eis.groupby('battery_id')['Zre_1000Hz'].mean().round(6))

gkf = GroupKFold(n_splits=4)
for i, (tr, te) in enumerate(gkf.split(X_eis, y_eis, groups=eis['battery_id']), 1):
    test_cell = eis['battery_id'].iloc[te].unique()[0]
    rf_eis.fit(X_eis.iloc[tr], y_eis.iloc[tr])
    e = np.abs(rf_eis.predict(X_eis.iloc[te]) - y_eis.iloc[te]).mean()
    print(f"fold {i}: 테스트 셀 = {test_cell} (n={len(te)}) → MAE = {e:.2f} %SOC")

from sklearn.model_selection import cross_val_predict

pred_eis = cross_val_predict(rf_eis, X_eis, y_eis, cv=GroupKFold(n_splits=4),
                             groups=eis['battery_id'])
err_eis = pd.DataFrame({'soc': y_eis, 'abs_err': np.abs(pred_eis - y_eis)})
print(err_eis.groupby('soc')['abs_err'].mean().round(2))
print("저SOC(10~30) 평균 :",
      round(err_eis.loc[err_eis['soc'] <= 30, 'abs_err'].mean(), 2))
print("중·고SOC(40~100)  :",
      round(err_eis.loc[err_eis['soc'] >= 40, 'abs_err'].mean(), 2))


# ====================================================================
# 검증 전략 — 공정한 시험 설계 `필수` / 실제 데이터 ② — 셀 단위로 나누면 드러나는 진짜 성능 (Battery RUL) `선택 학습(자율 복습)` · 실습 10분
# ====================================================================
rul = pd.read_csv('data/rul/battery_rul_clean.csv')
print("행 수, 열 수:", rul.shape)
print("셀 수:", rul['cell_id'].nunique())
print("셀별 행 수:", rul['cell_id'].value_counts().sort_index().tolist())
print(rul[['cell_id', 'cycle_index', 'max_volt_discharge_v', 'rul']].head(3))

ok = []
for cid, d in rul.groupby('cell_id'):
    ok.append(bool((d['rul'] == d['cycle_index'].max() - d['cycle_index']).all()))
print("rul == (셀별 마지막 사이클 - 현재 사이클) 이 성립하는 셀:",
      sum(ok), "/", len(ok))
print("상관계수 r(cycle_index, rul) =",
      round(float(rul['cycle_index'].corr(rul['rul'])), 6))

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_score

FEAT_ALL = ['cycle_index', 'discharge_time_s', 'decrement_36_34v_s',
            'max_volt_discharge_v', 'min_volt_charge_v', 'time_at_415v_s',
            'time_cc_s', 'charging_time_s']
FEAT_NOCYC = [c for c in FEAT_ALL if c != 'cycle_index']

rf_rul = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
y_rul, g_rul = rul['rul'], rul['cell_id']

for label, feats in [('cycle_index 포함 (8특징)', FEAT_ALL),
                     ('cycle_index 제외 (7특징)', FEAT_NOCYC)]:
    X_rul = rul[feats]
    a = -cross_val_score(rf_rul, X_rul, y_rul,
                         cv=KFold(5, shuffle=True, random_state=42),
                         scoring='neg_mean_absolute_error')
    b = -cross_val_score(rf_rul, X_rul, y_rul, cv=GroupKFold(n_splits=5),
                         groups=g_rul, scoring='neg_mean_absolute_error')
    print(f"[{label}]")
    print(f"   무작위 5-fold  MAE = {a.mean():6.2f} ± {a.std():.2f} 사이클")
    print(f"   셀 단위 5-fold MAE = {b.mean():6.2f} ± {b.std():.2f} 사이클  "
          f"fold별 {np.round(b, 1).tolist()}")


# ====================================================================
# 과적합 진단 — 학습곡선 읽는 법 `필수` / 실습 — 세 모델의 학습곡선 `필수`
# ====================================================================
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeRegressor

for name, est in [('깊은 결정트리', DecisionTreeRegressor(random_state=42)),
                  ('랜덤 포레스트', RandomForestRegressor(n_estimators=200,
                                                          random_state=42)),
                  ('선형회귀',      LinearRegression())]:
    sizes, tr_sc, va_sc = learning_curve(
        est, X, y_reg, cv=5, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=True, random_state=42)
    print(name)
    print("  훈련량   :", sizes.tolist())
    print("  훈련 R2  :", np.round(tr_sc.mean(axis=1), 4).tolist())
    print("  검증 R2  :", np.round(va_sc.mean(axis=1), 4).tolist())


# ====================================================================
# 모델 성능 비교 실험 — 통일 조건 벤치마크 / 실습 — 5모델 종합 벤치마크 `필수`
# ====================================================================
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# --- 회귀 벤치마크: 5-fold, RMSE와 R2 ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
reg_bench = {
    '선형회귀':      LinearRegression(),
    '릿지(α=1)':    Ridge(alpha=1.0),
    '결정트리(d=6)': DecisionTreeRegressor(max_depth=6, random_state=42),
    '랜덤 포레스트': RandomForestRegressor(n_estimators=200, random_state=42),
    'XGBoost':      XGBRegressor(n_estimators=200, learning_rate=0.1,
                                 max_depth=4, random_state=42),
}
for name, m in reg_bench.items():
    rmse = -cross_val_score(m, X, y_reg, cv=kf,
                            scoring='neg_root_mean_squared_error')
    r2 = cross_val_score(m, X, y_reg, cv=kf, scoring='r2')
    print(f"{name}: RMSE={rmse.mean():.2f}±{rmse.std():.2f} "
          f"R2={r2.mean():.4f}±{r2.std():.4f}")

# --- 분류 벤치마크: 층화 5-fold, F1과 ROC-AUC ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf_bench = {
    '로지스틱':      make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, random_state=42)),
    '결정트리(d=6)': DecisionTreeClassifier(max_depth=6, random_state=42),
    '랜덤 포레스트': RandomForestClassifier(n_estimators=200, random_state=42),
    '그래디언트부스팅': GradientBoostingClassifier(random_state=42),
    'XGBoost':      XGBClassifier(n_estimators=200, learning_rate=0.1,
                        max_depth=4, random_state=42, eval_metric='logloss'),
}
for name, m in clf_bench.items():
    f1  = cross_val_score(m, X, y_clf, cv=skf, scoring='f1')
    auc = cross_val_score(m, X, y_clf, cv=skf, scoring='roc_auc')
    print(f"{name}: F1={f1.mean():.4f}±{f1.std():.4f} "
          f"AUC={auc.mean():.4f}±{auc.std():.4f}")


# ====================================================================
# 모델 성능 비교 실험 — 통일 조건 벤치마크 / 그 차이는 우연인가 — 반복 실험의 관점 `선택 학습(자율 복습)`
# ====================================================================
rf_scores, xgb_scores = [], []
for seed in range(10):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rf_scores.append(cross_val_score(
        RandomForestClassifier(n_estimators=200, random_state=42),
        X, y_clf, cv=cv, scoring='f1').mean())
    xgb_scores.append(cross_val_score(
        XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,
                      random_state=42, eval_metric='logloss'),
        X, y_clf, cv=cv, scoring='f1').mean())
print("RF :", np.round(rf_scores, 3).tolist(),
      "| 평균", round(np.mean(rf_scores), 4), "±", round(np.std(rf_scores), 4))
print("XGB:", np.round(xgb_scores, 3).tolist(),
      "| 평균", round(np.mean(xgb_scores), 4), "±", round(np.std(xgb_scores), 4))


# ====================================================================
# 모델 성능 비교 실험 — 통일 조건 벤치마크 / 실제 데이터 ③ — 결함이 여러 개 동시에 나올 때: macro와 micro (CoatingVision) `선택 학습(자율 복습)` · 실습 20분
# ====================================================================
cvd = pd.read_csv('data/coatingvision/coating_labels_tidy.csv')
LABELS = ['Surface_Crack', 'Pinhole', 'Delamination']

print("패치 수:", len(cvd), "| 프레임 수:", cvd['group_frame'].nunique(),
      "| 영상 수:", cvd['video_id'].nunique())
print("클래스별 양성 수:", cvd[LABELS].sum().to_dict())
print("클래스별 양성 비율:", cvd[LABELS].mean().round(3).to_dict())
print("한 패치에 붙은 결함 종류 수:",
      cvd['n_defect_types'].value_counts().sort_index().to_dict())

cvd['pos_code'] = cvd['position'].astype('category').cat.codes
FEATS_CV = ['coating_gap_um', 'run', 'frame', 'patch', 'pos_code']
X_cv = cvd[FEATS_CV].to_numpy()
Y_cv = cvd[LABELS].to_numpy()
print("X_cv:", X_cv.shape, "| Y_cv:", Y_cv.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import f1_score, average_precision_score


def oof_proba(splitter, groups=None):
    """검증 fold에서만 얻은 예측 확률(out-of-fold)을 모아 돌려준다."""
    proba = np.zeros(Y_cv.shape, dtype=float)
    for tr, te in splitter.split(X_cv, Y_cv[:, 0], groups=groups):
        for j in range(Y_cv.shape[1]):              # 라벨 3개를 따로 학습
            m = RandomForestClassifier(n_estimators=200, random_state=42,
                                       n_jobs=-1)
            m.fit(X_cv[tr], Y_cv[tr, j])
            proba[te, j] = m.predict_proba(X_cv[te])[:, 1]
    return proba


plans_cv = [
    ('무작위 KFold(5)',    KFold(5, shuffle=True, random_state=42), None),
    ('GroupKFold(프레임)', GroupKFold(n_splits=5), cvd['group_frame']),
    ('GroupKFold(영상)',   GroupKFold(n_splits=4), cvd['video_id']),
]
store = {}
for name, sp, g in plans_cv:
    proba = oof_proba(sp, groups=g)
    store[name] = proba
    pred = (proba >= 0.5).astype(int)
    f1_each = pd.Series(f1_score(Y_cv, pred, average=None, zero_division=0),
                        index=LABELS)
    print(f"[{name}]")
    print("   클래스별 F1 :", f1_each.round(3).to_dict())
    print(f"   macro F1 = {f1_score(Y_cv, pred, average='macro', zero_division=0):.3f}"
          f"   micro F1 = {f1_score(Y_cv, pred, average='micro', zero_division=0):.3f}")

ones = np.ones(Y_cv.shape, dtype=int)
print("[바닥선] 모든 패치를 '결함 있음'으로 찍기")
print("   클래스별 F1 :",
      pd.Series(f1_score(Y_cv, ones, average=None, zero_division=0),
                index=LABELS).round(3).to_dict())
print(f"   macro F1 = {f1_score(Y_cv, ones, average='macro', zero_division=0):.3f}"
      f"   micro F1 = {f1_score(Y_cv, ones, average='micro', zero_division=0):.3f}")

proba_v = store['GroupKFold(영상)']
for j, lab in enumerate(LABELS):
    ap = average_precision_score(Y_cv[:, j], proba_v[:, j])
    base = Y_cv[:, j].mean()
    print(f"{lab:15s} PR-AUC = {ap:.3f}   무작위 기준선 = {base:.3f}   "
          f"배율 = {ap / base:.2f}배")

print(cvd.pivot_table(index='coating_gap_um', columns='run',
                      values='Delamination', aggfunc='mean').round(4))
print()
print(cvd.groupby('coating_gap_um')['video_id'].nunique().to_dict())


# ====================================================================
# 왜 설명가능성이 필요한가 `필수` / 실습 준비: 데이터와 모델 `필수`
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


df = clean_data("data/battery_process_data.csv")

# 정제 확인: 결측·이상치가 처리된 상태여야 한다
print(df.shape)                       # (1000, 10)
print(df.isna().sum().sum())          # 0
print(df["건조로_1구간_온도"].max())    # 122.4 (9999 대체 완료)

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


# ====================================================================
# 불순도 기반 중요도(MDI) `필수` / 실습: feature_importances_ `필수`
# ====================================================================
mdi_reg = pd.Series(reg.feature_importances_, index=FEATS)
print(mdi_reg.sort_values(ascending=False).round(4))

mdi_clf = pd.Series(clf.feature_importances_, index=FEATS)
print(mdi_clf.sort_values(ascending=False).round(4))


# ====================================================================
# 순열 중요도 `필수` / 실습: permutation_importance `필수`
# ====================================================================
from sklearn.inspection import permutation_importance

pi_reg = permutation_importance(reg, X_test, y_test, scoring="r2",
                                n_repeats=30, random_state=42)
pi_tab = pd.DataFrame({"중요도(R2 하락)": pi_reg.importances_mean,
                       "표준편차": pi_reg.importances_std}, index=FEATS)
print(pi_tab.sort_values("중요도(R2 하락)", ascending=False).round(4))

pi_clf = permutation_importance(clf, X_test_clf, y_test_clf, scoring="roc_auc",
                                n_repeats=30, random_state=42)
pi_tab2 = pd.DataFrame({"중요도(AUC 하락)": pi_clf.importances_mean,
                        "표준편차": pi_clf.importances_std}, index=FEATS)
print(pi_tab2.sort_values("중요도(AUC 하락)", ascending=False).round(4))


# ====================================================================
# SHAP: 기여도의 공정한 배분 `필수` / 실습: TreeExplainer와 전역 요약 `필수`
# ====================================================================
import shap   # pip install shap (실습 환경에는 설치되어 있음, v0.52 기준)

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_clf)

# 이진 분류에서 shap_values는 (표본, 변수, 클래스) 배열로 반환된다.
# 불량(클래스 1) 관점의 값만 취한다.
sv = shap_values[..., 1]
print(sv.shape)                        # (200, 6)
print(explainer.expected_value[1])     # 0.5023541666666663 (기준값: 평균 예측 확률)

# (1) bar plot: 변수별 평균 |SHAP| — 전역 중요도 순위
shap.summary_plot(sv, X_test_clf, plot_type="bar")

# (2) summary plot(beeswarm): 표본 하나가 점 하나
shap.summary_plot(sv, X_test_clf)

# 수치로도 확인
print(pd.Series(np.abs(sv).mean(axis=0), index=FEATS)
        .sort_values(ascending=False).round(4))


# ====================================================================
# SHAP: 기여도의 공정한 배분 `필수` / 개별 로트 해석: 이 로트는 왜 불량 고위험인가 `필수`
# ====================================================================
prob = clf.predict_proba(X_test_clf)[:, 1]
i = int(np.argmax(prob))
lot_id = df.loc[X_test_clf.index[i], "Lot_ID"]
print(lot_id, f"예측 불량 확률 = {prob[i]:.3f}")   # L25246, 0.710

# waterfall: 기준값에서 예측값까지의 변수별 기여 폭포
shap.plots.waterfall(shap.Explanation(
    values=sv[i], base_values=explainer.expected_value[1],
    data=X_test_clf.iloc[i].values, feature_names=FEATS))

# force plot도 같은 정보를 가로 막대로 보여 준다
shap.plots.force(explainer.expected_value[1], sv[i], X_test_clf.iloc[i])


# ====================================================================
# 핵심 공정 인자 도출 실습 `필수` / 실제 데이터 ④ — 변수 이름을 모를 때 SHAP은 무엇을 말해 주는가 (SECOM) `선택 학습(자율 복습)` · 실습 20분
# ====================================================================
sec = pd.read_csv('data/secom/secom_merged.csv', parse_dates=['timestamp'])
print("행 수, 열 수:", sec.shape)
print("불량 수:", int(sec['fail'].sum()), f"({sec['fail'].mean():.2%})")
print("측정 기간:", sec['timestamp'].min(), "~", sec['timestamp'].max())
print(sec[['timestamp', 'label', 'fail', 'sensor_000', 'sensor_001']].head(3))

sensors = [c for c in sec.columns if c.startswith('sensor_')]
zero_var = [c for c in sensors if sec[c].nunique(dropna=True) <= 1]
too_many_na = [c for c in sensors if sec[c].isna().mean() > 0.5]
keep = [c for c in sensors if c not in zero_var and c not in too_many_na]
print("전체 센서:", len(sensors))
print("분산 0 컬럼:", len(zero_var), "| 결측 50% 초과 컬럼:", len(too_many_na))
print("남은 센서:", len(keep))
print("결측이 하나도 없는 행의 수:",
      int((sec[sensors].isna().sum(axis=1) == 0).sum()))

X_sec = sec[keep].fillna(sec[keep].median())
y_sec = sec['fail']

n_tr = int(len(sec) * 0.8)               # 데이터는 이미 시간 오름차순이다
X_tr, X_te = X_sec.iloc[:n_tr], X_sec.iloc[n_tr:]
y_tr, y_te = y_sec.iloc[:n_tr], y_sec.iloc[n_tr:]
print("학습 불량:", int(y_tr.sum()), "/", len(y_tr),
      "| 테스트 불량:", int(y_te.sum()), "/", len(y_te))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

clf_t = RandomForestClassifier(n_estimators=300, random_state=42,
                               class_weight='balanced', n_jobs=-1)
clf_t.fit(X_tr, y_tr)
p_t = clf_t.predict_proba(X_te)[:, 1]
print(f"시간순 분할  ROC-AUC={roc_auc_score(y_te, p_t):.4f}  "
      f"PR-AUC={average_precision_score(y_te, p_t):.4f}  "
      f"(PR 무작위 기준선 {y_te.mean():.4f})")

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    X_sec, y_sec, test_size=0.2, random_state=42, stratify=y_sec)
clf_r = RandomForestClassifier(n_estimators=300, random_state=42,
                               class_weight='balanced', n_jobs=-1)
clf_r.fit(Xr_tr, yr_tr)
p_r = clf_r.predict_proba(Xr_te)[:, 1]
print(f"무작위 분할  ROC-AUC={roc_auc_score(yr_te, p_r):.4f}  "
      f"PR-AUC={average_precision_score(yr_te, p_r):.4f}  "
      f"(PR 무작위 기준선 {yr_te.mean():.4f})")

print(sec.groupby(sec['timestamp'].dt.to_period('M'))['fail']
        .agg(건수='size', 불량률='mean').round(4))

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

raw = sec[keep]                                  # 결측을 채우지 않은 원본
raw_tr, raw_te = raw.iloc[:n_tr], raw.iloc[n_tr:]

pipe = make_pipeline(
    SimpleImputer(strategy='median'),            # 중앙값을 학습 구간에서만 계산
    RandomForestClassifier(n_estimators=300, random_state=42,
                           class_weight='balanced', n_jobs=-1))
pipe.fit(raw_tr, y_tr)
p_pipe = pipe.predict_proba(raw_te)[:, 1]
print(f"파이프라인(학습 구간 중앙값)  ROC-AUC={roc_auc_score(y_te, p_pipe):.4f}  "
      f"PR-AUC={average_precision_score(y_te, p_pipe):.4f}")

# 대치값 자체는 얼마나 달라졌나
med_all = sec[keep].median()
med_tr = sec[keep].iloc[:n_tr].median()
rel = ((med_all - med_tr).abs() / med_all.abs().replace(0, np.nan)).dropna()
print("중앙값이 1 % 이상 달라진 센서:", int((rel > 0.01).sum()), "/", len(rel))

import shap

expl = shap.TreeExplainer(clf_t)
sv_sec = expl.shap_values(X_te, check_additivity=False)[..., 1]   # 불량(클래스 1)
imp_sec = pd.Series(np.abs(sv_sec).mean(axis=0), index=keep)
print(imp_sec.sort_values(ascending=False).head(10).round(5))

top10 = imp_sec.nlargest(10).index
print(X_sec[top10].corrwith(y_sec).round(4))
print("446개 센서 전체에서 |상관| 최대:",
      round(float(X_sec.corrwith(y_sec).abs().max()), 4))

sets = []
for seed in range(5):
    m = RandomForestClassifier(n_estimators=300, random_state=seed,
                               class_weight='balanced', n_jobs=-1).fit(X_tr, y_tr)
    s = shap.TreeExplainer(m).shap_values(X_te, check_additivity=False)[..., 1]
    t = pd.Series(np.abs(s).mean(axis=0), index=keep).nlargest(10).index.tolist()
    sets.append(set(t))
    print(f"seed={seed}: {t}")

common = set.intersection(*sets)
union = set.union(*sets)
print("5개 시드 전부에 든 센서:", len(common), sorted(common))
print("한 번이라도 든 센서:", len(union),
      f"| 자카드 = {len(common) / len(union):.3f}")


# ====================================================================
# 핵심 공정 인자 도출 실습 `필수` / 실제 데이터 ⑤ — 정답을 아는 데이터로 SHAP을 채점한다 (WMG DoE) `필수` · 실습 12분
# ====================================================================
wmg = pd.read_csv('data/wmg/wmg_cells_54.csv')
DESIGN = ['target_coat_weight_gsm', 'target_density_g_cm3', 'roll_temp_c']
TARGET = 'grav_dis_5c_mahg'

print("셀 수:", len(wmg), "| 조건 수:", wmg['group'].nunique())
print(pd.crosstab([wmg['coat_weight_level'], wmg['density_level']],
                  wmg['roll_temp_c']))
print("\n설계인자끼리의 상관계수:")
print(wmg[DESIGN].corr().round(4).to_string())

for f in DESIGN:
    mm = wmg.groupby(f)[TARGET].mean()
    print(f"{f:24s} 수준별 평균 {mm.round(2).to_dict()}"
          f"  → 주효과 폭 {mm.max() - mm.min():.2f}")
print("전체 평균:", round(float(wmg[TARGET].mean()), 2), "mAh/g")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_score

X_wmg, y_wmg = wmg[DESIGN], wmg[TARGET]
rf_wmg = RandomForestRegressor(n_estimators=500, random_state=42)
r1 = cross_val_score(rf_wmg, X_wmg, y_wmg,
                     cv=KFold(6, shuffle=True, random_state=42), scoring='r2')
r2 = cross_val_score(rf_wmg, X_wmg, y_wmg, cv=GroupKFold(n_splits=6),
                     groups=wmg['group'], scoring='r2')
print(f"무작위 6-fold   R2 = {r1.mean():.3f} ± {r1.std():.3f}")
print(f"조건 단위 6-fold R2 = {r2.mean():.3f} ± {r2.std():.3f}")

import shap

rf_wmg.fit(X_wmg, y_wmg)
sv_wmg = shap.TreeExplainer(rf_wmg).shap_values(X_wmg)
shap_imp = pd.Series(np.abs(sv_wmg).mean(axis=0), index=DESIGN)

truth = pd.Series({f: wmg.groupby(f)[TARGET].mean()
                      .pipe(lambda s: s.max() - s.min()) for f in DESIGN})
table = pd.DataFrame({'주효과폭(정답)': truth.round(2),
                      'SHAP평균|φ|': shap_imp.round(2)})
table['정답순위'] = table['주효과폭(정답)'].rank(ascending=False).astype(int)
table['SHAP순위'] = table['SHAP평균|φ|'].rank(ascending=False).astype(int)
print(table.to_string())

print("r(코팅중량, 캘린더링 전 인장강도) =",
      round(float(wmg['target_coat_weight_gsm']
                  .corr(wmg['precal_tensile_strength_kpa'])), 4))

DESIGN2 = DESIGN + ['precal_tensile_strength_kpa']
rf2_wmg = RandomForestRegressor(n_estimators=500,
                                random_state=42).fit(wmg[DESIGN2], y_wmg)
sv2_wmg = shap.TreeExplainer(rf2_wmg).shap_values(wmg[DESIGN2])
imp2 = pd.Series(np.abs(sv2_wmg).mean(axis=0), index=DESIGN2)
print(imp2.sort_values(ascending=False).round(2))
print("두 공선 변수의 SHAP 합:",
      round(float(imp2['precal_tensile_strength_kpa']
                  + imp2['target_coat_weight_gsm']), 2),
      "| 추가 전 코팅중량 단독:", round(float(shap_imp['target_coat_weight_gsm']), 2))

print(wmg.pivot_table(index='coat_weight_level', columns='density_level',
                      values=TARGET, aggfunc='mean').round(1))


# ====================================================================
# 모델 저장과 활용 `필수` / joblib으로 저장하고 불러오기
# ====================================================================
import joblib

joblib.dump(reg, "rf_capacity.joblib")   # 용량 회귀 모델
joblib.dump(clf, "rf_defect.joblib")     # 불량 분류 모델

# 불러오기 — 학습 코드 없이 즉시 사용 가능
clf_loaded = joblib.load("rf_defect.joblib")


# ====================================================================
# 모델 저장과 활용 `필수` / 새 로트 예측 함수 만들기 `필수`
# ====================================================================
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

