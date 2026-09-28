# -*- coding: utf-8 -*-
"""5장 예제 — 5.6 모델 성능 비교 실험 — 통일 조건 벤치마크

교재 출처 : manuscript/50_ch5_평가.md
실행 방법 : example 폴더에서  python "5장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 03_5-3_분류_평가지표.py, 04_5-4_검증_전략.py, 05_5-5_과적합_진단.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd
import numpy as np

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 5.1절~5.5절에서 만든 객체를 이 파일만으로 재현한 것이다.
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
y_reg = df['최종_용량_mAh']   # 회귀 타깃
y_clf = df['불량_여부']        # 분류 타깃

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [실습 — 5모델 종합 벤치마크 `필수`]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   선형회귀: RMSE=14.93±0.77 R2=0.8039±0.0374
#   릿지(α=1): RMSE=14.93±0.77 R2=0.8039±0.0374
#   결정트리(d=6): RMSE=17.97±0.63 R2=0.7183±0.0424
#   랜덤 포레스트: RMSE=15.75±0.35 R2=0.7833±0.0314
#   XGBoost: RMSE=16.17±0.32 R2=0.7723±0.0299
#   로지스틱: F1=0.1300±0.1661 AUC=0.9843±0.0085
#   결정트리(d=6): F1=0.2899±0.1508 AUC=0.6734±0.0959
#   랜덤 포레스트: F1=0.0667±0.1333 AUC=0.9403±0.0586
#   그래디언트부스팅: F1=0.2371±0.2052 AUC=0.9150±0.0932
#   XGBoost: F1=0.1833±0.1528 AUC=0.9194±0.0921

# --------------------------------------------------------------------
# [그 차이는 우연인가 — 반복 실험의 관점 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   RF : [0.147, 0.197, 0.147, 0.1, 0.247, 0.147, 0.08, 0.18, 0.08, 0.147] | 평균 0.147 ± 0.0497
#   XGB: [0.29, 0.251, 0.187, 0.21, 0.377, 0.298, 0.273, 0.311, 0.29, 0.284] | 평균 0.2772 ± 0.0502

# --------------------------------------------------------------------
# [실제 데이터 ③ — 결함이 여러 개 동시에 나올 때: macro와 micro (CoatingVision) `선택 학습(자율 복습)` · 실습 20분]
# --------------------------------------------------------------------
cvd = pd.read_csv('data/coatingvision/coating_labels_tidy.csv')
LABELS = ['Surface_Crack', 'Pinhole', 'Delamination']

print("패치 수:", len(cvd), "| 프레임 수:", cvd['group_frame'].nunique(),
      "| 영상 수:", cvd['video_id'].nunique())
print("클래스별 양성 수:", cvd[LABELS].sum().to_dict())
print("클래스별 양성 비율:", cvd[LABELS].mean().round(3).to_dict())
print("한 패치에 붙은 결함 종류 수:",
      cvd['n_defect_types'].value_counts().sort_index().to_dict())

# 교재 실행 결과 ------------------------------------------------------
#   패치 수: 2227 | 프레임 수: 367 | 영상 수: 8
#   클래스별 양성 수: {'Surface_Crack': 1947, 'Pinhole': 519, 'Delamination': 203}
#   클래스별 양성 비율: {'Surface_Crack': 0.874, 'Pinhole': 0.233, 'Delamination': 0.091}
#   한 패치에 붙은 결함 종류 수: {0: 81, 1: 1654, 2: 461, 3: 31}

# --------------------------------------------------------------------
# [실제 데이터 ③ — 결함이 여러 개 동시에 나올 때: macro와 micro (CoatingVision) `선택 학습(자율 복습)` · 실습 20분]
# --------------------------------------------------------------------
cvd['pos_code'] = cvd['position'].astype('category').cat.codes
FEATS_CV = ['coating_gap_um', 'run', 'frame', 'patch', 'pos_code']
X_cv = cvd[FEATS_CV].to_numpy()
Y_cv = cvd[LABELS].to_numpy()
print("X_cv:", X_cv.shape, "| Y_cv:", Y_cv.shape)

# 교재 실행 결과 ------------------------------------------------------
#   X_cv: (2227, 5) | Y_cv: (2227, 3)

# --------------------------------------------------------------------
# [실제 데이터 ③ — 결함이 여러 개 동시에 나올 때: macro와 micro (CoatingVision) `선택 학습(자율 복습)` · 실습 20분]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   [무작위 KFold(5)]
#      클래스별 F1 : {'Surface_Crack': 0.984, 'Pinhole': 0.523, 'Delamination': 0.741}
#      macro F1 = 0.749   micro F1 = 0.885
#   [GroupKFold(프레임)]
#      클래스별 F1 : {'Surface_Crack': 0.982, 'Pinhole': 0.654, 'Delamination': 0.79}
#      macro F1 = 0.809   micro F1 = 0.904
#   [GroupKFold(영상)]
#      클래스별 F1 : {'Surface_Crack': 0.898, 'Pinhole': 0.27, 'Delamination': 0.111}
#      macro F1 = 0.426   micro F1 = 0.737

# --------------------------------------------------------------------
# [실제 데이터 ③ — 결함이 여러 개 동시에 나올 때: macro와 micro (CoatingVision) `선택 학습(자율 복습)` · 실습 20분]
# --------------------------------------------------------------------
ones = np.ones(Y_cv.shape, dtype=int)
print("[바닥선] 모든 패치를 '결함 있음'으로 찍기")
print("   클래스별 F1 :",
      pd.Series(f1_score(Y_cv, ones, average=None, zero_division=0),
                index=LABELS).round(3).to_dict())
print(f"   macro F1 = {f1_score(Y_cv, ones, average='macro', zero_division=0):.3f}"
      f"   micro F1 = {f1_score(Y_cv, ones, average='micro', zero_division=0):.3f}")

# 교재 실행 결과 ------------------------------------------------------
#   [바닥선] 모든 패치를 '결함 있음'으로 찍기
#      클래스별 F1 : {'Surface_Crack': 0.933, 'Pinhole': 0.378, 'Delamination': 0.167}
#      macro F1 = 0.493   micro F1 = 0.571

# --------------------------------------------------------------------
# [실제 데이터 ③ — 결함이 여러 개 동시에 나올 때: macro와 micro (CoatingVision) `선택 학습(자율 복습)` · 실습 20분]
# --------------------------------------------------------------------
proba_v = store['GroupKFold(영상)']
for j, lab in enumerate(LABELS):
    ap = average_precision_score(Y_cv[:, j], proba_v[:, j])
    base = Y_cv[:, j].mean()
    print(f"{lab:15s} PR-AUC = {ap:.3f}   무작위 기준선 = {base:.3f}   "
          f"배율 = {ap / base:.2f}배")

# 교재 실행 결과 ------------------------------------------------------
#   Surface_Crack   PR-AUC = 0.955   무작위 기준선 = 0.874   배율 = 1.09배
#   Pinhole         PR-AUC = 0.295   무작위 기준선 = 0.233   배율 = 1.27배
#   Delamination    PR-AUC = 0.262   무작위 기준선 = 0.091   배율 = 2.88배

# --------------------------------------------------------------------
# [실제 데이터 ③ — 결함이 여러 개 동시에 나올 때: macro와 micro (CoatingVision) `선택 학습(자율 복습)` · 실습 20분]
# --------------------------------------------------------------------
print(cvd.pivot_table(index='coating_gap_um', columns='run',
                      values='Delamination', aggfunc='mean').round(4))
print()
print(cvd.groupby('coating_gap_um')['video_id'].nunique().to_dict())

# 교재 실행 결과 ------------------------------------------------------
#   run                  1       7
#   coating_gap_um                
#   600             0.0000     NaN
#   700             0.0036  0.3258
#   800             0.1650     NaN
#   900             0.0164     NaN
#   1000            0.0913     NaN
#   1100            0.1226     NaN

