# -*- coding: utf-8 -*-
"""5장 예제 — 5.4 검증 전략 — 공정한 시험 설계 `필수`

교재 출처 : manuscript/50_ch5_평가.md
실행 방법 : example 폴더에서  python "5장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 01_5-1_왜_평가가_모델링보다_중요한가.py, 02_5-2_회귀_평가지표.py, 03_5-3_분류_평가지표.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 5.1절·5.3절에서 만든 객체를 이 파일만으로 재현한 것이다.
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

from sklearn.ensemble import RandomForestClassifier
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [Stratified K-fold — 불균형 분류의 기본값]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   KFold fold별 불량 수: [4, 5, 1, 3, 6]
#   Stratified fold별 불량 수: [3, 4, 4, 4, 4]
#   Stratified 5-fold F1: [0.0, 0.3333, 0.0, 0.0, 0.0] | 평균 0.0667 | 표준편차 0.1333

# --------------------------------------------------------------------
# [시계열 데이터 — 미래를 훔쳐보게 하지 마라]
# --------------------------------------------------------------------
from sklearn.model_selection import TimeSeriesSplit
tss = TimeSeriesSplit(n_splits=5)
for i, (tr, te) in enumerate(tss.split(X), 1):
    print(f"fold {i}: 훈련 [{tr.min()}..{tr.max()}] → 테스트 [{te.min()}..{te.max()}]")

# 교재 실행 결과 ------------------------------------------------------
#   fold 1: 훈련 [0..169] → 테스트 [170..335]
#   fold 2: 훈련 [0..335] → 테스트 [336..501]
#   fold 3: 훈련 [0..501] → 테스트 [502..667]
#   fold 4: 훈련 [0..667] → 테스트 [668..833]
#   fold 5: 훈련 [0..833] → 테스트 [834..999]

# --------------------------------------------------------------------
# [실제 데이터 ① — "GroupKFold를 썼다"는 답이 아니다 (EIS) `필수` · 실습 15분]
# --------------------------------------------------------------------
import pandas as pd
import numpy as np

eis = pd.read_csv('data/eis/eis_wide.csv')
print("표본 수, 열 수:", eis.shape)
print("셀 목록:", sorted(eis['battery_id'].unique()))
print("셀별 표본 수:", eis['battery_id'].value_counts().sort_index().to_dict())
print("측정 회차(measure_id) 수:", eis['measure_id'].nunique())
print(eis[['sample_id', 'battery_id', 'repeat_id', 'soc',
           'Zre_0.05Hz', 'Zre_1000Hz']].head(3))

# 교재 실행 결과 ------------------------------------------------------
#   표본 수, 열 수: (240, 33)
#   셀 목록: ['B02', 'B03', 'B05', 'B06']
#   셀별 표본 수: {'B02': 60, 'B03': 60, 'B05': 60, 'B06': 60}
#   측정 회차(measure_id) 수: 24
#        sample_id battery_id  repeat_id  soc  Zre_0.05Hz  Zre_1000Hz
#   0  02_4_soc010        B02          4   10    0.117933    0.081658
#   1  02_4_soc020        B02          4   20    0.108914    0.080611
#   2  02_4_soc030        B02          4   30    0.105084    0.079936

# --------------------------------------------------------------------
# [실제 데이터 ① — "GroupKFold를 썼다"는 답이 아니다 (EIS) `필수` · 실습 15분]
# --------------------------------------------------------------------
ID_COLS = ['sample_id', 'measure_id', 'battery_id', 'repeat_id', 'soc']
X_eis = eis.drop(columns=ID_COLS)   # 피처 28개 = 주파수 14점 x (실수부, 허수부)
y_eis = eis['soc']                  # 타깃: 충전 상태 10~100 %
print("X_eis:", X_eis.shape, "| y_eis 범위:", y_eis.min(), "~", y_eis.max())

# 교재 실행 결과 ------------------------------------------------------
#   X_eis: (240, 28) | y_eis 범위: 10 ~ 100

# --------------------------------------------------------------------
# [실제 데이터 ① — "GroupKFold를 썼다"는 답이 아니다 (EIS) `필수` · 실습 15분]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   ① 무작위 KFold(4)             평균 MAE= 4.12 %SOC | fold별 [4.24, 4.08, 4.03, 4.13] | 표준편차 0.08
#   ② GroupKFold(measure_id)   평균 MAE= 3.71 %SOC | fold별 [2.83, 4.26, 2.35, 5.41] | 표준편차 1.20
#   ③ GroupKFold(battery_id)   평균 MAE= 8.73 %SOC | fold별 [14.31, 6.07, 3.84, 10.69] | 표준편차 4.06

# --------------------------------------------------------------------
# [실제 데이터 ① — "GroupKFold를 썼다"는 답이 아니다 (EIS) `필수` · 실습 15분]
# --------------------------------------------------------------------
print(eis.groupby('battery_id')['Zre_1000Hz'].mean().round(6))

# 교재 실행 결과 ------------------------------------------------------
#   battery_id
#   B02    0.079334
#   B03    0.080733
#   B05    0.081270
#   B06    0.083345
#   Name: Zre_1000Hz, dtype: float64

# --------------------------------------------------------------------
# [실제 데이터 ① — "GroupKFold를 썼다"는 답이 아니다 (EIS) `필수` · 실습 15분]
# --------------------------------------------------------------------
gkf = GroupKFold(n_splits=4)
for i, (tr, te) in enumerate(gkf.split(X_eis, y_eis, groups=eis['battery_id']), 1):
    test_cell = eis['battery_id'].iloc[te].unique()[0]
    rf_eis.fit(X_eis.iloc[tr], y_eis.iloc[tr])
    e = np.abs(rf_eis.predict(X_eis.iloc[te]) - y_eis.iloc[te]).mean()
    print(f"fold {i}: 테스트 셀 = {test_cell} (n={len(te)}) → MAE = {e:.2f} %SOC")

# 교재 실행 결과 ------------------------------------------------------
#   fold 1: 테스트 셀 = B06 (n=60) → MAE = 14.31 %SOC
#   fold 2: 테스트 셀 = B05 (n=60) → MAE = 6.07 %SOC
#   fold 3: 테스트 셀 = B03 (n=60) → MAE = 3.84 %SOC
#   fold 4: 테스트 셀 = B02 (n=60) → MAE = 10.69 %SOC

# --------------------------------------------------------------------
# [실제 데이터 ① — "GroupKFold를 썼다"는 답이 아니다 (EIS) `필수` · 실습 15분]
# --------------------------------------------------------------------
from sklearn.model_selection import cross_val_predict

pred_eis = cross_val_predict(rf_eis, X_eis, y_eis, cv=GroupKFold(n_splits=4),
                             groups=eis['battery_id'])
err_eis = pd.DataFrame({'soc': y_eis, 'abs_err': np.abs(pred_eis - y_eis)})
print(err_eis.groupby('soc')['abs_err'].mean().round(2))
print("저SOC(10~30) 평균 :",
      round(err_eis.loc[err_eis['soc'] <= 30, 'abs_err'].mean(), 2))
print("중·고SOC(40~100)  :",
      round(err_eis.loc[err_eis['soc'] >= 40, 'abs_err'].mean(), 2))

# 교재 실행 결과 ------------------------------------------------------
#   soc
#   10      3.98
#   20      4.37
#   30      5.04
#   40      4.95
#   50      9.59
#   60      5.02
#   70      7.50
#   80     11.18
#   90     13.04
#   100    22.58
#   Name: abs_err, dtype: float64
#   저SOC(10~30) 평균 : 4.46
#   중·고SOC(40~100)  : 10.55

# --------------------------------------------------------------------
# [실제 데이터 ② — 셀 단위로 나누면 드러나는 진짜 성능 (Battery RUL) `선택 학습(자율 복습)` · 실습 10분]
# --------------------------------------------------------------------
rul = pd.read_csv('data/rul/battery_rul_clean.csv')
print("행 수, 열 수:", rul.shape)
print("셀 수:", rul['cell_id'].nunique())
print("셀별 행 수:", rul['cell_id'].value_counts().sort_index().tolist())
print(rul[['cell_id', 'cycle_index', 'max_volt_discharge_v', 'rul']].head(3))

# 교재 실행 결과 ------------------------------------------------------
#   행 수, 열 수: (14996, 10)
#   셀 수: 14
#   셀별 행 수: [1074, 1077, 1074, 1078, 1072, 1075, 1079, 1075, 1076, 1077, 1071, 1070, 1064, 1034]
#      cell_id  cycle_index  max_volt_discharge_v   rul
#   0        1          1.0                 3.670  1112
#   1        1          2.0                 4.246  1111
#   2        1          3.0                 4.249  1110

# --------------------------------------------------------------------
# [실제 데이터 ② — 셀 단위로 나누면 드러나는 진짜 성능 (Battery RUL) `선택 학습(자율 복습)` · 실습 10분]
# --------------------------------------------------------------------
ok = []
for cid, d in rul.groupby('cell_id'):
    ok.append(bool((d['rul'] == d['cycle_index'].max() - d['cycle_index']).all()))
print("rul == (셀별 마지막 사이클 - 현재 사이클) 이 성립하는 셀:",
      sum(ok), "/", len(ok))
print("상관계수 r(cycle_index, rul) =",
      round(float(rul['cycle_index'].corr(rul['rul'])), 6))

# 교재 실행 결과 ------------------------------------------------------
#   rul == (셀별 마지막 사이클 - 현재 사이클) 이 성립하는 셀: 13 / 14
#   상관계수 r(cycle_index, rul) = -0.999756

# --------------------------------------------------------------------
# [실제 데이터 ② — 셀 단위로 나누면 드러나는 진짜 성능 (Battery RUL) `선택 학습(자율 복습)` · 실습 10분]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   [cycle_index 포함 (8특징)]
#      무작위 5-fold  MAE =   2.01 ± 0.06 사이클
#      셀 단위 5-fold MAE =   5.32 ± 2.12 사이클  fold별 [9.2, 3.7, 5.9, 3.4, 4.4]
#   [cycle_index 제외 (7특징)]
#      무작위 5-fold  MAE =   9.75 ± 0.53 사이클
#      셀 단위 5-fold MAE =  31.62 ± 5.42 사이클  fold별 [34.6, 36.5, 24.1, 36.9, 26.0]

