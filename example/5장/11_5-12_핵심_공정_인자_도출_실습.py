# -*- coding: utf-8 -*-
"""5장 예제 — 5.12 핵심 공정 인자 도출 실습 `필수`

교재 출처 : manuscript/51_ch5_중요도.md
실행 방법 : example 폴더에서  python "5장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 08_5-9_불순도_기반_중요도.py, 09_5-10_순열_중요도.py, 10_5-11_SHAP_기여도의_공정한_배분.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd
import numpy as np

# --------------------------------------------------------------------
# [실제 데이터 ④ — 변수 이름을 모를 때 SHAP은 무엇을 말해 주는가 (SECOM) `선택 학습(자율 복습)` · 실습 20분]
# --------------------------------------------------------------------
sec = pd.read_csv('data/secom/secom_merged.csv', parse_dates=['timestamp'])
print("행 수, 열 수:", sec.shape)
print("불량 수:", int(sec['fail'].sum()), f"({sec['fail'].mean():.2%})")
print("측정 기간:", sec['timestamp'].min(), "~", sec['timestamp'].max())
print(sec[['timestamp', 'label', 'fail', 'sensor_000', 'sensor_001']].head(3))

# 교재 실행 결과 ------------------------------------------------------
#   행 수, 열 수: (1567, 593)
#   불량 수: 104 (6.64%)
#   측정 기간: 2008-07-19 11:55:00 ~ 2008-10-17 06:07:00
#               timestamp  label  fail  sensor_000  sensor_001
#   0 2008-07-19 11:55:00     -1     0     3030.93     2564.00
#   1 2008-07-19 12:32:00     -1     0     3095.78     2465.14
#   2 2008-07-19 13:17:00      1     1     2932.61     2559.94

# --------------------------------------------------------------------
# [실제 데이터 ④ — 변수 이름을 모를 때 SHAP은 무엇을 말해 주는가 (SECOM) `선택 학습(자율 복습)` · 실습 20분]
# --------------------------------------------------------------------
sensors = [c for c in sec.columns if c.startswith('sensor_')]
zero_var = [c for c in sensors if sec[c].nunique(dropna=True) <= 1]
too_many_na = [c for c in sensors if sec[c].isna().mean() > 0.5]
keep = [c for c in sensors if c not in zero_var and c not in too_many_na]
print("전체 센서:", len(sensors))
print("분산 0 컬럼:", len(zero_var), "| 결측 50% 초과 컬럼:", len(too_many_na))
print("남은 센서:", len(keep))
print("결측이 하나도 없는 행의 수:",
      int((sec[sensors].isna().sum(axis=1) == 0).sum()))

# 교재 실행 결과 ------------------------------------------------------
#   전체 센서: 590
#   분산 0 컬럼: 116 | 결측 50% 초과 컬럼: 28
#   남은 센서: 446
#   결측이 하나도 없는 행의 수: 0

# --------------------------------------------------------------------
# [실제 데이터 ④ — 변수 이름을 모를 때 SHAP은 무엇을 말해 주는가 (SECOM) `선택 학습(자율 복습)` · 실습 20분]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   학습 불량: 87 / 1253 | 테스트 불량: 17 / 314
#   시간순 분할  ROC-AUC=0.5224  PR-AUC=0.0668  (PR 무작위 기준선 0.0541)
#   무작위 분할  ROC-AUC=0.7622  PR-AUC=0.2232  (PR 무작위 기준선 0.0669)

# --------------------------------------------------------------------
# [실제 데이터 ④ — 변수 이름을 모를 때 SHAP은 무엇을 말해 주는가 (SECOM) `선택 학습(자율 복습)` · 실습 20분]
# --------------------------------------------------------------------
print(sec.groupby(sec['timestamp'].dt.to_period('M'))['fail']
        .agg(건수='size', 불량률='mean').round(4))

# 교재 실행 결과 ------------------------------------------------------
#               건수     불량률
#   timestamp             
#   2008-07     63  0.2222
#   2008-08    555  0.0919
#   2008-09    590  0.0288
#   2008-10    359  0.0613

# --------------------------------------------------------------------
# [실제 데이터 ④ — 변수 이름을 모를 때 SHAP은 무엇을 말해 주는가 (SECOM) `선택 학습(자율 복습)` · 실습 20분]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   파이프라인(학습 구간 중앙값)  ROC-AUC=0.5250  PR-AUC=0.0674
#   중앙값이 1 % 이상 달라진 센서: 237 / 423

# --------------------------------------------------------------------
# [실제 데이터 ④ — 변수 이름을 모를 때 SHAP은 무엇을 말해 주는가 (SECOM) `선택 학습(자율 복습)` · 실습 20분]
# --------------------------------------------------------------------
import shap

expl = shap.TreeExplainer(clf_t)
sv_sec = expl.shap_values(X_te, check_additivity=False)[..., 1]   # 불량(클래스 1)
imp_sec = pd.Series(np.abs(sv_sec).mean(axis=0), index=keep)
print(imp_sec.sort_values(ascending=False).head(10).round(5))

# 교재 실행 결과 ------------------------------------------------------
#   sensor_059    0.02311
#   sensor_460    0.00922
#   sensor_033    0.00852
#   sensor_065    0.00672
#   sensor_064    0.00665
#   sensor_103    0.00620
#   sensor_341    0.00578
#   sensor_333    0.00560
#   sensor_290    0.00537
#   sensor_477    0.00531
#   dtype: float64

# --------------------------------------------------------------------
# [실제 데이터 ④ — 변수 이름을 모를 때 SHAP은 무엇을 말해 주는가 (SECOM) `선택 학습(자율 복습)` · 실습 20분]
# --------------------------------------------------------------------
top10 = imp_sec.nlargest(10).index
print(X_sec[top10].corrwith(y_sec).round(4))
print("446개 센서 전체에서 |상관| 최대:",
      round(float(X_sec.corrwith(y_sec).abs().max()), 4))

# 교재 실행 결과 ------------------------------------------------------
#   sensor_059    0.1560
#   sensor_460    0.0606
#   sensor_033    0.0810
#   sensor_065    0.0550
#   sensor_064    0.0767
#   sensor_103    0.1512
#   sensor_341    0.0498
#   sensor_333    0.0488
#   sensor_290   -0.0143
#   sensor_477    0.0563
#   dtype: float64
#   446개 센서 전체에서 |상관| 최대: 0.156

# --------------------------------------------------------------------
# [실제 데이터 ④ — 변수 이름을 모를 때 SHAP은 무엇을 말해 주는가 (SECOM) `선택 학습(자율 복습)` · 실습 20분]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   seed=0: ['sensor_059', 'sensor_033', 'sensor_460', 'sensor_103', 'sensor_477', 'sensor_290', 'sensor_121', 'sensor_130', 'sensor_341', 'sensor_182']
#   seed=1: ['sensor_059', 'sensor_033', 'sensor_460', 'sensor_290', 'sensor_341', 'sensor_130', 'sensor_152', 'sensor_103', 'sensor_333', 'sensor_028']
#   seed=2: ['sensor_059', 'sensor_033', 'sensor_103', 'sensor_477', 'sensor_460', 'sensor_287', 'sensor_290', 'sensor_468', 'sensor_065', 'sensor_000']
#   seed=3: ['sensor_059', 'sensor_033', 'sensor_460', 'sensor_290', 'sensor_121', 'sensor_152', 'sensor_341', 'sensor_477', 'sensor_065', 'sensor_468']
#   seed=4: ['sensor_059', 'sensor_033', 'sensor_460', 'sensor_103', 'sensor_130', 'sensor_028', 'sensor_341', 'sensor_477', 'sensor_152', 'sensor_091']
#   5개 시드 전부에 든 센서: 3 ['sensor_033', 'sensor_059', 'sensor_460']
#   한 번이라도 든 센서: 18 | 자카드 = 0.167

# --------------------------------------------------------------------
# [실제 데이터 ⑤ — 정답을 아는 데이터로 SHAP을 채점한다 (WMG DoE) `필수` · 실습 12분]
# --------------------------------------------------------------------
wmg = pd.read_csv('data/wmg/wmg_cells_54.csv')
DESIGN = ['target_coat_weight_gsm', 'target_density_g_cm3', 'roll_temp_c']
TARGET = 'grav_dis_5c_mahg'

print("셀 수:", len(wmg), "| 조건 수:", wmg['group'].nunique())
print(pd.crosstab([wmg['coat_weight_level'], wmg['density_level']],
                  wmg['roll_temp_c']))
print("\n설계인자끼리의 상관계수:")
print(wmg[DESIGN].corr().round(4).to_string())

# 교재 실행 결과 ------------------------------------------------------
#   셀 수: 54 | 조건 수: 18
#   roll_temp_c                      85.0   120.0  145.0
#   coat_weight_level density_level                     
#   H                 D                  3      3      3
#                     M                  3      3      3
#                     P                  3      3      3
#   L                 D                  3      3      3
#                     M                  3      3      3
#                     P                  3      3      3
#   설계인자끼리의 상관계수:
#                           target_coat_weight_gsm  target_density_g_cm3  roll_temp_c
#   target_coat_weight_gsm                     1.0                   0.0          0.0
#   target_density_g_cm3                       0.0                   1.0          0.0
#   roll_temp_c                                0.0                   0.0          1.0

# --------------------------------------------------------------------
# [실제 데이터 ⑤ — 정답을 아는 데이터로 SHAP을 채점한다 (WMG DoE) `필수` · 실습 12분]
# --------------------------------------------------------------------
for f in DESIGN:
    mm = wmg.groupby(f)[TARGET].mean()
    print(f"{f:24s} 수준별 평균 {mm.round(2).to_dict()}"
          f"  → 주효과 폭 {mm.max() - mm.min():.2f}")
print("전체 평균:", round(float(wmg[TARGET].mean()), 2), "mAh/g")

# 교재 실행 결과 ------------------------------------------------------
#   target_coat_weight_gsm   수준별 평균 {122.48: 123.97, 182.73: 51.02}  → 주효과 폭 72.95
#   target_density_g_cm3     수준별 평균 {2.7: 73.96, 2.95: 98.77, 3.2: 89.76}  → 주효과 폭 24.81
#   roll_temp_c              수준별 평균 {85.0: 87.38, 120.0: 91.22, 145.0: 83.88}  → 주효과 폭 7.34
#   전체 평균: 87.49 mAh/g

# --------------------------------------------------------------------
# [실제 데이터 ⑤ — 정답을 아는 데이터로 SHAP을 채점한다 (WMG DoE) `필수` · 실습 12분]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   무작위 6-fold   R2 = 0.951 ± 0.028
#   조건 단위 6-fold R2 = 0.822 ± 0.104

# --------------------------------------------------------------------
# [실제 데이터 ⑤ — 정답을 아는 데이터로 SHAP을 채점한다 (WMG DoE) `필수` · 실습 12분]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#                           주효과폭(정답)  SHAP평균|φ|  정답순위  SHAP순위
#   target_coat_weight_gsm     72.95      36.45     1       1
#   target_density_g_cm3       24.81       8.93     2       2
#   roll_temp_c                 7.34       3.08     3       3

# --------------------------------------------------------------------
# [실제 데이터 ⑤ — 정답을 아는 데이터로 SHAP을 채점한다 (WMG DoE) `필수` · 실습 12분]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   r(코팅중량, 캘린더링 전 인장강도) = -1.0
#   precal_tensile_strength_kpa    19.98
#   target_coat_weight_gsm         16.47
#   target_density_g_cm3            8.93
#   roll_temp_c                     3.08
#   dtype: float64
#   두 공선 변수의 SHAP 합: 36.45 | 추가 전 코팅중량 단독: 36.45

# --------------------------------------------------------------------
# [실제 데이터 ⑤ — 정답을 아는 데이터로 SHAP을 채점한다 (WMG DoE) `필수` · 실습 12분]
# --------------------------------------------------------------------
print(wmg.pivot_table(index='coat_weight_level', columns='density_level',
                      values=TARGET, aggfunc='mean').round(1))

# 교재 실행 결과 ------------------------------------------------------
#   density_level          D      M      P
#   coat_weight_level                     
#   H                   51.6   71.7   29.8
#   L                  127.9  125.9  118.1

