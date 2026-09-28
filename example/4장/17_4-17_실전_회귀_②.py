# -*- coding: utf-8 -*-
"""4장 예제 — 4.17 실전 회귀 ② — 타깃 누수 2단 시연 (Battery RUL) `필수` 〔실습 30분〕

교재 출처 : manuscript/41_ch4_분류.md
실행 방법 : example 폴더에서  python "4장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 14_4-14_종합_불량_분류_파이프라인_완성본.py, 15_4-15_연습문제.py, 16_4-16_실전_회귀_①.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd

# --------------------------------------------------------------------
# [이번에 다루는 데이터]
# --------------------------------------------------------------------
rul = pd.read_csv('data/rul/battery_rul_clean.csv')

print(rul.shape)
print(rul.columns.tolist())
print(rul.head(3).to_string(index=False))

# 교재 실행 결과 ------------------------------------------------------
#   (14996, 10)
#   ['cell_id', 'cycle_index', 'discharge_time_s', 'decrement_36_34v_s', 'max_volt_discharge_v', 'min_volt_charge_v', 'time_at_415v_s', 'time_cc_s', 'charging_time_s', 'rul']
#    cell_id  cycle_index  discharge_time_s  decrement_36_34v_s  max_volt_discharge_v  min_volt_charge_v  time_at_415v_s  time_cc_s  charging_time_s  rul
#          1          1.0           2595.30           1151.4885                 3.670              3.211        5460.001    6755.01         10777.82 1112
#          1          2.0           7408.64           1172.5125                 4.246              3.220        5508.992    6762.02         10500.35 1111
#          1          3.0           7393.76           1112.9920                 4.249              3.224        5508.993    6762.02         10420.38 1110

# --------------------------------------------------------------------
# [1단계 — 아무 생각 없이 전부 넣는다]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   입력 8개 : ['cycle_index', 'discharge_time_s', 'decrement_36_34v_s', 'max_volt_discharge_v', 'min_volt_charge_v', 'time_at_415v_s', 'time_cc_s', 'charging_time_s']
#   테스트 R²  = 0.999873
#   테스트 MAE = 1.97 사이클

# --------------------------------------------------------------------
# [범인을 찾는다 — 특성 중요도부터]
# --------------------------------------------------------------------
imp = pd.Series(rf.feature_importances_, index=X_all.columns)
print(imp.sort_values(ascending=False).round(4).to_string())

# 교재 실행 결과 ------------------------------------------------------
#   cycle_index             0.9996
#   max_volt_discharge_v    0.0001
#   charging_time_s         0.0001
#   min_volt_charge_v       0.0001
#   decrement_36_34v_s      0.0001
#   discharge_time_s        0.0001
#   time_at_415v_s          0.0001
#   time_cc_s               0.0000

# --------------------------------------------------------------------
# [범인을 찾는다 — 특성 중요도부터]
# --------------------------------------------------------------------
from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(Xtr[['cycle_index']], ytr)   # 변수 딱 하나
p1 = lr.predict(Xte[['cycle_index']])

print(f"cycle_index 하나만 쓴 선형회귀 R² = {r2_score(yte, p1):.6f}")
print(f"기울기 = {lr.coef_[0]:.4f}, 절편 = {lr.intercept_:.2f}")

# 교재 실행 결과 ------------------------------------------------------
#   cycle_index 하나만 쓴 선형회귀 R² = 0.999530
#   기울기 = -0.9999, 절편 = 1110.30

# --------------------------------------------------------------------
# [범인을 찾는다 — 특성 중요도부터]
# --------------------------------------------------------------------
합 = rul['rul'] + rul['cycle_index']
확인 = 합.groupby(rul['cell_id']).nunique()      # 셀마다 서로 다른 값이 몇 개인가

print(확인.to_string())
print("셀마다 값이 하나뿐인가?", bool((확인 == 1).all()))
print()
print(합.groupby(rul['cell_id']).first().astype(int).to_string())

# 교재 실행 결과 ------------------------------------------------------
#   cell_id
#   1     1
#   2     1
#   3     1
#   4     1
#   5     1
#   6     1
#   7     1
#   8     1
#   9     1
#   10    1
#   11    1
#   12    1
#   13    1
#   14    1
#   셀마다 값이 하나뿐인가? True
#   cell_id
#   1     1113
#   2     1108
#   3     1108

# --------------------------------------------------------------------
# [2단계 — 누수 컬럼을 빼고 다시 학습한다]
# --------------------------------------------------------------------
X7 = X_all.drop(columns=['cycle_index'])          # 8개 → 7개

Xtr7, Xte7, ytr7, yte7 = train_test_split(X7, y, test_size=0.2, random_state=42)
rf7 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf7.fit(Xtr7, ytr7)
pred7 = rf7.predict(Xte7)

print(f"테스트 R²  = {r2_score(yte7, pred7):.6f}")
print(f"테스트 MAE = {mean_absolute_error(yte7, pred7):.2f} 사이클")

# 교재 실행 결과 ------------------------------------------------------
#   테스트 R²  = 0.996142
#   테스트 MAE = 9.45 사이클

# --------------------------------------------------------------------
# [3단계 — 셀 단위로 나누면 한 번 더 무너진다]
# --------------------------------------------------------------------
from sklearn.model_selection import GroupKFold, cross_val_score

gkf = GroupKFold(n_splits=14)      # 셀이 14개 → 한 번에 한 셀씩 통째로 평가

for tag, X in [('cycle_index 포함(8특징)', X_all), ('cycle_index 제외(7특징)', X7)]:
    mae = -cross_val_score(RandomForestRegressor(n_estimators=200,
                                                 random_state=42, n_jobs=-1),
                           X, y, cv=gkf, groups=rul['cell_id'],
                           scoring='neg_mean_absolute_error')
    print(f"{tag:24s} MAE = {mae.mean():6.2f} ± {mae.std():5.2f} "
          f"(최악 셀 {mae.max():.2f})")

# 교재 실행 결과 ------------------------------------------------------
#   cycle_index 포함(8특징)      MAE =   5.45 ±  5.73 (최악 셀 25.20)
#   cycle_index 제외(7특징)      MAE =  37.02 ± 15.11 (최악 셀 73.61)

