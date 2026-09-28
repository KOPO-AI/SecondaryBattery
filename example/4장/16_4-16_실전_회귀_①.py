# -*- coding: utf-8 -*-
"""4장 예제 — 4.16 실전 회귀 ① — 소표본 DoE 데이터 (WMG 캘린더링 54셀) `필수` 〔실습 45분〕

교재 출처 : manuscript/41_ch4_분류.md
실행 방법 : example 폴더에서  python "4장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 13_4-13_확률_예측과_임계값.py, 14_4-14_종합_불량_분류_파이프라인_완성본.py, 15_4-15_연습문제.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --------------------------------------------------------------------
# [이번에 다루는 데이터]
# --------------------------------------------------------------------
import pandas as pd
import numpy as np

wmg = pd.read_csv('data/wmg/wmg_cells_54.csv')

print(wmg.shape)
print(wmg[['cell_id', 'group', 'replicate', 'coat_weight_level',
           'density_level', 'grav_dis_5c_mahg']].head(6).to_string(index=False))

# 교재 실행 결과 ------------------------------------------------------
#   (54, 58)
#   cell_id  group  replicate coat_weight_level density_level  grav_dis_5c_mahg
#     DD001      1          1                 L             P        124.586033
#     DD002      1          2                 L             P        124.280444
#     DD056      1          3                 L             P        122.635007
#     DD032      2          1                 L             M        126.891526
#     DD033      2          2                 L             M        126.963216
#     DD034      2          3                 L             M        125.784610

# --------------------------------------------------------------------
# [표본 54개는 실험 54번이 아니다]
# --------------------------------------------------------------------
# 조건(group)마다 셀이 몇 개인가
print(wmg.groupby('group').size().value_counts())

# 설계가 균형인지 교차표로 확인
print(wmg.pivot_table(index='coat_weight_level', columns='density_level',
                      values='grav_dis_5c_mahg', aggfunc='size'))

# 교재 실행 결과 ------------------------------------------------------
#   3    18
#   Name: count, dtype: int64
#   density_level      D  M  P
#   coat_weight_level         
#   H                  9  9  9
#   L                  9  9  9

# --------------------------------------------------------------------
# [표본 54개는 실험 54번이 아니다]
# --------------------------------------------------------------------
print(wmg['grav_dis_5c_mahg'].describe().round(2))

# 교재 실행 결과 ------------------------------------------------------
#   count     54.00
#   mean      87.49
#   std       39.89
#   min        4.81
#   25%       52.17
#   50%       97.76
#   75%      125.28
#   max      135.35
#   Name: grav_dis_5c_mahg, dtype: float64

# --------------------------------------------------------------------
# [4.2~4.4절 코드를 한 글자도 바꾸지 않고 이식한다]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   (43, 5) (11, 5)
#   선형회귀   테스트 R² = 0.6990  RMSE = 17.25 mAh/g  학습 R² = 0.9124
#   랜덤포레스트 테스트 R² = 0.9502  RMSE =  7.02 mAh/g  학습 R² = 0.9769

# --------------------------------------------------------------------
# [첫 번째 의심 — 시드를 열 번 바꿔 보라]
# --------------------------------------------------------------------
for rs in range(10):
    a, b, c, d = train_test_split(Xw, yw, test_size=0.2, random_state=rs)
    s_lr = r2_score(d, LinearRegression().fit(a, c).predict(b))
    s_rf = r2_score(d, RandomForestRegressor(n_estimators=200, random_state=42,
                                             n_jobs=-1).fit(a, c).predict(b))
    print(f"random_state={rs}  선형 R²={s_lr:.4f}   랜덤포레스트 R²={s_rf:.4f}")

# 교재 실행 결과 ------------------------------------------------------
#   random_state=0  선형 R²=0.6419   랜덤포레스트 R²=0.9240
#   random_state=1  선형 R²=0.4065   랜덤포레스트 R²=0.7664
#   random_state=2  선형 R²=0.7604   랜덤포레스트 R²=0.9491
#   random_state=3  선형 R²=0.9043   랜덤포레스트 R²=0.9863
#   random_state=4  선형 R²=0.8546   랜덤포레스트 R²=0.9345
#   random_state=5  선형 R²=0.9026   랜덤포레스트 R²=0.9819
#   random_state=6  선형 R²=0.9072   랜덤포레스트 R²=0.9645
#   random_state=7  선형 R²=0.7387   랜덤포레스트 R²=0.9411
#   random_state=8  선형 R²=0.6828   랜덤포레스트 R²=0.8613
#   random_state=9  선형 R²=0.7974   랜덤포레스트 R²=0.9690

# --------------------------------------------------------------------
# [두 번째 의심 — 테스트셋에 무엇이 들어 있었나]
# --------------------------------------------------------------------
train_groups = set(wmg.loc[Xw_tr.index, 'group'])   # 학습셋이 담고 있는 조건 번호
test_groups = wmg.loc[Xw_te.index, 'group']         # 테스트 셀들의 조건 번호

print("학습셋이 담고 있는 조건 수 :", len(train_groups), "/ 18")
print("테스트 셀 11개의 조건 번호 :", sorted(test_groups.tolist()))
print("이 중 학습셋에도 있는 조건 :", int(test_groups.isin(train_groups).sum()), "개")

# 교재 실행 결과 ------------------------------------------------------
#   학습셋이 담고 있는 조건 수 : 18 / 18
#   테스트 셀 11개의 조건 번호 : [2, 2, 5, 5, 6, 7, 11, 15, 17, 17, 18]
#   이 중 학습셋에도 있는 조건 : 11 개

# --------------------------------------------------------------------
# [두 번째 의심 — 테스트셋에 무엇이 들어 있었나]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   선형회귀   셀 단위 KFold(5)     RMSE = 14.91 ±  1.97
#   선형회귀   조건 단위 GroupKFold RMSE = 17.80 ±  8.55
#   랜덤포레스트 셀 단위 KFold(5)     RMSE =  8.29 ±  1.85
#   랜덤포레스트 조건 단위 GroupKFold RMSE = 18.44 ± 12.46

# --------------------------------------------------------------------
# [두 번째 의심 — 테스트셋에 무엇이 들어 있었나]
# --------------------------------------------------------------------
b = -cross_val_score(RandomForestRegressor(n_estimators=200, random_state=42,
                                           n_jobs=-1),
                     Xw, yw, cv=gkf, groups=groups,
                     scoring='neg_root_mean_squared_error')
print("GroupKFold fold별 RMSE:", b.round(2))

# 교재 실행 결과 ------------------------------------------------------
#   GroupKFold fold별 RMSE: [16.59 30.85 39.02  6.7   5.27 12.19]

# --------------------------------------------------------------------
# [교호작용 — 두 인자가 서로의 효과를 바꾼다]
# --------------------------------------------------------------------
print(wmg.pivot_table(index='coat_weight_level', columns='density_level',
                      values='grav_dis_5c_mahg',
                      aggfunc='mean')[['P', 'M', 'D']].round(1))

# 교재 실행 결과 ------------------------------------------------------
#   density_level          P      M      D
#   coat_weight_level
#   H                   29.8   71.7   51.6
#   L                  118.1  125.9  127.9

# --------------------------------------------------------------------
# [소표본에서는 변수를 줄여야 이긴다]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   ① 설계인자 5개    선형 17.80 ±  8.55   랜덤포레스트 18.44 ± 12.46
#   ② 결과변수 제외    선형 17.18 ±  8.05   랜덤포레스트 14.21 ±  6.65
#   ③ 핵심 2개      선형 17.09 ±  7.83   랜덤포레스트 12.17 ±  6.19
#   ④ 중량 1개      선형 15.98 ±  9.27   랜덤포레스트 16.00 ±  9.25

# --------------------------------------------------------------------
# [여기가 한계다 — 재현성이 정하는 성능 상한]
# --------------------------------------------------------------------
반복SD = wmg.groupby('group')['grav_dis_5c_mahg'].std()

print(f"조건 안 반복 3셀의 SD  중앙값 = {반복SD.median():5.2f} mAh/g")
print(f"                      최소 = {반복SD.min():5.2f} mAh/g")
print(f"                      최대 = {반복SD.max():5.2f} mAh/g")
print(f"54셀 전체 SD                = {yw.std():5.2f} mAh/g")

# 교재 실행 결과 ------------------------------------------------------
#   조건 안 반복 3셀의 SD  중앙값 =  2.26 mAh/g
#                         최소 =  0.66 mAh/g
#                         최대 = 22.21 mAh/g
#   54셀 전체 SD                = 39.89 mAh/g

# --------------------------------------------------------------------
# [여기가 한계다 — 재현성이 정하는 성능 상한]
# --------------------------------------------------------------------
재현성 = pd.DataFrame({'조건내SD': 반복SD.round(2),
                    '중량': wmg.groupby('group')['coat_weight_level'].first(),
                    '밀도': wmg.groupby('group')['density_level'].first(),
                    '평균용량': wmg.groupby('group')['grav_dis_5c_mahg'].mean().round(1)})
print(재현성.sort_values('조건내SD').to_string())

# 교재 실행 결과 ------------------------------------------------------
#          조건내SD 중량 밀도   평균용량
#   group
#   2       0.66  L  M  126.5
#   9       0.79  L  D  125.6
#   6       1.05  L  D  128.2
#   1       1.05  L  P  123.8
#   18      1.53  H  D   45.6
#   5       1.64  L  M  125.2
#   7       2.04  L  P  110.4
#   4       2.10  L  P  120.1
#   8       2.14  L  M  125.8
#   12      2.38  H  D   49.2
#   3       4.59  L  D  130.1
#   17      6.21  H  M   68.0
#   16      6.87  H  P   27.9
#   15      6.90  H  D   59.9
#   11      7.23  H  M   78.9
#   14      7.65  H  M   68.2
#   10     12.06  H  P   15.8
#   13     22.21  H  P   45.6

# --------------------------------------------------------------------
# [여기가 한계다 — 재현성이 정하는 성능 상한]
# --------------------------------------------------------------------
print(재현성.groupby('중량')['조건내SD'].agg(['min', 'median', 'max']).round(2))

# 교재 실행 결과 ------------------------------------------------------
#        min  median    max
#   중량
#   H   1.53    6.90  22.21
#   L   0.66    1.64   4.59

# --------------------------------------------------------------------
# [여기가 한계다 — 재현성이 정하는 성능 상한]
# --------------------------------------------------------------------
편차 = wmg['grav_dis_5c_mahg'] - wmg.groupby('group')['grav_dis_5c_mahg'].transform('mean')

풀드SD = np.sqrt((편차 ** 2).sum() / (len(wmg) - wmg['group'].nunique()))
설명비율 = 1 - (편차 ** 2).sum() / ((yw - yw.mean()) ** 2).sum()

print(f"풀드 조건내 표준편차     = {풀드SD:.2f} mAh/g   ← RMSE의 하한")
print(f"조건이 설명하는 분산 비율 = {설명비율*100:.1f}%")

# 교재 실행 결과 ------------------------------------------------------
#   풀드 조건내 표준편차     = 7.19 mAh/g   ← RMSE의 하한
#   조건이 설명하는 분산 비율 = 97.8%

# --------------------------------------------------------------------
# [여기가 한계다 — 재현성이 정하는 성능 상한]
# --------------------------------------------------------------------
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()          # 18개 조건을 하나씩 빼며 18번 검증
X3 = wmg[['target_coat_weight_gsm', 'target_density_g_cm3']]

s = -cross_val_score(RandomForestRegressor(n_estimators=200, random_state=42,
                                           n_jobs=-1),
                     X3, yw, cv=logo, groups=groups,
                     scoring='neg_root_mean_squared_error')
print(f"LeaveOneGroupOut(18조건) RMSE = {s.mean():.2f} ± {s.std():.2f}")
print(f"  가장 잘 맞힌 조건 {s.min():.2f} / 가장 못 맞힌 조건 {s.max():.2f}")

# 교재 실행 결과 ------------------------------------------------------
#   LeaveOneGroupOut(18조건) RMSE = 8.42 ± 7.55
#     가장 잘 맞힌 조건 0.92 / 가장 못 맞힌 조건 29.95

