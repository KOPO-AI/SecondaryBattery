# -*- coding: utf-8 -*-
"""6장 예제 — 6.17 실전 공정 최적화: WMG 캘린더링 DoE (실습 45분) `선택 학습(자율 복습)`

교재 출처 : manuscript/61_ch6_최적화.md
실행 방법 : example 폴더에서  python "6장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 11_6-12_데이터_기반_공정_최적화_전략.py, 12_6-13_분석_결과_리포팅.py, 13_6-16_실전_이상탐지_배터리_RUL_사이클_데이터.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd
import numpy as np

# --------------------------------------------------------------------
# [데이터 소개 — 관찰 데이터가 아니라 '설계된 실험']
# --------------------------------------------------------------------
wmg = pd.read_csv("data/wmg/wmg_cells_54.csv")
print("행·열:", wmg.shape, "/ 조건 수:", wmg["group"].nunique())
print(pd.crosstab([wmg["coat_weight_level"], wmg["density_level"]],
                  wmg["roll_temp_c"]))

# 교재 실행 결과 ------------------------------------------------------
#   행·열: (54, 58) / 조건 수: 18
#   roll_temp_c                      85.0   120.0  145.0
#   coat_weight_level density_level
#   H                 D                  3      3      3
#                     M                  3      3      3
#                     P                  3      3      3
#   L                 D                  3      3      3
#                     M                  3      3      3
#                     P                  3      3      3

# --------------------------------------------------------------------
# [조건 평균표 — 18개 점이 말해 주는 것 (5분)]
# --------------------------------------------------------------------
Y = "grav_dis_5c_mahg"
표 = wmg.pivot_table(index=["coat_weight_level", "density_level"],
                     columns="roll_temp_c", values=Y, aggfunc="mean").round(1)
print(표)

# 교재 실행 결과 ------------------------------------------------------
#   roll_temp_c                      85.0   120.0  145.0
#   coat_weight_level density_level
#   H                 D               49.2   59.9   45.6
#                     M               78.9   68.2   68.0
#                     P               15.8   45.6   27.9
#   L                 D              130.1  128.2  125.6
#                     M              126.5  125.2  125.8
#                     P              123.8  120.1  110.4

# --------------------------------------------------------------------
# [먼저 잡음의 크기를 잰다 — 재현성 하한선 (10분)]
# --------------------------------------------------------------------
조건내SD = wmg.groupby("group")[Y].std()
잔차 = wmg[Y] - wmg.groupby("group")[Y].transform("mean")
순수오차SD = np.sqrt((잔차**2).sum() / (54 - 18))

print(f"조건 내 3반복 SD 중앙값 : {조건내SD.median():.2f} mAh/g")
print(f"조건 내 3반복 SD 최댓값 : {조건내SD.max():.2f} mAh/g")
print(f"순수오차 표준편차       : {순수오차SD:.2f} mAh/g")
print(f"전체 54셀 표준편차      : {wmg[Y].std():.2f} mAh/g")
for lev in ["L", "H"]:
    s = wmg[wmg["coat_weight_level"] == lev]
    r = s[Y] - s.groupby("group")[Y].transform("mean")
    print(f"  코팅중량 {lev} 의 순수오차 SD : {np.sqrt((r**2).sum()/(27-9)):.2f} mAh/g")

# 교재 실행 결과 ------------------------------------------------------
#   조건 내 3반복 SD 중앙값 : 2.26 mAh/g
#   조건 내 3반복 SD 최댓값 : 22.21 mAh/g
#   순수오차 표준편차       : 7.19 mAh/g
#   전체 54셀 표준편차      : 39.89 mAh/g
#     코팅중량 L 의 순수오차 SD : 2.11 mAh/g
#     코팅중량 H 의 순수오차 SD : 9.95 mAh/g

# --------------------------------------------------------------------
# [인자별 기여율 — 무엇이 지배하는가 (5분)]
# --------------------------------------------------------------------
전체평균 = wmg[Y].mean()
SS_전체 = ((wmg[Y] - 전체평균) ** 2).sum()

결과 = []
for 인자, 이름 in [("coat_weight_level", "코팅중량"),
                   ("density_level", "목표밀도"),
                   ("roll_temp_c", "롤온도")]:
    평균 = wmg.groupby(인자)[Y].mean()
    개수 = wmg.groupby(인자)[Y].size()
    SS = (개수 * (평균 - 전체평균) ** 2).sum()
    결과.append([이름, str(평균.round(1).to_dict()),
                 round(평균.max() - 평균.min(), 1), round(SS / SS_전체 * 100, 1)])
결과.append(["반복오차", "(조건 내 3반복의 산포)", "-",
             round((잔차 ** 2).sum() / SS_전체 * 100, 1)])
print(pd.DataFrame(결과, columns=["인자", "수준별 평균(mAh/g)", "범위", "기여율%"]
                   ).to_string(index=False))

# 교재 실행 결과 ------------------------------------------------------
#     인자                          수준별 평균(mAh/g)    범위  기여율%
#   코팅중량                {'H': 51.0, 'L': 124.0}  72.9  85.2
#   목표밀도      {'D': 89.8, 'M': 98.8, 'P': 74.0}  24.8   6.7
#    롤온도 {85.0: 87.4, 120.0: 91.2, 145.0: 83.9}   7.3   0.6
#   반복오차                         (조건 내 3반복의 산포)     -   2.2

# --------------------------------------------------------------------
# [반응면을 그린다 — 그리고 반드시 검증한다 (10분)]
# --------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score

X2 = ["roll_temp_c", "target_density_g_cm3"]
반응면 = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                       LinearRegression())

for lev in ["L", "H"]:
    s = wmg[wmg["coat_weight_level"] == lev]
    반응면.fit(s[X2], s[Y])
    적합 = 반응면.predict(s[X2])
    교차 = cross_val_predict(반응면, s[X2], s[Y],
                             groups=s["group"], cv=LeaveOneGroupOut())
    print(f"[코팅중량 {lev}]  적합 R² = {r2_score(s[Y], 적합):6.3f}, "
          f"RMSE = {np.sqrt(((s[Y]-적합)**2).mean()):5.2f}")
    print(f"{'':12s} 교차검증 R² = {r2_score(s[Y], 교차):6.3f}, "
          f"RMSE = {np.sqrt(((s[Y]-교차)**2).mean()):5.2f}  (조건 1개씩 빼고 검증)")

# 교재 실행 결과 ------------------------------------------------------
#   [코팅중량 L]  적합 R² =  0.762, RMSE =  2.79
#                교차검증 R² = -0.032, RMSE =  5.80  (조건 1개씩 빼고 검증)
#   [코팅중량 H]  적합 R² =  0.229, RMSE = 18.20
#                교차검증 R² = -3.304, RMSE = 42.99  (조건 1개씩 빼고 검증)

# --------------------------------------------------------------------
# [그래서 최적점은 어디인가 — 18개 점으로 말할 수 있는 것 (5분)]
# --------------------------------------------------------------------
격자온도 = np.linspace(85, 145, 61)
격자밀도 = np.linspace(2.7, 3.2, 51)
TT, DD = np.meshgrid(격자온도, 격자밀도)
격자 = pd.DataFrame({"roll_temp_c": TT.ravel(), "target_density_g_cm3": DD.ravel()})

for lev in ["L", "H"]:
    s = wmg[wmg["coat_weight_level"] == lev]
    반응면.fit(s[X2], s[Y])
    예측 = 반응면.predict(격자)
    i = int(np.argmax(예측))
    실측최고 = s.groupby("group")[Y].mean().max()
    print(f"[코팅중량 {lev}] 반응면 최댓값 {예측[i]:6.1f} mAh/g "
          f"@ 롤온도 {격자.loc[i,'roll_temp_c']:.0f}°C, "
          f"목표밀도 {격자.loc[i,'target_density_g_cm3']:.2f} g/cm³")
    print(f"{'':12s} 실제 실험한 9개 조건의 최고 평균 = {실측최고:.1f} mAh/g")

# 교재 실행 결과 ------------------------------------------------------
#   [코팅중량 L] 반응면 최댓값  129.8 mAh/g @ 롤온도 107°C, 목표밀도 3.20 g/cm³
#                실제 실험한 9개 조건의 최고 평균 = 130.1 mAh/g
#   [코팅중량 H] 반응면 최댓값   69.2 mAh/g @ 롤온도 112°C, 목표밀도 3.20 g/cm³
#                실제 실험한 9개 조건의 최고 평균 = 78.9 mAh/g

# --------------------------------------------------------------------
# [최고 조건 하나를 고를 수 있는가 — 조건 평균의 신뢰구간 (5분)]
# --------------------------------------------------------------------
from scipy import stats

MSE = (잔차 ** 2).sum() / (54 - 18)      # 순수오차 평균제곱(6.17.3절 SD의 제곱)
t = stats.t.ppf(0.975, 36)               # 자유도 36 t분포의 97.5% 분위수
반폭 = t * np.sqrt(MSE / 3)              # 조건 평균 1개의 95% 신뢰구간 반폭
LSD = t * np.sqrt(2 * MSE / 3)           # 두 조건 평균을 비교하는 최소유의차

조건평균 = (wmg.groupby(["coat_weight_level", "density_level", "roll_temp_c"])[Y]
            .mean().sort_values(ascending=False))
최고 = 조건평균.max()
동률 = 조건평균[조건평균 >= 최고 - LSD]

print(f"순수오차 평균제곱 MSE     = {MSE:.1f}  (자유도 36)")
print(f"조건 평균의 95% 신뢰구간  = ±{반폭:.1f} mAh/g")
print(f"최소유의차 LSD            = {LSD:.1f} mAh/g")
print(f"최고 조건 평균            = {최고:.1f} mAh/g  →  구분 하한 {최고-LSD:.1f}")
print(f"\n최고와 통계적으로 구분되지 않는 조건: {len(동률)}개 / 18개")
print(동률.round(1).to_string())

# 교재 실행 결과 ------------------------------------------------------
#   순수오차 평균제곱 MSE     = 51.7  (자유도 36)
#   조건 평균의 95% 신뢰구간  = ±8.4 mAh/g
#   최소유의차 LSD            = 11.9 mAh/g
#   최고 조건 평균            = 130.1 mAh/g  →  구분 하한 118.1
#   최고와 통계적으로 구분되지 않는 조건: 8개 / 18개
#   coat_weight_level  density_level  roll_temp_c
#   L                  D              85.0           130.1
#                                     120.0          128.2
#                      M              85.0           126.5
#                                     145.0          125.8
#                      D              145.0          125.6
#                      M              120.0          125.2
#                      P              85.0           123.8
#                                     120.0          120.1

# --------------------------------------------------------------------
# [최고 조건 하나를 고를 수 있는가 — 조건 평균의 신뢰구간 (5분)]
# --------------------------------------------------------------------
q = stats.studentized_range.ppf(0.95, 18, 36)   # 조건 18개를 서로 다 비교할 때
HSD = q * np.sqrt(MSE / 3)
print(f"Tukey HSD = {HSD:.1f} mAh/g  (LSD {LSD:.1f}보다 넓다)")
print("HSD 기준 동률 조건:", (조건평균 >= 최고 - HSD).sum(), "개")
print("그중 코팅중량 L:", sum(i[0] == "L" for i in 조건평균[조건평균 >= 최고 - HSD].index), "개")

# 교재 실행 결과 ------------------------------------------------------
#   Tukey HSD = 22.0 mAh/g  (LSD 11.9보다 넓다)
#   HSD 기준 동률 조건: 9 개
#   그중 코팅중량 L: 9 개

# --------------------------------------------------------------------
# [이 데이터의 함정 — 실전 데이터를 쓸 때 반드시 확인할 것 (5분)]
# --------------------------------------------------------------------
실패 = wmg[wmg["grav_dis_10c_mahg"] < 1]
print("10C 용량이 1 mAh/g 미만인 셀:", len(실패), "개")
print("결측(NaN) 개수:", wmg["grav_dis_10c_mahg"].isna().sum(),
      "→ dropna()로는 한 건도 걸러지지 않는다")
print(실패[["cell_id", "coat_weight_level", "density_level",
            "grav_dis_10c_mahg"]].to_string(index=False))

r_전체 = wmg["porosity_pct"].corr(wmg["grav_dis_10c_mahg"])
남은것 = wmg[wmg["grav_dis_10c_mahg"] >= 1]
r_제외 = 남은것["porosity_pct"].corr(남은것["grav_dis_10c_mahg"])
print(f"\n공극률 vs 10C 용량 상관계수: 전체 54셀 {r_전체:+.4f} / "
      f"실패 7셀 제외 {r_제외:+.4f}  ← 부호가 뒤집힌다")

# 교재 실행 결과 ------------------------------------------------------
#   10C 용량이 1 mAh/g 미만인 셀: 7 개
#   결측(NaN) 개수: 0 → dropna()로는 한 건도 걸러지지 않는다
#   cell_id coat_weight_level density_level  grav_dis_10c_mahg
#     DD059                 H             P           0.000887
#     DD027                 H             P           0.000892
#     DD028                 H             P           0.000890
#     DD051                 H             P           0.000895
#     DD029                 H             P           0.000893
#     DD030                 H             P           0.000891
#     DD031                 H             P           0.000892
#   공극률 vs 10C 용량 상관계수: 전체 54셀 -0.1154 / 실패 7셀 제외 +0.0812  ← 부호가 뒤집힌다

# --------------------------------------------------------------------
# [이 데이터의 함정 — 실전 데이터를 쓸 때 반드시 확인할 것 (5분)]
# --------------------------------------------------------------------
cond = pd.read_csv("data/wmg/wmg_conditions_18.csv")

print("[교락 확인] 캘린더링 날짜 × 롤 온도 교차표")
print(pd.crosstab(cond["calendering_date"], cond["roll_temp_c"]))

print("\n[공선성 확인] 18조건에 걸친 고유값 개수")
for col in ["precal_grad_carbon", "precal_tensile_strength_kpa",
            "precal_moran_carbon", "target_coat_weight_gsm"]:
    # .tolist()를 붙이지 않으면 넘파이 실수라서 np.float64(0.00654)처럼 찍힌다
    값들 = sorted(cond[col].unique().tolist())
    print(f"  {col:32s} nunique = {cond[col].nunique()}  {값들}")

# 교재 실행 결과 ------------------------------------------------------
#   [교락 확인] 캘린더링 날짜 × 롤 온도 교차표
#   roll_temp_c       85.0   120.0  145.0
#   calendering_date
#   2021-10-15            6      0      0
#   2021-10-18            0      6      0
#   2021-10-27            0      0      6
#   [공선성 확인] 18조건에 걸친 고유값 개수
#     precal_grad_carbon               nunique = 2  [0.00654, 0.01758]
#     precal_tensile_strength_kpa      nunique = 2  [683.25, 728.62]
#     precal_moran_carbon              nunique = 2  [0.605, 0.646]
#     target_coat_weight_gsm           nunique = 2  [122.48, 182.73]

