# -*- coding: utf-8 -*-
"""6장 예제 — 6.16 실전 이상탐지: 배터리 RUL 사이클 데이터 (실습 55분) `선택 학습(자율 복습)`

교재 출처 : manuscript/61_ch6_최적화.md
실행 방법 : example 폴더에서  python "6장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 10_6-11_공정능력지수.py, 11_6-12_데이터_기반_공정_최적화_전략.py, 12_6-13_분석_결과_리포팅.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --------------------------------------------------------------------
# [1단계 — `describe()`가 던지는 첫 경고 (5분)]
# --------------------------------------------------------------------
import pandas as pd
import numpy as np

rul = pd.read_csv("data/rul/battery_rul_cells.csv")
print("행·열:", rul.shape)
print("셀 개수:", rul["cell_id"].nunique())

time_cols = ["Discharge Time (s)", "Decrement 3.6-3.4V (s)",
             "Time at 4.15V (s)", "Charging time (s)"]
print(rul[time_cols].describe().T[["min", "50%", "max"]].round(1))

# 교재 실행 결과 ------------------------------------------------------
#   행·열: (15064, 18)
#   셀 개수: 14
#                                min     50%       max
#   Discharge Time (s)           8.7  1557.2  958320.4
#   Decrement 3.6-3.4V (s) -397645.9   439.2  406703.8
#   Time at 4.15V (s)         -113.6  2930.2  245101.1
#   Charging time (s)            6.0  8320.4  880728.1

# --------------------------------------------------------------------
# [2단계 — 도메인 규칙을 코드로 쓴다 (10분)]
# --------------------------------------------------------------------
r1 = (rul["Decrement 3.6-3.4V (s)"] < 0) | (rul["Time at 4.15V (s)"] < 0)
r2 = rul["Time at 4.15V (s)"] > rul["Charging time (s)"]
r3 = rul["Decrement 3.6-3.4V (s)"].abs() > rul["Discharge Time (s)"]
r4 = rul["Time constant current (s)"] > rul["Charging time (s)"]

rul["도메인_위반"] = r1 | r2 | r3 | r4

for name, rule in [("① 시간이 음수", r1), ("② 4.15V 도달 > 충전 전체", r2),
                   ("③ |감소 구간| > 방전 전체", r3), ("④ CC 구간 > 충전 전체", r4)]:
    print(f"{name}: {rule.sum():>3}건")
print(f"합집합(물리적 불가): {rul['도메인_위반'].sum()}건 "
      f"({rul['도메인_위반'].mean()*100:.2f}%)")

# 교재 실행 결과 ------------------------------------------------------
#   ① 시간이 음수:  33건
#   ② 4.15V 도달 > 충전 전체:  28건
#   ③ |감소 구간| > 방전 전체:  34건
#   ④ CC 구간 > 충전 전체:   0건
#   합집합(물리적 불가): 68건 (0.45%)

# --------------------------------------------------------------------
# [3단계 — 통계적 이상탐지를 같은 조건으로 돌린다 (10분)]
# --------------------------------------------------------------------
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

FEATS = ["Discharge Time (s)", "Decrement 3.6-3.4V (s)", "Max. Voltage Dischar. (V)",
         "Min. Voltage Charg. (V)", "Time at 4.15V (s)",
         "Time constant current (s)", "Charging time (s)"]

X = StandardScaler().fit_transform(rul[FEATS])
cont = rul["도메인_위반"].mean()          # 0.0045 — 도메인 위반과 같은 비율로 맞춘다

iso = IsolationForest(n_estimators=300, contamination=cont, random_state=42).fit(X)
rul["IF_이상"] = iso.predict(X) == -1
rul["LOF_이상"] = LocalOutlierFactor(n_neighbors=20,
                                    contamination=cont).fit_predict(X) == -1

print(f"contamination = {cont:.4f}")
print("IsolationForest 검출:", rul["IF_이상"].sum(), "건")
print("LOF 검출          :", rul["LOF_이상"].sum(), "건")
print("도메인 규칙 위반  :", rul["도메인_위반"].sum(), "건")

# 교재 실행 결과 ------------------------------------------------------
#   contamination = 0.0045
#   IsolationForest 검출: 64 건
#   LOF 검출          : 68 건
#   도메인 규칙 위반  : 68 건

# --------------------------------------------------------------------
# [두 집합은 얼마나 겹치는가 — 그리고 왜 어긋나는가 (10분)]
# --------------------------------------------------------------------
dom, ifd, lof = rul["도메인_위반"], rul["IF_이상"], rul["LOF_이상"]
rows = [["IF ∩ 도메인",  (ifd & dom).sum()],
        ["LOF ∩ 도메인", (lof & dom).sum()],
        ["IF ∩ LOF",     (ifd & lof).sum()],
        ["도메인만 (통계 2종이 모두 놓침)", (dom & ~ifd & ~lof).sum()],
        ["IF만 (물리적으로는 합법)",        (ifd & ~dom).sum()]]
print(pd.DataFrame(rows, columns=["집합", "행 수"]).to_string(index=False))
print("\n도메인 위반 68건 중 IF가 잡아낸 비율: "
      f"{(ifd & dom).sum() / dom.sum() * 100:.1f}%")

# 교재 실행 결과 ------------------------------------------------------
#                    집합  행 수
#              IF ∩ 도메인    5
#             LOF ∩ 도메인    6
#              IF ∩ LOF   12
#   도메인만 (통계 2종이 모두 놓침)   61
#       IF만 (물리적으로는 합법)   59
#   도메인 위반 68건 중 IF가 잡아낸 비율: 7.4%

# --------------------------------------------------------------------
# [두 집합은 얼마나 겹치는가 — 그리고 왜 어긋나는가 (10분)]
# --------------------------------------------------------------------
Z = ((rul[FEATS] - rul[FEATS].mean()) / rul[FEATS].std()).abs()
rul["최대_z"] = Z.max(axis=1)

viol = rul[rul["도메인_위반"]]
print("도메인 위반 68행의 최대 |z| :")
print(viol["최대_z"].describe().round(2)[["min", "50%", "max"]])
print("그중 |z| < 3 인 행:", (viol["최대_z"] < 3).sum(), "건")

보기 = ["cell_id", "Cycle_Index", "Discharge Time (s)",
        "Decrement 3.6-3.4V (s)", "최대_z"]
print("\n[통계로는 완전히 정상으로 보이는 위반 3건]")
print(viol.nsmallest(3, "최대_z")[보기].round(2).to_string(index=False))

# 교재 실행 결과 ------------------------------------------------------
#   도메인 위반 68행의 최대 |z| :
#   min     0.32
#   50%     8.55
#   max    26.96
#   Name: 최대_z, dtype: float64
#   그중 |z| < 3 인 행: 5 건
#   [통계로는 완전히 정상으로 보이는 위반 3건]
#    cell_id  Cycle_Index  Discharge Time (s)  Decrement 3.6-3.4V (s)  최대_z
#         14        428.0               324.0                  342.86  0.32
#          1        369.0               168.0                  342.86  0.68
#          4        364.0               336.0                  400.00  0.73

# --------------------------------------------------------------------
# [규칙을 특징으로 바꾸면 얼마나 좋아지는가 (5분)]
# --------------------------------------------------------------------
rul["비_감소_방전"] = rul["Decrement 3.6-3.4V (s)"] / rul["Discharge Time (s)"]
rul["비_415_충전"] = rul["Time at 4.15V (s)"] / rul["Charging time (s)"]

X2 = StandardScaler().fit_transform(rul[FEATS + ["비_감소_방전", "비_415_충전"]])
if2 = IsolationForest(n_estimators=300, contamination=cont,
                      random_state=42).fit(X2).predict(X2) == -1

print("비율 특징 2개 추가 후 IF 적중:", (if2 & dom).sum(), "/ 68건")
print("추가 전                    :", (ifd & dom).sum(), "/ 68건")

# 교재 실행 결과 ------------------------------------------------------
#   비율 특징 2개 추가 후 IF 적중: 14 / 68건
#   추가 전                    : 5 / 68건

# --------------------------------------------------------------------
# ["그럼 더 많이 잡으면 되지 않나" — 검출 비율 민감도 (5분)]
# --------------------------------------------------------------------
비율들 = [0.0045, 0.01, 0.02, 0.05, 0.10]

결과 = []
for c in 비율들:
    판정 = IsolationForest(n_estimators=300, contamination=c,
                           random_state=42).fit(X).predict(X) == -1
    적중 = (판정 & dom).sum()
    결과.append([f"{c*100:g}%", 판정.sum(), 적중,
                 round(적중 / dom.sum() * 100, 1),
                 round(적중 / 판정.sum() * 100, 1)])

print(pd.DataFrame(결과, columns=["검출 비율", "검출 행 수", "위반 적중",
                                  "재현율%", "정밀도%"]).to_string(index=False))

# 교재 실행 결과 ------------------------------------------------------
#   검출 비율  검출 행 수  위반 적중  재현율%  정밀도%
#   0.45%      64      5   7.4   7.8
#      1%     151     21  30.9  13.9
#      2%     302     64  94.1  21.2
#      5%     754     65  95.6   8.6
#     10%    1507     65  95.6   4.3

# --------------------------------------------------------------------
# ["그럼 더 많이 잡으면 되지 않나" — 검출 비율 민감도 (5분)]
# --------------------------------------------------------------------
rul["IF_10퍼센트"] = IsolationForest(n_estimators=300, contamination=0.10,
                                     random_state=42).fit(X).predict(X) == -1
끝내못잡음 = dom & ~rul["IF_10퍼센트"]

점수 = pd.Series(-iso.score_samples(X))          # 값이 클수록 이상
rul["상위_%"] = (점수.rank(ascending=False) / len(rul) * 100).values

print("10%까지 풀어도 못 잡는 위반:", 끝내못잡음.sum(), "건")
보기 = ["cell_id", "Cycle_Index", "Discharge Time (s)",
        "Decrement 3.6-3.4V (s)", "최대_z", "상위_%"]
print(rul.loc[끝내못잡음, 보기].round(2).to_string(index=False))
print("\n위반 68건 중 이상점수 상위 2% 안에 든 행:",
      int((rul.loc[dom, "상위_%"] <= 2).sum()), "건")

# 교재 실행 결과 ------------------------------------------------------
#   10%까지 풀어도 못 잡는 위반: 3 건
#    cell_id  Cycle_Index  Discharge Time (s)  Decrement 3.6-3.4V (s)  최대_z  상위_%
#          1        369.0               168.0                  342.86  0.68 13.47
#          4        364.0               336.0                  400.00  0.73 15.14
#         14        428.0               324.0                  342.86  0.32 16.07
#   위반 68건 중 이상점수 상위 2% 안에 든 행: 64 건

# --------------------------------------------------------------------
# ["그럼 더 많이 잡으면 되지 않나" — 검출 비율 민감도 (5분)]
# --------------------------------------------------------------------
극단 = IsolationForest(n_estimators=300, contamination=0.17,
                       random_state=42).fit(X).predict(X) == -1
print(f"검출 비율 17% → {극단.sum():,}행 검출, 위반 {(극단 & dom).sum()}/68건 적중"
      f" (재현율 {(극단 & dom).sum()/dom.sum()*100:.0f}%,"
      f" 정밀도 {(극단 & dom).sum()/극단.sum()*100:.1f}%)")

# 교재 실행 결과 ------------------------------------------------------
#   검출 비율 17% → 2,561행 검출, 위반 68/68건 적중 (재현율 100%, 정밀도 2.7%)

# --------------------------------------------------------------------
# [셀 단위로 보면 이상의 정의가 달라진다 (10분)]
# --------------------------------------------------------------------
percell = np.zeros(len(rul), dtype=bool)
for cid, g in rul.groupby("cell_id"):
    Xg = StandardScaler().fit_transform(g[FEATS])
    m = IsolationForest(n_estimators=300, contamination=cont, random_state=42).fit(Xg)
    percell[g.index] = m.predict(Xg) == -1
rul["셀별_IF"] = percell

요약 = rul.groupby("cell_id").agg(행수=("도메인_위반", "size"),
                                 도메인=("도메인_위반", "sum"),
                                 전체IF=("IF_이상", "sum"),
                                 셀별IF=("셀별_IF", "sum"))
요약["위반율_%"] = (요약["도메인"] / 요약["행수"] * 100).round(2)
print(요약.to_string())
print("\n전체 학습 IF와 셀별 학습 IF의 교집합:", (rul["IF_이상"] & rul["셀별_IF"]).sum(), "건")
print("셀별 학습에서만 잡힌 행           :", (~rul["IF_이상"] & rul["셀별_IF"]).sum(), "건")
print("셀별 학습 IF ∩ 도메인 위반        :", (rul["셀별_IF"] & dom).sum(), "건",
      f"(전체 학습 IF는 {(ifd & dom).sum()}건)")

# 교재 실행 결과 ------------------------------------------------------
#              행수  도메인  전체IF  셀별IF  위반율_%
#   cell_id
#   1        1076    2     7     5   0.19
#   2        1079    2     4     5   0.19
#   3        1077    3     4     5   0.28
#   4        1081    3     5     5   0.28
#   5        1077    5     4     5   0.46
#   6        1078    3     6     5   0.28
#   7        1081    2     6     5   0.19
#   8        1080    5     3     5   0.46
#   9        1079    3     4     5   0.28
#   10       1079    2     6     5   0.19
#   11       1077    6     2     5   0.56
#   12       1077    7     4     5   0.65
#   13       1072    8     5     5   0.75
#   14       1051   17     4     5   1.62
#   전체 학습 IF와 셀별 학습 IF의 교집합: 56 건
#   셀별 학습에서만 잡힌 행           : 14 건
#   셀별 학습 IF ∩ 도메인 위반        : 10 건 (전체 학습 IF는 5건)

