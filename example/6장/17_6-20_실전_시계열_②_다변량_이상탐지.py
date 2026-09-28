# -*- coding: utf-8 -*-
"""6장 예제 — 6.20 실전 시계열 ② 다변량 이상탐지 — 편차 9개를 하나의 점수로 (실습 40분) `필수`

교재 출처 : manuscript/61_ch6_최적화.md
포함 소절 : 특징 만들기 — 원시 PV가 아니라 편차를 / 마할라노비스 거리 — 참조 기간의 평균·공분산으로 채점 / 같은 입력에 Isolation Forest를 쓰면 — 그리고 왜 다른가 / 반례 — 원시 PV를 넣으면 어떻게 되는가 / 정리
실행 방법 : example 폴더에서  python "6장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --------------------------------------------------------------------
# [특징 만들기 — 원시 PV가 아니라 편차를]
# --------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

log = pd.read_csv("data/plc/plc_line_log.csv", parse_dates=["timestamp"], index_col="timestamp")
events = pd.read_csv("data/plc/plc_events.csv")
운전중 = log["EC_C_LINE_SPEED_PV"] > 1
코터SV = [c for c in log.columns if c.startswith("EC_C_") and c.endswith("_SV")]
안정화중 = (log[코터SV].diff().abs().sum(axis=1) > 0).rolling("15min").max().fillna(0) > 0

X = pd.DataFrame({
    "속도_편차":   log["EC_C_LINE_SPEED_PV"] - log["EC_C_LINE_SPEED_SV"],
    "DR1_편차":    log["EC_C_DR1_TEMP_PV"] - log["EC_C_DR1_TEMP_SV"],
    "DR2_편차":    log["EC_C_DR2_TEMP_PV"] - log["EC_C_DR2_TEMP_SV"],
    "DR3_편차":    log["EC_C_DR3_TEMP_PV"] - log["EC_C_DR3_TEMP_SV"],
    "DR4_편차":    log["EC_C_DR4_TEMP_PV"] - log["EC_C_DR4_TEMP_SV"],
    "언와인더_편차": log["EC_C_UNWIN_TEN_PV"] - log["EC_C_UNWIN_TEN_SV"],
    "리와인더_편차": log["EC_C_REWIN_TEN_PV"] - log["EC_C_REWIN_TEN_SV"],
    "상부롤_온도":  log["EC_P_UPPER_ROLL_TEMP_PV"],
    "하부롤_온도":  log["EC_P_LOWER_ROLL_TEMP_PV"],
})[운전중 & ~안정화중]
print(X.shape)
print(X.describe().T[["mean", "std", "min", "max"]].round(2))

# 교재 실행 결과 ------------------------------------------------------
#   (9785, 9)
#             mean   std    min    max
#   속도_편차    -0.01  0.26  -0.90   0.94
#   DR1_편차    0.06  0.67  -2.42   2.72
#   DR2_편차   -0.02  0.68  -2.45   9.01
#   DR3_편차   -0.11  0.82  -5.18   2.26
#   DR4_편차   -0.03  0.66  -2.35   2.34
#   언와인더_편차  -0.15  1.44 -15.66   4.34
#   리와인더_편차  -0.04  1.45  -4.90   5.21
#   상부롤_온도   80.31  1.14  78.66  85.22
#   하부롤_온도   80.01  0.42  78.58  81.46

# --------------------------------------------------------------------
# [마할라노비스 거리 — 참조 기간의 평균·공분산으로 채점]
# --------------------------------------------------------------------
참조 = X.loc[:"2026-09-02 23:59"]                       # 1~2일차 = 정상 참조
mu = 참조.mean().to_numpy()
Sigma_inv = np.linalg.inv(참조.cov().to_numpy())
diff = X.to_numpy() - mu
D2 = pd.Series(np.einsum("ij,jk,ik->i", diff, Sigma_inv, diff), index=X.index)

thr = stats.chi2.ppf(0.999, df=X.shape[1])
초과 = D2 > thr
연속3 = 초과.rolling(3).sum() >= 3
print(f"임계값 D² > {thr:.2f} | 참조 기간 초과: {int(초과.loc[:'2026-09-02 23:59'].sum())}분")
print("전체 초과:", int(초과.sum()), "분 → 연속 3분 규칙 후:", int(연속3.sum()), "분")
print("일자별 검출 분 수:"); print(연속3.groupby(연속3.index.date).sum().to_string())

# 교재 실행 결과 ------------------------------------------------------
#   임계값 D² > 27.88 | 참조 기간 초과: 0분
#   전체 초과: 983 분 → 연속 3분 규칙 후: 940 분
#   일자별 검출 분 수:
#   2026-09-01      0
#   2026-09-02      0
#   2026-09-03     33
#   2026-09-04      0
#   2026-09-05      1
#   2026-09-06    118
#   2026-09-07    788

# --------------------------------------------------------------------
# [마할라노비스 거리 — 참조 기간의 평균·공분산으로 채점]
# --------------------------------------------------------------------
def 사건별_검출(검출, events):
    """6.19.5절과 동일"""
    rows = []
    for _, e in events[events["이상여부"] == "Y"].iterrows():
        구간 = 검출.loc[e["시작"]:e["종료"]]
        첫검출 = 구간[구간].index.min() if 구간.any() else pd.NaT
        지연 = (첫검출 - pd.Timestamp(e["시작"])).total_seconds() / 60 if 구간.any() else np.nan
        rows.append([e["사건"], e["태그"], len(구간), int(구간.sum()), 지연])
    표 = pd.DataFrame(rows, columns=["사건", "태그", "구간(분)", "검출(분)", "첫 검출 지연(분)"])
    사건구간 = pd.Series(False, index=검출.index)
    for _, e in events[events["이상여부"] == "Y"].iterrows():
        사건구간.loc[e["시작"]:e["종료"]] = True
    표.attrs["오탐(분)"] = int((검출 & ~사건구간).sum())
    return 표

표2 = 사건별_검출(연속3, events)
print(표2.to_string(index=False)); print("사건 밖 검출(오탐):", 표2.attrs["오탐(분)"], "분")

Z = (X - 참조.mean()) / 참조.std()                        # 참조 기준 표준화
주범 = Z.abs().idxmax(axis=1)
for 사건 in ["C1", "C4", "C2", "C3", "P1"]:
    e = events.set_index("사건").loc[사건]
    구간 = 연속3.loc[e["시작"]:e["종료"]]
    print(사건, "→ 주범 태그:", 주범[구간[구간].index].value_counts().head(1).to_dict())

# 교재 실행 결과 ------------------------------------------------------
#   사건                      태그  구간(분)  검출(분)  첫 검출 지연(분)
#   C1        EC_C_DR2_TEMP_PV      5      3         2.0
#   C4       EC_C_UNWIN_TEN_PV     31     29         2.0
#   M1           EC_M_MMF_A_PV     91      0         NaN
#   C2        EC_C_DR2_TEMP_PV      3      1         2.0
#   C3        EC_C_DR3_TEMP_PV    361    118       136.0
#   P1 EC_P_UPPER_ROLL_TEMP_PV    840    788        43.0
#   사건 밖 검출(오탐): 1 분
#   C1 → 주범 태그: {'DR2_편차': 3}
#   C4 → 주범 태그: {'언와인더_편차': 29}
#   C2 → 주범 태그: {'DR2_편차': 1}
#   C3 → 주범 태그: {'DR3_편차': 115}
#   P1 → 주범 태그: {'상부롤_온도': 788}

# --------------------------------------------------------------------
# [마할라노비스 거리 — 참조 기간의 평균·공분산으로 채점]
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))
np.sqrt(D2).plot(ax=ax, lw=0.6, color="gray", label="마할라노비스 거리 D")
np.sqrt(D2[연속3]).plot(ax=ax, style="r.", ms=5, label="검출(연속 3분)")
ax.axhline(np.sqrt(thr), color="r", ls=":", label="임계 √χ²(9, 99.9%)")
for _, e in events[events["이상여부"] == "Y"].iterrows():
    ax.axvspan(pd.Timestamp(e["시작"]), pd.Timestamp(e["종료"]), color="orange", alpha=0.25)
    ax.text(pd.Timestamp(e["시작"]), np.sqrt(D2).max() * 0.98, e["사건"], fontsize=9, va="top")
ax.axvline(pd.Timestamp("2026-09-04 12:00"), color="b", ls="--", lw=1, label="레시피 변경(정상)")
ax.set_ylabel("D (참조 기간 기준)"); ax.legend(loc="upper left", ncol=2)
plt.tight_layout(); plt.savefig("fig_6_41_plc_maha.png", dpi=150); plt.close()

# --------------------------------------------------------------------
# [같은 입력에 Isolation Forest를 쓰면 — 그리고 왜 다른가]
# --------------------------------------------------------------------
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(참조)
iso = IsolationForest(n_estimators=200, contamination=0.001, random_state=42).fit(scaler.transform(참조))
판정_if = pd.Series(iso.predict(scaler.transform(X)) == -1, index=X.index)
연속3_if = 판정_if.rolling(3).sum() >= 3
표3 = 사건별_검출(연속3_if, events)
print(표3.to_string(index=False)); print("사건 밖 검출(오탐):", 표3.attrs["오탐(분)"], "분")
점수_if = pd.Series(-iso.score_samples(scaler.transform(X)), index=X.index)
print("C1 구간 IF 점수:", 점수_if.loc["2026-09-03 14:20":"2026-09-03 14:24"].round(3).tolist(),
      "| 임계:", round(-iso.offset_, 3), "| 참조 최대:", round(점수_if.loc[:"2026-09-02 23:59"].max(), 3))

# 교재 실행 결과 ------------------------------------------------------
#   사건                      태그  구간(분)  검출(분)  첫 검출 지연(분)
#   C1        EC_C_DR2_TEMP_PV      5      0         NaN
#   C4       EC_C_UNWIN_TEN_PV     31      0         NaN
#   M1           EC_M_MMF_A_PV     91      0         NaN
#   C2        EC_C_DR2_TEMP_PV      3      0         NaN
#   C3        EC_C_DR3_TEMP_PV    361      6       339.0
#   P1 EC_P_UPPER_ROLL_TEMP_PV    840      9        93.0
#   사건 밖 검출(오탐): 5 분
#   C1 구간 IF 점수: [0.458, 0.464, 0.459, 0.481, 0.474] | 임계: 0.554 | 참조 최대: 0.56

# --------------------------------------------------------------------
# [반례 — 원시 PV를 넣으면 어떻게 되는가]
# --------------------------------------------------------------------
X_raw = log[["EC_C_LINE_SPEED_PV", "EC_C_DR1_TEMP_PV", "EC_C_DR2_TEMP_PV", "EC_C_DR3_TEMP_PV",
             "EC_C_DR4_TEMP_PV", "EC_C_UNWIN_TEN_PV", "EC_C_REWIN_TEN_PV",
             "EC_P_UPPER_ROLL_TEMP_PV", "EC_P_LOWER_ROLL_TEMP_PV"]][운전중 & ~안정화중]
참조r = X_raw.loc[:"2026-09-02 23:59"]
diff_r = X_raw.to_numpy() - 참조r.mean().to_numpy()
D2_raw = pd.Series(np.einsum("ij,jk,ik->i", diff_r, np.linalg.inv(참조r.cov().to_numpy()), diff_r), index=X_raw.index)
print("레시피 변경 전 검출률:", round((D2_raw.loc[:"2026-09-04 11:59"] > thr).mean() * 100, 1), "%")
print("레시피 변경 후 검출률:", round((D2_raw.loc["2026-09-04 12:00":] > thr).mean() * 100, 1), "%")

# 교재 실행 결과 ------------------------------------------------------
#   레시피 변경 전 검출률: 0.8 %
#   레시피 변경 후 검출률: 100.0 %

