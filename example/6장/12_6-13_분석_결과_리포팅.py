# -*- coding: utf-8 -*-
"""6장 예제 — 6.13 분석 결과 리포팅 `필수`

교재 출처 : manuscript/61_ch6_최적화.md
실행 방법 : example 폴더에서  python "6장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 09_6-10_SPC_관리도.py, 10_6-11_공정능력지수.py, 11_6-12_데이터_기반_공정_최적화_전략.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 6장 후반부 도입 셀·6.8절~6.12절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
# (6장 후반부 도입 셀) 한글 축 라벨·제목이 깨지지 않도록 폰트를 먼저 지정한다(3.9.1절)
plt.rcParams["font.family"] = "Malgun Gothic"   # macOS는 "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# (6장 후반부 도입 셀, 6.1.4절과 동일) 표준 정제 규약 clean_data() + 센서 오류(9999) Lot 5건 제외 → df 995 Lot
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


raw = pd.read_csv("data/battery_process_data.csv", encoding="utf-8-sig")
err_lots = raw.loc[raw["건조로_1구간_온도"] >= 9999, "Lot_ID"].tolist()

df = clean_data("data/battery_process_data.csv")          # 표준 정제 규약

# 6장 공통의 추가 조치: 온도가 대체값인 Lot을 제외(6.1.4절과 동일)
df = df[~df["Lot_ID"].isin(err_lots)].reset_index(drop=True)

# (6.8.2절) 공정 변수 6개
FEATS = ["믹싱_RPM", "믹싱_온도", "코팅_토출압력",
         "건조로_1구간_온도", "프레스_압력", "프레스_Gap"]

# (6.8.2절) 정상·불량 그룹
normal = df[df["불량_여부"] == 0]
defect = df[df["불량_여부"] == 1]

# (6.9.1절) 정상 그룹 기준 변수별 z-score 행렬
mu = normal[FEATS].mean()
sigma = normal[FEATS].std()
Z = (df[FEATS] - mu) / sigma

# (6.9.2절) 불량 Lot의 z-score와 심층 분석 대상(L25949)
z_defect = Z.loc[defect.index]
worst_idx = z_defect.abs().max(axis=1).idxmax()

# (6.9.3절) SHAP 교차 검증에서 만든 입력 X·정답 y
X, y = df[FEATS], df["불량_여부"]

# (6.10.3절) 건조로 온도 X-bar/R 관리도와 관리 이탈 부분군
def xbar_r_chart(series, n=5, A2=0.577, D3=0.0, D4=2.114):
    x = series.values
    k = len(x) // n
    sub = x[:k * n].reshape(k, n)
    xbar = sub.mean(axis=1)
    R = sub.max(axis=1) - sub.min(axis=1)
    xbb, rbar = xbar.mean(), R.mean()
    limits = {"CL": xbb, "UCL": xbb + A2 * rbar, "LCL": xbb - A2 * rbar,
              "CL_R": rbar, "UCL_R": D4 * rbar, "LCL_R": D3 * rbar}
    return xbar, R, limits

xbar, R, lim = xbar_r_chart(df["건조로_1구간_온도"])

ooc_x = np.where((xbar > lim["UCL"]) | (xbar < lim["LCL"]))[0]

# (6.11.2절) 규격 한계와 Cp/Cpk 함수
specs = {  # (LSL, USL)
    "믹싱_RPM":        (1650, 1950),
    "믹싱_온도":        (22, 28),
    "코팅_토출압력":     (105, 135),
    "건조로_1구간_온도":  (100, 130),
    "프레스_압력":       (35, 55),
    "프레스_Gap":       (85, 105),
}

def capability(s, lsl, usl):
    mu, sd = s.mean(), s.std()
    cp = (usl - lsl) / (6 * sd)
    cpk = min(usl - mu, mu - lsl) / (3 * sd)
    return mu, sd, cp, cpk

# (6.12.2절) 용량 예측 회귀 모델과 믹싱 RPM × 건조 온도 격자의 예측 용량
#   — 대시보드 ⑤ 패널에는 예측_용량만 쓰이므로 불량 확률 분류기(clf_full)는 생략한다
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(n_estimators=300, random_state=42)
reg.fit(X, df["최종_용량_mAh"])

rpm_grid = np.linspace(df["믹싱_RPM"].quantile(0.02), df["믹싱_RPM"].quantile(0.98), 30)
tmp_grid = np.linspace(df["건조로_1구간_온도"].quantile(0.02),
                       df["건조로_1구간_온도"].quantile(0.98), 30)
base = df[FEATS].median()

rows = [{**base.to_dict(), "믹싱_RPM": r, "건조로_1구간_온도": t}
        for r in rpm_grid for t in tmp_grid]
grid = pd.DataFrame(rows)[FEATS]
grid["예측_용량"] = reg.predict(grid)
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [1페이지 종합 대시보드 실습]
# --------------------------------------------------------------------
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# (1) 불량 패턴: 프레스_Gap 박스플롯
ax1 = fig.add_subplot(gs[0, 0])
ax1.boxplot([normal["프레스_Gap"], defect["프레스_Gap"]],
            tick_labels=["정상", "불량"])       # labels= 가 아니라 tick_labels=
ax1.set_title("① 불량 패턴: 프레스_Gap (+2.19σ)")

# (2) 원인 추적: L25949 z-score
ax2 = fig.add_subplot(gs[0, 1])
zl = z_defect.loc[worst_idx].sort_values()
ax2.barh(zl.index, zl.values, color=["tab:red" if abs(v) > 2 else "grey" for v in zl])
ax2.axvline(0, color="k", lw=0.8); ax2.set_title("② L25949 원인 분해 (z-score)")

# (3) SPC: 건조로 온도 X-bar 관리도
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(xbar, lw=0.8)
for yv, ls in [(lim["CL"], "-"), (lim["UCL"], "--"), (lim["LCL"], "--")]:
    ax3.axhline(yv, color="r", ls=ls, lw=0.8)
ax3.plot(ooc_x, xbar[ooc_x], "ro")               # 이 데이터에서는 이탈점 0건
ax3.axvspan(176, 184, color="orange", alpha=0.2)  # 런 규칙 4 위반 구간
ax3.set_title("③ 건조로 온도: 3σ 이탈 0건, 런 규칙 1건")

# (4) 공정능력: Cp vs Cpk
ax4 = fig.add_subplot(gs[1, 0])
caps = {c: capability(df[c], *specs[c])[2:] for c in FEATS}
idx_ = np.arange(len(caps))
ax4.bar(idx_ - 0.2, [v[0] for v in caps.values()], 0.4, label="Cp")
ax4.bar(idx_ + 0.2, [v[1] for v in caps.values()], 0.4, label="Cpk")
ax4.axhline(1.33, color="r", ls="--", lw=0.8)
ax4.set_xticks(idx_); ax4.set_xticklabels(caps.keys(), rotation=30, ha="right")
ax4.legend(); ax4.set_title("④ 공정능력 (기준선 1.33)")

# (5) 최적 조건 히트맵
ax5 = fig.add_subplot(gs[1, 1])
H = grid["예측_용량"].values.reshape(30, 30)
im = ax5.imshow(H.T, origin="lower", aspect="auto",
                extent=[rpm_grid[0], rpm_grid[-1], tmp_grid[0], tmp_grid[-1]])
ax5.plot(1786.2, 109.0, "w*", ms=15)
fig.colorbar(im, ax=ax5, shrink=0.8)
ax5.set_title("⑤ 예측 용량 지형과 최적점(★, 검증 전 가설)")

# (6) 핵심 메시지 텍스트 패널 — 패널 제목을 따로 달지 않고 본문만 크게 쓴다
ax6 = fig.add_subplot(gs[1, 2]); ax6.axis("off")
ax6.text(0.02, 0.98,
         "• 불량 19건 중 13건: Gap↑ 또는 RPM 극단\n\n"
         "• 믹싱 RPM 목표 1800 / 경보 1700~1880\n\n"
         "• 프레스 압력 45 ton 재센터링 (Cpk 0.41→0.84)\n\n"
         "• 최적점 RPM 1786 / 건조 109.0℃ — 이득 +3.1 mAh는\n"
         "  모델 RMSE(약 15.5 mAh) 이내, 검증 전 가설\n\n"
         "• 파일럿 30 Lot 검증 후 SOP 개정",
         va="top", fontsize=15, linespacing=1.3)
ax6.set_title("⑥ 핵심 요약")

fig.suptitle("6월 공정 품질 분석 종합 대시보드", fontsize=16)
plt.savefig("dashboard_ch6.png", dpi=150, bbox_inches="tight")

