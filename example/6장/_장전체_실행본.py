# -*- coding: utf-8 -*-
"""6장 — 장 전체 예제 실행본

교재 6장의 파이썬 코드를 등장 순서대로 이어 붙였다.
절별 파일은 앞 절에서 만든 변수를 이어 쓰는 경우가 있어 단독 실행이 안 될 수 있는데,
이 파일은 처음부터 끝까지 순서대로 실행되므로 그대로 돌려 볼 수 있다.

실행 : example 폴더에서  python "6장/_장전체_실행본.py"
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""
import matplotlib
matplotlib.use("Agg")   # 창을 띄우지 않고 그림을 파일로만 처리


# ====================================================================
# 이상탐지 문제의 정의 `필수` / 실습 데이터 준비 — 표준 정제 규약 + 이 장만의 추가 조치
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


raw = pd.read_csv("data/battery_process_data.csv", encoding="utf-8-sig")
err_lots = raw.loc[raw["건조로_1구간_온도"] >= 9999, "Lot_ID"].tolist()
print("센서 오류(9999) Lot:", err_lots)

df = clean_data("data/battery_process_data.csv")          # 표준 정제 규약
print("clean_data() 직후:", df.shape)

# 이 장만의 추가 조치: 온도가 대체값인 Lot을 이상탐지 입력에서 제외
df = df[~df["Lot_ID"].isin(err_lots)].reset_index(drop=True)
print("센서 오류 Lot 제외 후:", df.shape, "/ 불량 Lot 수:", df["불량_여부"].sum())


# ====================================================================
# 통계 기반 기법 `필수` / z-score: 가장 단순한 출발점
# ====================================================================
from scipy import stats

z = np.abs(stats.zscore(df["최종_용량_mAh"]))
out_z = df.loc[z > 3, "Lot_ID"].tolist()
print(f"용량 z-score>3 이상 Lot: {len(out_z)}개 → {out_z}")


# ====================================================================
# 통계 기반 기법 `필수` / 수정 z-score: 중앙값과 MAD
# ====================================================================
x = df["최종_용량_mAh"]
med = np.median(x)
mad = np.median(np.abs(x - med))
mz = 0.6745 * (x - med) / mad
out_mz = df.loc[np.abs(mz) > 3.5, "Lot_ID"].tolist()
print(f"수정 z-score>3.5 이상 Lot: {len(out_mz)}개 → {out_mz}")


# ====================================================================
# 통계 기반 기법 `필수` / 변수를 늘리면 생기는 문제: 다중 비교
# ====================================================================
feats = ["믹싱_RPM", "믹싱_온도", "코팅_토출압력", "건조로_1구간_온도",
         "프레스_압력", "프레스_Gap",              # 공정 변수 6개
         "최종_용량_mAh", "전극_면저항_mOhmcm2"]    # 품질 변수 2개

zall = np.abs(stats.zscore(df[feats]))
print("어느 한 변수라도 |z|>3인 Lot:", (zall > 3).any(axis=1).sum(), "개")


# ====================================================================
# 통계 기반 기법 `필수` / 다변량으로: 마할라노비스 거리
# ====================================================================
import textwrap   # 긴 한 줄 출력을 지정한 폭에서 접어 주는 파이썬 표준 모듈

# feats는 6.2.3절에서 정의한 8개 변수 목록을 그대로 사용한다
X = df[feats].values

mu = X.mean(axis=0)
Sigma_inv = np.linalg.inv(np.cov(X.T))
d2 = np.einsum("ij,jk,ik->i", X - mu, Sigma_inv, X - mu)  # D_M^2

thr = stats.chi2.ppf(0.99, df=len(feats))  # χ²(8)의 99% 분위수
out_maha = df.loc[d2 > thr, "Lot_ID"].tolist()
print(f"임계값 D²>{thr:.2f} → 이상 Lot {len(out_maha)}개")
# 그냥 print(out_maha)로도 되지만 한 줄이 너무 길어지므로 62자에서 접어 출력한다.
# subsequent_indent=" "는 두 번째 줄부터 한 칸 들여쓰라는 뜻이다(내용은 그대로).
print(textwrap.fill(str(out_maha), width=62, subsequent_indent=" "))


# ====================================================================
# Isolation Forest `필수` / 실습: 정상 데이터만으로 학습 → 전체 적용
# ====================================================================
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1) 참조(정상) 데이터로 스케일러·모델 학습
normal = df[df["불량_여부"] == 0]          # 정상 참조 976 Lot
scaler = StandardScaler().fit(normal[feats])
X_train = scaler.transform(normal[feats])
X_all   = scaler.transform(df[feats])       # 전체 995 Lot

iforest = IsolationForest(n_estimators=200, contamination=0.02,
                          random_state=42)
iforest.fit(X_train)

# 2) 전체 데이터에 적용
pred  = iforest.predict(X_all)              # -1: 이상, +1: 정상
score = iforest.decision_function(X_all)    # 낮을수록 이상

detected_if = df.loc[pred == -1, "Lot_ID"].tolist()
print(f"검출 Lot 수: {len(detected_if)}")
print(f"점수 범위: {score.min():.3f} ~ {score.max():.3f}")
print(textwrap.fill(str(sorted(detected_if)), width=62, subsequent_indent=" "))


# ====================================================================
# One-Class SVM과 LOF `선택 학습(자율 복습)` / LOF: 이웃과 밀도를 비교한다
# ====================================================================
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# One-Class SVM: 정상 참조로 학습 → 전체 적용
ocsvm = OneClassSVM(kernel="rbf", nu=0.02, gamma="scale").fit(X_train)
detected_oc = df.loc[ocsvm.predict(X_all) == -1, "Lot_ID"].tolist()

# LOF: 국소 밀도 기반 (기본형은 fit_predict로 일괄 판정)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
detected_lof = df.loc[lof.fit_predict(X_all) == -1, "Lot_ID"].tolist()

set_if, set_oc, set_lof = map(set, (detected_if, detected_oc, detected_lof))
inter3 = set_if & set_oc & set_lof
print(f"OCSVM {len(set_oc)} / LOF {len(set_lof)} / 3기법 교집합 {len(inter3)}")
print("교집합 Lot:", sorted(inter3))


# ====================================================================
# 검출 결과의 검증 `필수` / 이상 판정과 실제 불량의 대조
# ====================================================================
bad = set(df.loc[df["불량_여부"] == 1, "Lot_ID"])   # 실제 불량 19 Lot

rows = []
for name, s in [("Isolation Forest", set_if), ("One-Class SVM", set_oc),
                ("LOF", set_lof), ("3기법 교집합", inter3)]:
    hit = len(s & bad)
    rows.append([name, len(s), hit, len(s) - hit, len(bad) - hit,
                 hit / len(s), hit / len(bad)])

print(pd.DataFrame(rows, columns=["기법", "검출", "적중(불량)", "오탐",
                                  "미탐", "정밀도", "재현율"]).round(3))
print("IF가 적중한 불량 Lot:", sorted(set_if & bad))


# ====================================================================
# 
# ====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 축 라벨·제목이 깨지지 않도록 폰트를 먼저 지정한다(3.9.1절)
plt.rcParams["font.family"] = "Malgun Gothic"   # macOS는 "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


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
print("센서 오류(9999) Lot:", err_lots)

df = clean_data("data/battery_process_data.csv")          # 표준 정제 규약
print("clean_data() 직후:", df.shape)

# 6장 공통의 추가 조치: 온도가 대체값인 Lot을 제외(6.1.4절과 동일)
df = df[~df["Lot_ID"].isin(err_lots)].reset_index(drop=True)
print("센서 오류 Lot 제외 후:", df.shape)
print("불량률: {:.2%}".format(df["불량_여부"].mean()))


# ====================================================================
# 불량 발생 패턴 분석 `필수` / 그룹 통계 비교
# ====================================================================
FEATS = ["믹싱_RPM", "믹싱_온도", "코팅_토출압력",
         "건조로_1구간_온도", "프레스_압력", "프레스_Gap"]

# 그룹별 평균·표준편차
group_stats = df.groupby("불량_여부")[FEATS].agg(["mean", "std"]).round(2)
print(group_stats.T)

# 표준화된 평균 차이(정상 그룹 표준편차 기준)
normal = df[df["불량_여부"] == 0]
defect = df[df["불량_여부"] == 1]
for col in FEATS:
    d = (defect[col].mean() - normal[col].mean()) / normal[col].std()
    print(f"{col:14s}  정상 {normal[col].mean():8.2f}  불량 {defect[col].mean():8.2f}  차이 {d:+.2f}σ")


# ====================================================================
# 불량 발생 패턴 분석 `필수` / 박스플롯으로 확인하기
# ====================================================================
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), FEATS):
    data = [normal[col], defect[col]]
    # matplotlib 3.9부터 labels= 는 tick_labels= 로 이름이 바뀌었다
    ax.boxplot(data, tick_labels=["정상\n(n=976)", "불량\n(n=19)"])
    t, p = stats.ttest_ind(normal[col], defect[col], equal_var=False)
    ax.set_title(f"{col}\n(Welch t={t:.2f}, p={p:.2e})")
    print(f"{col:14s} t={t:<+7.2f} p={p:.2e}")   # 제목에 넣은 값을 화면에도 남긴다
plt.tight_layout()
plt.savefig("fig_6_9_boxplot.png", dpi=150)


# ====================================================================
# 불량 발생 패턴 분석 `필수` / 데이터가 숨겨둔 패턴을 '발견'하다
# ====================================================================
import textwrap   # 긴 한 줄 출력을 지정한 폭에서 접어 주는 파이썬 표준 모듈

print("불량 Lot RPM 정렬:")
# 값 19개를 한 줄에 찍으면 화면을 넘어가므로 62자에서 접는다(내용은 그대로).
print(textwrap.fill(str(sorted(defect["믹싱_RPM"])), width=62,
                    subsequent_indent=" "))
print("정상 Lot RPM 범위:", normal["믹싱_RPM"].min(), "~", normal["믹싱_RPM"].max())

# 정상 분포 기준 z-score가 ±2를 벗어나는 불량 Lot 수
z_rpm = (defect["믹싱_RPM"] - normal["믹싱_RPM"].mean()) / normal["믹싱_RPM"].std()
z_gap = (defect["프레스_Gap"] - normal["프레스_Gap"].mean()) / normal["프레스_Gap"].std()
print("RPM |z|>2 인 불량 Lot:", (z_rpm.abs() > 2).sum(), "/ 19")
print("Gap  z>2 인 불량 Lot:", (z_gap > 2).sum(), "/ 19")
print("둘 중 하나라도 해당:", ((z_rpm.abs() > 2) | (z_gap > 2)).sum(), "/ 19")


# ====================================================================
# 원인 추적(Root Cause) 기법 `필수` / 변수별 z-score 기여도
# ====================================================================
mu = normal[FEATS].mean()
sigma = normal[FEATS].std()
Z = (df[FEATS] - mu) / sigma

# 불량 Lot 전체의 변수별 평균 |z| — 집단 수준의 기여도
contrib = Z.loc[defect.index].abs().mean().sort_values(ascending=False)
print(contrib.round(2))


# ====================================================================
# 원인 추적(Root Cause) 기법 `필수` / 이상 Lot 1건 심층 분석 워크스루
# ====================================================================
z_defect = Z.loc[defect.index]
worst_idx = z_defect.abs().max(axis=1).idxmax()
print("심층 분석 대상:", df.loc[worst_idx, "Lot_ID"])
print(df.loc[worst_idx])
print("\n변수별 z-score:")
print(z_defect.loc[worst_idx].round(2).sort_values())


# ====================================================================
# 원인 추적(Root Cause) 기법 `필수` / SHAP 개별 설명으로 교차 검증 (5장 연계)
# ====================================================================
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = df[FEATS], df["불량_여부"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
# roc_auc_score는 파이썬 float를 반환하므로 .round()가 아니라 내장 round()를 쓴다
print("검증 AUC:", round(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]), 3))

explainer = shap.TreeExplainer(clf)
sv = explainer.shap_values(df.loc[[worst_idx], FEATS])
shap_lot = pd.Series(np.array(sv[..., 1]).ravel(), index=FEATS)
print("기저 불량 확률:", explainer.expected_value[1].round(3))
print("예측 불량 확률:", clf.predict_proba(df.loc[[worst_idx], FEATS])[0, 1].round(3))
print(shap_lot.round(3).sort_values(ascending=False))


# ====================================================================
# SPC 관리도 `필수` / 파이썬 구현 — 건조로 온도 관리도
# ====================================================================
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
# round()에 넘파이 실수를 그대로 넣으면 np.float64(109.99)처럼 찍히므로
# float()으로 파이썬 기본 실수로 바꿔 준다.
print({k: round(float(v), 2) for k, v in lim.items()})

ooc_x = np.where((xbar > lim["UCL"]) | (xbar < lim["LCL"]))[0]
ooc_r = np.where(R > lim["UCL_R"])[0]
print("X-bar 관리 이탈 부분군:", ooc_x, xbar[ooc_x].round(2))
print("R 관리 이탈 부분군:", ooc_r, R[ooc_r].round(2))


# ====================================================================
# SPC 관리도 `필수` / Western Electric 런 규칙 — 한계 안쪽의 이상 신호
# ====================================================================
def rule_run8(xbar, cl):
    side = np.sign(xbar - cl)
    hits, run = [], 1
    for i in range(1, len(side)):
        run = run + 1 if side[i] == side[i - 1] else 1
        if run == 8:
            hits.append(i)
    return hits

print("연속 8점 편측 위반(종료 부분군):", rule_run8(xbar, lim["CL"]))


# ====================================================================
# 공정능력지수 `필수` / 실습 데이터의 Cp/Cpk 산출
# ====================================================================
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

print(f"{'변수':14s} {'평균':>8s} {'표준편차':>8s} {'Cp':>6s} {'Cpk':>6s}")
for col, (l, u) in specs.items():
    mu, sd, cp, cpk = capability(df[col], l, u)
    print(f"{col:14s} {mu:8.2f} {sd:8.2f} {cp:6.2f} {cpk:6.2f}")


# ====================================================================
# 데이터 기반 공정 최적화 전략 `필수` / 그리드 탐색 실습 — 믹싱 RPM × 건조 온도
# ====================================================================
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(n_estimators=300, random_state=42)
reg.fit(X, df["최종_용량_mAh"])
clf_full = RandomForestClassifier(n_estimators=300, random_state=42).fit(X, y)

rpm_grid = np.linspace(df["믹싱_RPM"].quantile(0.02), df["믹싱_RPM"].quantile(0.98), 30)
tmp_grid = np.linspace(df["건조로_1구간_온도"].quantile(0.02),
                       df["건조로_1구간_온도"].quantile(0.98), 30)
base = df[FEATS].median()

rows = [{**base.to_dict(), "믹싱_RPM": r, "건조로_1구간_온도": t}
        for r in rpm_grid for t in tmp_grid]
grid = pd.DataFrame(rows)[FEATS]
grid["예측_용량"] = reg.predict(grid)
grid["예측_불량확률"] = clf_full.predict_proba(grid[FEATS])[:, 1]

feasible = grid[grid["예측_불량확률"] < 0.01]   # 불량 확률 1% 미만 제약
best = feasible.sort_values("예측_용량", ascending=False).head(3)
print(best[["믹싱_RPM", "건조로_1구간_온도", "예측_용량", "예측_불량확률"]].round(2))


# ====================================================================
# 분석 결과 리포팅 `필수` / 1페이지 종합 대시보드 실습
# ====================================================================
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


# ====================================================================
# 실전 이상탐지: 배터리 RUL 사이클 데이터 (실습 55분) `선택 학습(자율 복습)` / 1단계 — `describe()`가 던지는 첫 경고 (5분)
# ====================================================================
import pandas as pd
import numpy as np

rul = pd.read_csv("data/rul/battery_rul_cells.csv")
print("행·열:", rul.shape)
print("셀 개수:", rul["cell_id"].nunique())

time_cols = ["Discharge Time (s)", "Decrement 3.6-3.4V (s)",
             "Time at 4.15V (s)", "Charging time (s)"]
print(rul[time_cols].describe().T[["min", "50%", "max"]].round(1))


# ====================================================================
# 실전 이상탐지: 배터리 RUL 사이클 데이터 (실습 55분) `선택 학습(자율 복습)` / 2단계 — 도메인 규칙을 코드로 쓴다 (10분)
# ====================================================================
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


# ====================================================================
# 실전 이상탐지: 배터리 RUL 사이클 데이터 (실습 55분) `선택 학습(자율 복습)` / 3단계 — 통계적 이상탐지를 같은 조건으로 돌린다 (10분)
# ====================================================================
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


# ====================================================================
# 실전 이상탐지: 배터리 RUL 사이클 데이터 (실습 55분) `선택 학습(자율 복습)` / 두 집합은 얼마나 겹치는가 — 그리고 왜 어긋나는가 (10분)
# ====================================================================
dom, ifd, lof = rul["도메인_위반"], rul["IF_이상"], rul["LOF_이상"]
rows = [["IF ∩ 도메인",  (ifd & dom).sum()],
        ["LOF ∩ 도메인", (lof & dom).sum()],
        ["IF ∩ LOF",     (ifd & lof).sum()],
        ["도메인만 (통계 2종이 모두 놓침)", (dom & ~ifd & ~lof).sum()],
        ["IF만 (물리적으로는 합법)",        (ifd & ~dom).sum()]]
print(pd.DataFrame(rows, columns=["집합", "행 수"]).to_string(index=False))
print("\n도메인 위반 68건 중 IF가 잡아낸 비율: "
      f"{(ifd & dom).sum() / dom.sum() * 100:.1f}%")

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


# ====================================================================
# 실전 이상탐지: 배터리 RUL 사이클 데이터 (실습 55분) `선택 학습(자율 복습)` / 규칙을 특징으로 바꾸면 얼마나 좋아지는가 (5분)
# ====================================================================
rul["비_감소_방전"] = rul["Decrement 3.6-3.4V (s)"] / rul["Discharge Time (s)"]
rul["비_415_충전"] = rul["Time at 4.15V (s)"] / rul["Charging time (s)"]

X2 = StandardScaler().fit_transform(rul[FEATS + ["비_감소_방전", "비_415_충전"]])
if2 = IsolationForest(n_estimators=300, contamination=cont,
                      random_state=42).fit(X2).predict(X2) == -1

print("비율 특징 2개 추가 후 IF 적중:", (if2 & dom).sum(), "/ 68건")
print("추가 전                    :", (ifd & dom).sum(), "/ 68건")


# ====================================================================
# 실전 이상탐지: 배터리 RUL 사이클 데이터 (실습 55분) `선택 학습(자율 복습)` / "그럼 더 많이 잡으면 되지 않나" — 검출 비율 민감도 (5분)
# ====================================================================
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

극단 = IsolationForest(n_estimators=300, contamination=0.17,
                       random_state=42).fit(X).predict(X) == -1
print(f"검출 비율 17% → {극단.sum():,}행 검출, 위반 {(극단 & dom).sum()}/68건 적중"
      f" (재현율 {(극단 & dom).sum()/dom.sum()*100:.0f}%,"
      f" 정밀도 {(극단 & dom).sum()/극단.sum()*100:.1f}%)")


# ====================================================================
# 실전 이상탐지: 배터리 RUL 사이클 데이터 (실습 55분) `선택 학습(자율 복습)` / 셀 단위로 보면 이상의 정의가 달라진다 (10분)
# ====================================================================
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


# ====================================================================
# 실전 공정 최적화: WMG 캘린더링 DoE (실습 45분) `선택 학습(자율 복습)` / 데이터 소개 — 관찰 데이터가 아니라 '설계된 실험'
# ====================================================================
wmg = pd.read_csv("data/wmg/wmg_cells_54.csv")
print("행·열:", wmg.shape, "/ 조건 수:", wmg["group"].nunique())
print(pd.crosstab([wmg["coat_weight_level"], wmg["density_level"]],
                  wmg["roll_temp_c"]))


# ====================================================================
# 실전 공정 최적화: WMG 캘린더링 DoE (실습 45분) `선택 학습(자율 복습)` / 조건 평균표 — 18개 점이 말해 주는 것 (5분)
# ====================================================================
Y = "grav_dis_5c_mahg"
표 = wmg.pivot_table(index=["coat_weight_level", "density_level"],
                     columns="roll_temp_c", values=Y, aggfunc="mean").round(1)
print(표)


# ====================================================================
# 실전 공정 최적화: WMG 캘린더링 DoE (실습 45분) `선택 학습(자율 복습)` / 먼저 잡음의 크기를 잰다 — 재현성 하한선 (10분)
# ====================================================================
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


# ====================================================================
# 실전 공정 최적화: WMG 캘린더링 DoE (실습 45분) `선택 학습(자율 복습)` / 인자별 기여율 — 무엇이 지배하는가 (5분)
# ====================================================================
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


# ====================================================================
# 실전 공정 최적화: WMG 캘린더링 DoE (실습 45분) `선택 학습(자율 복습)` / 반응면을 그린다 — 그리고 반드시 검증한다 (10분)
# ====================================================================
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


# ====================================================================
# 실전 공정 최적화: WMG 캘린더링 DoE (실습 45분) `선택 학습(자율 복습)` / 그래서 최적점은 어디인가 — 18개 점으로 말할 수 있는 것 (5분)
# ====================================================================
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


# ====================================================================
# 실전 공정 최적화: WMG 캘린더링 DoE (실습 45분) `선택 학습(자율 복습)` / 최고 조건 하나를 고를 수 있는가 — 조건 평균의 신뢰구간 (5분)
# ====================================================================
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

q = stats.studentized_range.ppf(0.95, 18, 36)   # 조건 18개를 서로 다 비교할 때
HSD = q * np.sqrt(MSE / 3)
print(f"Tukey HSD = {HSD:.1f} mAh/g  (LSD {LSD:.1f}보다 넓다)")
print("HSD 기준 동률 조건:", (조건평균 >= 최고 - HSD).sum(), "개")
print("그중 코팅중량 L:", sum(i[0] == "L" for i in 조건평균[조건평균 >= 최고 - HSD].index), "개")


# ====================================================================
# 실전 공정 최적화: WMG 캘린더링 DoE (실습 45분) `선택 학습(자율 복습)` / 이 데이터의 함정 — 실전 데이터를 쓸 때 반드시 확인할 것 (5분)
# ====================================================================
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

cond = pd.read_csv("data/wmg/wmg_conditions_18.csv")

print("[교락 확인] 캘린더링 날짜 × 롤 온도 교차표")
print(pd.crosstab(cond["calendering_date"], cond["roll_temp_c"]))

print("\n[공선성 확인] 18조건에 걸친 고유값 개수")
for col in ["precal_grad_carbon", "precal_tensile_strength_kpa",
            "precal_moran_carbon", "target_coat_weight_gsm"]:
    # .tolist()를 붙이지 않으면 넘파이 실수라서 np.float64(0.00654)처럼 찍힌다
    값들 = sorted(cond[col].unique().tolist())
    print(f"  {col:32s} nunique = {cond[col].nunique()}  {값들}")


# ====================================================================
# 종합: 두 데이터, 두 결론 (강의 10분) `필수` / 하나의 자로 재기 — 효과 크기 ÷ 잡음
# ====================================================================
# (효과 크기, 잡음, 단위) — 앞 절들에서 직접 계산해 얻은 값만 옮겨 적는다
항목 = [
    ["가상 CSV · 그리드 탐색 최적점",  3.060, 15.500, "mAh (6.12.2절 / 5장 RMSE)"],
    ["WMG DoE · 롤 온도",             7.336,  7.192, "mAh/g (6.17.3절)"],
    ["WMG DoE · 목표 밀도",          24.809,  7.192, "mAh/g (6.17.3절)"],
    ["WMG DoE · 코팅 중량",          72.949,  7.192, "mAh/g (6.17.3절)"],
]
표 = pd.DataFrame(항목, columns=["결론", "효과 크기", "잡음", "단위·출처"])
표["효과÷잡음"] = (표["효과 크기"] / 표["잡음"]).round(2)
표["말할 수 있는가"] = ["아니오 (오차 이내)" if r < 1 else
                        ("경계선" if r < 1.5 else "예") for r in 표["효과÷잡음"]]
print(표.to_string(index=False))

# ====================================================================
# 실전 시계열 ① PLC 태그 로그 읽기와 기초 이상탐지 (실습 40분) `필수` / 데이터 소개 — 태그 사전과 로그
# ====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

tags = pd.read_csv("data/plc/plc_tag_list.csv")
print(tags.shape)
print(tags[["태그명", "설비", "구분", "설명", "단위"]].to_string(index=False))

log = pd.read_csv("data/plc/plc_line_log.csv", parse_dates=["timestamp"], index_col="timestamp")
print(log.shape)
print(log.index.min(), "~", log.index.max(), "| 간격:", log.index[1] - log.index[0])
print(log[["EC_C_LINE_SPEED_PV", "EC_C_DR2_TEMP_PV", "EC_C_UNWIN_TEN_PV", "EC_N_금형횟수_현재값"]].describe().round(2))


# ====================================================================
# 실전 시계열 ① PLC 태그 로그 읽기와 기초 이상탐지 (실습 40분) `필수` / 운전·정지 구분과 전체 조망
# ====================================================================
운전중 = log["EC_C_LINE_SPEED_PV"] > 1          # 라인 속도가 0 근처면 정지
print("정지 분 수:", (~운전중).sum(), "/", len(log),
      "→ 운전 중 비율", round(운전중.mean() * 100, 1), "%")

fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
log["EC_C_LINE_SPEED_PV"].plot(ax=axes[0], lw=0.8)
axes[0].set_ylabel("라인 속도\n(m/min)")
log["EC_C_DR2_TEMP_PV"].plot(ax=axes[1], lw=0.8, label="PV(현재값)")
log["EC_C_DR2_TEMP_SV"].plot(ax=axes[1], lw=1.2, ls="--", color="k", label="SV(설정값)")
axes[1].set_ylabel("건조로 2구간\n온도(℃)"); axes[1].legend(loc="upper left")
log["EC_C_UNWIN_TEN_PV"].plot(ax=axes[2], lw=0.8)
axes[2].set_ylabel("언와인더\n장력(N)")
log["EC_N_금형횟수_현재값"].plot(ax=axes[3], lw=1.0)
axes[3].axhline(600000, color="r", ls=":", label="경보 설정 600,000")
axes[3].set_ylabel("금형 타발\n누적 횟수"); axes[3].legend(loc="upper left")
plt.tight_layout(); plt.savefig("fig_6_39_plc_overview.png", dpi=150); plt.close()


# ====================================================================
# 실전 시계열 ① PLC 태그 로그 읽기와 기초 이상탐지 (실습 40분) `필수` / 규칙 ① PV−SV 편차 — 가장 단순하고 가장 현장적인 기준
# ====================================================================
편차 = log["EC_C_DR2_TEMP_PV"] - log["EC_C_DR2_TEMP_SV"]
규칙1 = (편차.abs() > 3.0) & 운전중
print("|PV-SV| > 3℃ 인 분 수:", 규칙1.sum())
print(편차[규칙1].round(2).to_string())

설정변경 = log["EC_C_DR2_TEMP_SV"].diff().abs() > 0
안정화중 = 설정변경.rolling("15min").max().fillna(0) > 0      # 변경 후 15분간 True
규칙1 = (편차.abs() > 3.0) & 운전중 & ~안정화중
print("안정화 15분 제외 후:", 규칙1.sum(), "분")


# ====================================================================
# 실전 시계열 ① PLC 태그 로그 읽기와 기초 이상탐지 (실습 40분) `필수` / 규칙 ② 이동 z-score와 연속 N회 규칙 — 설정값이 없는 태그를 위해
# ====================================================================
def rolling_z(s, window="60min", min_periods=30):
    """직전 window 구간(현재값 제외)의 평균·표준편차로 표준화한 z-score"""
    m = s.rolling(window, min_periods=min_periods).mean().shift(1)
    sd = s.rolling(window, min_periods=min_periods).std().shift(1)
    return (s - m) / sd

z2 = rolling_z(log.loc[운전중, "EC_C_DR2_TEMP_PV"])
초과 = z2.abs() > 3
연속3 = 초과.rolling(3).sum() >= 3                    # 3분 연속 초과일 때만
print("2구간 온도 |z| > 3:", 초과.sum(), "분 → 연속 3분 규칙 후:", 연속3.sum(), "분")
print(z2[연속3].round(1).to_string())
z속도 = rolling_z(log.loc[운전중, "EC_C_LINE_SPEED_PV"])
print("라인 속도 12:00 전후 z:", z속도.loc["2026-09-04 11:59":"2026-09-04 12:02"].round(1).tolist())

z_상부롤 = rolling_z(log.loc[운전중, "EC_P_UPPER_ROLL_TEMP_PV"])
연속3_롤 = (z_상부롤.abs() > 3).rolling(3).sum() >= 3
print("상부 롤 온도 연속 3분 검출:", 연속3_롤.sum(), "분 —", 연속3_롤[연속3_롤].index.strftime("%m-%d %H:%M").tolist())


# ====================================================================
# 실전 시계열 ① PLC 태그 로그 읽기와 기초 이상탐지 (실습 40분) `필수` / 채점 — 정답지와 대조
# ====================================================================
events = pd.read_csv("data/plc/plc_events.csv")
print(events[["사건", "설비", "태그", "시작", "종료", "유형", "이상여부"]].to_string(index=False))

def 사건별_검출(검출, events):
    """검출(True/False 시계열)을 사건 구간과 대조 — 사건별 검출 분 수와 첫 검출까지 지연(분)"""
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

표1 = 사건별_검출(규칙1, events)
print(표1.to_string(index=False)); print("사건 밖 검출(오탐):", 표1.attrs["오탐(분)"], "분")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
편차[운전중].plot(ax=axes[0], lw=0.7, color="gray")
편차[규칙1].plot(ax=axes[0], style="r.", ms=6, label="|PV-SV|>3℃ 검출(안정화 제외)")
axes[0].axhline(3, color="r", ls=":"); axes[0].axhline(-3, color="r", ls=":")
axes[0].set_ylabel("건조로 2구간\nPV-SV (℃)"); axes[0].legend(loc="upper left")
z2.plot(ax=axes[1], lw=0.7, color="gray", label="이동 z-score")
z2[연속3].plot(ax=axes[1], style="r.", ms=6, label="|z|>3 연속 3분")
axes[1].axhline(3, color="r", ls=":"); axes[1].axhline(-3, color="r", ls=":")
axes[1].set_ylabel("건조로 2구간\n이동 z-score"); axes[1].legend(loc="upper left")
plt.tight_layout(); plt.savefig("fig_6_40_plc_rules.png", dpi=150); plt.close()


# ====================================================================
# 실전 시계열 ② 다변량 이상탐지 — 편차 9개를 하나의 점수로 (실습 40분) `필수` / 특징 만들기 — 원시 PV가 아니라 편차를
# ====================================================================
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


# ====================================================================
# 실전 시계열 ② 다변량 이상탐지 — 편차 9개를 하나의 점수로 (실습 40분) `필수` / 마할라노비스 거리 — 참조 기간의 평균·공분산으로 채점
# ====================================================================
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


# ====================================================================
# 실전 시계열 ② 다변량 이상탐지 — 편차 9개를 하나의 점수로 (실습 40분) `필수` / 같은 입력에 Isolation Forest를 쓰면 — 그리고 왜 다른가
# ====================================================================
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


# ====================================================================
# 실전 시계열 ② 다변량 이상탐지 — 편차 9개를 하나의 점수로 (실습 40분) `필수` / 반례 — 원시 PV를 넣으면 어떻게 되는가
# ====================================================================
X_raw = log[["EC_C_LINE_SPEED_PV", "EC_C_DR1_TEMP_PV", "EC_C_DR2_TEMP_PV", "EC_C_DR3_TEMP_PV",
             "EC_C_DR4_TEMP_PV", "EC_C_UNWIN_TEN_PV", "EC_C_REWIN_TEN_PV",
             "EC_P_UPPER_ROLL_TEMP_PV", "EC_P_LOWER_ROLL_TEMP_PV"]][운전중 & ~안정화중]
참조r = X_raw.loc[:"2026-09-02 23:59"]
diff_r = X_raw.to_numpy() - 참조r.mean().to_numpy()
D2_raw = pd.Series(np.einsum("ij,jk,ik->i", diff_r, np.linalg.inv(참조r.cov().to_numpy()), diff_r), index=X_raw.index)
print("레시피 변경 전 검출률:", round((D2_raw.loc[:"2026-09-04 11:59"] > thr).mean() * 100, 1), "%")
print("레시피 변경 후 검출률:", round((D2_raw.loc["2026-09-04 12:00":] > thr).mean() * 100, 1), "%")


# ====================================================================
# 실전 시계열 ③ 시계열 예측 — 1분 뒤 온도와 금형 교체 시점 (실습 45분) `필수` / 지연 특징 만들기와 시계열 교차검증
# ====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

log = pd.read_csv("data/plc/plc_line_log.csv", parse_dates=["timestamp"], index_col="timestamp")
events = pd.read_csv("data/plc/plc_events.csv")

y = log["EC_C_DR2_TEMP_PV"] - log["EC_C_DR2_TEMP_SV"]        # 예측 대상도 편차(PV-SV)다
feat = pd.DataFrame({f"lag{k}": y.shift(k) for k in (1, 2, 3, 5, 10)})
feat["이동평균10"] = y.shift(1).rolling(10).mean()
data = feat.join(y.rename("y")).dropna()
print(data.shape)
print(data.head(3).round(2).to_string())

학습 = data.loc[:"2026-09-04 23:59"]                    # 1~4일차로 학습·검증, 5~7일차는 봉인
Xtr, ytr = 학습.drop(columns="y"), 학습["y"]
tscv = TimeSeriesSplit(n_splits=5)
결과 = {"naive(lag1)": [], "선형 회귀": [], "랜덤 포레스트": []}
for tr, va in tscv.split(Xtr):
    결과["naive(lag1)"].append(mean_absolute_error(ytr.iloc[va], Xtr["lag1"].iloc[va]))
    lr = LinearRegression().fit(Xtr.iloc[tr], ytr.iloc[tr])
    결과["선형 회귀"].append(mean_absolute_error(ytr.iloc[va], lr.predict(Xtr.iloc[va])))
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(Xtr.iloc[tr], ytr.iloc[tr])
    결과["랜덤 포레스트"].append(mean_absolute_error(ytr.iloc[va], rf.predict(Xtr.iloc[va])))
print(pd.DataFrame(결과).round(3).assign(fold=range(1, 6)).set_index("fold").T.to_string())
print("평균 MAE(℃):", {k: round(float(np.mean(v)), 3) for k, v in 결과.items()})


# ====================================================================
# 실전 시계열 ③ 시계열 예측 — 1분 뒤 온도와 금형 교체 시점 (실습 45분) `필수` / 예측 잔차로 이상 잡기
# ====================================================================
lr = LinearRegression().fit(Xtr, ytr)
잔차_학습 = ytr - lr.predict(Xtr)
σ = 잔차_학습.std()
print("학습 기간 잔차 표준편차:", round(σ, 3), "℃ → 임계 ±", round(3 * σ, 2), "℃")

시험 = data.loc["2026-09-05":]
예측 = pd.Series(lr.predict(시험.drop(columns="y")), index=시험.index)
잔차 = 시험["y"] - 예측
검출 = 잔차.abs() > 3 * σ
print("5~7일차 예측 MAE:", round(mean_absolute_error(시험["y"], 예측), 3), "℃")
print("잔차 이상 검출:", int(검출.sum()), "분")
print(잔차[검출].round(2).to_string())

구간 = slice("2026-09-05 02:40", "2026-09-05 03:40")
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
시험.loc[구간, "y"].plot(ax=axes[0], lw=1.2, label="실측 편차")
예측.loc[구간].plot(ax=axes[0], lw=1.2, ls="--", label="1분 뒤 예측(선형 회귀)")
axes[0].set_ylabel("건조로 2구간\nPV-SV (℃)"); axes[0].legend(loc="upper left")
잔차.loc[구간].plot(ax=axes[1], lw=1.0, color="gray")
잔차.loc[구간][검출.loc[구간]].plot(ax=axes[1], style="r.", ms=8, label="|잔차| > 3σ")
axes[1].axhline(3 * σ, color="r", ls=":"); axes[1].axhline(-3 * σ, color="r", ls=":")
axes[1].set_ylabel("잔차 = 실측 - 예측(℃)"); axes[1].legend(loc="upper left")
plt.tight_layout(); plt.savefig("fig_6_42_plc_forecast_resid.png", dpi=150); plt.close()


# ====================================================================
# 실전 시계열 ③ 시계열 예측 — 1분 뒤 온도와 금형 교체 시점 (실습 45분) `필수` / 예측 잔차가 못 잡는 것 — 완만한 드리프트
# ====================================================================
y3 = log["EC_C_DR3_TEMP_PV"] - log["EC_C_DR3_TEMP_SV"]
f3 = pd.DataFrame({f"lag{k}": y3.shift(k) for k in (1, 2, 3, 5, 10)})
f3["이동평균10"] = y3.shift(1).rolling(10).mean()
d3 = f3.join(y3.rename("y")).dropna()
tr3, te3 = d3.loc[:"2026-09-04 23:59"], d3.loc["2026-09-05":]
lr3 = LinearRegression().fit(tr3.drop(columns="y"), tr3["y"])
σ3 = (tr3["y"] - lr3.predict(tr3.drop(columns="y"))).std()
잔차3 = te3["y"] - lr3.predict(te3.drop(columns="y"))
C3구간 = slice("2026-09-06 09:00", "2026-09-06 15:00")
print("C3 구간 잔차 절댓값 최대:", round(잔차3.loc[C3구간].abs().max(), 2), "℃ | 임계 3σ =", round(3 * σ3, 2),
      "℃ → 검출", int((잔차3.loc[C3구간].abs() > 3 * σ3).sum()), "분 /", len(잔차3.loc[C3구간]))

편차3 = log["EC_C_DR3_TEMP_PV"] - log["EC_C_DR3_TEMP_SV"]
편차3_60 = 편차3.rolling("60min").mean()                     # 편차의 60분 이동평균 — 기준은 SV(고정)
기준sd = 편차3_60.loc[:"2026-09-02 23:59"].std()
검출3 = 편차3_60.abs() > 3 * 기준sd
print("편차 60분 이동평균 |·| > 3σ(참조):", int(검출3.loc[C3구간].sum()), "분 검출, 첫 검출",
      검출3.loc[C3구간][검출3.loc[C3구간]].index.min())


# ====================================================================
# 실전 시계열 ③ 시계열 예측 — 1분 뒤 온도와 금형 교체 시점 (실습 45분) `필수` / 금형 교체 시점 예측 — 추세 외삽
# ====================================================================
횟수 = log["EC_N_금형횟수_현재값"]
경보 = int(log["EC_N_금형횟수_경보_설정"].iloc[-1])
교체시각 = 횟수.index[횟수.diff() < 0]                     # 횟수가 줄어든 순간 = 교체
print("금형 교체 시각:", list(교체시각.strftime("%m-%d %H:%M")))
최근 = 횟수.loc[교체시각[-1]:]                             # 마지막 교체 이후만 사용
print("마지막 교체 이후 경과:", round((최근.index[-1] - 최근.index[0]).total_seconds() / 3600, 1),
      "시간 | 현재 횟수:", int(최근.iloc[-1]))

경과분 = ((최근.index - 최근.index[0]).total_seconds() / 60).to_numpy().reshape(-1, 1)
lr = LinearRegression().fit(경과분, 최근.to_numpy())
기울기 = lr.coef_[0]                                       # 회/분 (정지 시간이 섞인 평균 속도)
남은분 = (경보 - 최근.iloc[-1]) / 기울기
도달예상 = 최근.index[-1] + pd.Timedelta(minutes=남은분)
print("평균 속도:", round(기울기, 1), "회/분 →", round(기울기 * 1440), "회/일")
print("경보 도달 예상:", 도달예상.strftime("%Y-%m-%d %H:%M"), "(", round(남은분 / 60, 1), "시간 뒤 )")

fig, ax = plt.subplots(figsize=(12, 4))
횟수.plot(ax=ax, lw=1, x_compat=True, label="금형 타발 누적 횟수")   # x_compat: 날짜 축을 matplotlib 방식으로
미래 = pd.date_range(최근.index[0], 도달예상, freq="60min")
미래분 = ((미래 - 최근.index[0]).total_seconds() / 60).to_numpy().reshape(-1, 1)
ax.plot(미래, lr.predict(미래분), "r--", label="추세 외삽")
ax.axhline(경보, color="k", ls=":", label="경보 설정 600,000")
ax.axvline(도달예상, color="r", ls=":")
ax.text(도달예상, 경보 * 0.45, "도달 예상\n" + 도달예상.strftime("%m-%d %H:%M"), color="r", ha="right")
ax.set_ylabel("횟수"); ax.legend(loc="upper left")
plt.savefig("fig_6_43_plc_die_forecast.png", dpi=150, bbox_inches="tight"); plt.close()

# ====================================================================
# 실전 시계열 ④ 딥러닝 맛보기 — LSTM 오토인코더로 같은 로그를 감시하다 (실습 45분) `선택 학습(자율 복습)` / 창(window) 만들기 — 한 행이 아니라 30분을 한 장으로
# ====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
})[운전중 & ~안정화중]                                   # 6.20절과 동일한 입력
참조 = X.loc[:"2026-09-02 23:59"]
Z = ((X - 참조.mean()) / 참조.std()).astype("float32")   # 참조 기준 표준화

W = 30                                                   # 창 길이(분)
def 창_자르기(Z, W):
    """연속한 W행씩 잘라 (창 수, W, 특징 수) 배열로 — 각 창은 마지막 분의 시각으로 대표한다"""
    arr = Z.to_numpy()
    창 = np.stack([arr[i - W:i] for i in range(W, len(arr) + 1)])
    return 창, Z.index[W - 1:]

창_전체, 시각 = 창_자르기(Z, W)
참조끝 = int((시각 <= pd.Timestamp("2026-09-02 23:59")).sum())
창_참조 = 창_전체[:참조끝]
print("전체 창:", 창_전체.shape, "| 참조 창:", 창_참조.shape)


# ====================================================================
# 실전 시계열 ④ 딥러닝 맛보기 — LSTM 오토인코더로 같은 로그를 감시하다 (실습 45분) `선택 학습(자율 복습)` / 모델 — 누르고(encoder) 펼친다(decoder)
# ====================================================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"                # TensorFlow의 정보성 메시지를 줄인다(경고는 무해하다)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
keras.utils.set_random_seed(42)                          # 재현을 위한 시드(완전히 같지는 않을 수 있다)

n_feat = 창_참조.shape[2]
model = keras.Sequential([
    layers.Input(shape=(W, n_feat)),
    layers.LSTM(32),                                     # encoder: 30분 × 9특징 → 벡터 32개
    layers.RepeatVector(W),                              # 그 벡터를 30번 복사해 시간 축을 되살린다
    layers.LSTM(32, return_sequences=True),              # decoder
    layers.TimeDistributed(layers.Dense(n_feat)),        # 매 분마다 9특징을 복원
])
model.compile(optimizer="adam", loss="mse")
print("학습 파라미터 수:", model.count_params())

hist = model.fit(창_참조, 창_참조, epochs=15, batch_size=64, validation_split=0.1, shuffle=True, verbose=0)
print("학습 손실(MSE) 처음→끝:", round(hist.history["loss"][0], 4), "→", round(hist.history["loss"][-1], 4),
      "| 검증 손실 끝:", round(hist.history["val_loss"][-1], 4))


# ====================================================================
# 실전 시계열 ④ 딥러닝 맛보기 — LSTM 오토인코더로 같은 로그를 감시하다 (실습 45분) `선택 학습(자율 복습)` / 채점 — 복원 오차와 정답지 대조
# ====================================================================
복원 = model.predict(창_전체, verbose=0)
오차 = pd.Series(((창_전체 - 복원) ** 2).mean(axis=(1, 2)), index=시각)      # 창 하나의 평균 제곱 오차
임계 = np.percentile(오차.iloc[:참조끝], 99.9)                                # 참조 기간 오차의 99.9 백분위
초과 = 오차 > 임계
연속3 = 초과.rolling(3).sum() >= 3
print(f"임계(참조 99.9%): {임계:.3f} | 참조 기간 초과: {int(초과.iloc[:참조끝].sum())}분 | 전체 초과 {int(초과.sum())}분 → 연속 3분 {int(연속3.sum())}분")

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

표4 = 사건별_검출(연속3, events)
print(표4.to_string(index=False)); print("사건 밖 검출(오탐):", 표4.attrs["오탐(분)"], "분")

특징오차 = pd.DataFrame(((창_전체 - 복원) ** 2)[:, -1, :], index=시각, columns=X.columns)   # 창의 마지막 분, 특징별 오차
주범 = 특징오차.idxmax(axis=1)
for 사건 in ["C1", "C4", "C2", "C3", "P1"]:
    e = events.set_index("사건").loc[사건]
    구간 = 연속3.loc[e["시작"]:e["종료"]]
    print(사건, "→ 주범 태그:", 주범[구간[구간].index].value_counts().head(1).to_dict())

fig, ax = plt.subplots(figsize=(12, 4))
오차.plot(ax=ax, lw=0.6, color="gray", label="복원 오차(30분 창 MSE)")
오차[연속3].plot(ax=ax, style="r.", ms=5, label="검출(연속 3분)")
ax.axhline(임계, color="r", ls=":", label="임계(참조 99.9%)")
for _, e in events[events["이상여부"] == "Y"].iterrows():
    ax.axvspan(pd.Timestamp(e["시작"]), pd.Timestamp(e["종료"]), color="orange", alpha=0.25)
    ax.text(pd.Timestamp(e["시작"]), 오차.max() * 0.98, e["사건"], fontsize=9, va="top")
ax.set_yscale("log"); ax.set_ylabel("복원 오차 (log)"); ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig("fig_6_44_plc_lstm_ae.png", dpi=150); plt.close()

