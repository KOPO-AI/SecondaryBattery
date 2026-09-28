# -*- coding: utf-8 -*-
"""6장 예제 — 6.8 불량 발생 패턴 분석 `필수`

교재 출처 : manuscript/61_ch6_최적화.md
실행 방법 : example 폴더에서  python "6장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 04_6-4_One.py, 05_6-5_검출_결과의_검증.py, 06_6_예제.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv("data/battery_process_data.csv")
df["건조로_1구간_온도"] = df["건조로_1구간_온도"].replace(9999, np.nan)
df = df.fillna(df.median(numeric_only=True))   # 3.7절 clean_data() 규약

# --------------------------------------------------------------------
# [그룹 통계 비교]
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# [박스플롯으로 확인하기]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   믹싱_RPM         t=+0.46   p=6.53e-01
#   믹싱_온도          t=-1.10   p=2.86e-01
#   코팅_토출압력        t=+2.01   p=5.89e-02
#   건조로_1구간_온도     t=-0.06   p=9.49e-01
#   프레스_압력         t=-1.88   p=7.56e-02
#   프레스_Gap        t=-16.68  p=2.44e-13

# --------------------------------------------------------------------
# [데이터가 숨겨둔 패턴을 '발견'하다]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   불량 Lot RPM 정렬:
#   [1458, 1545, 1624, 1670, 1671, 1681, 1687, 1715, 1741, 1742,
#    1751, 1760, 1771, 1777, 1800, 1839, 1854, 1876, 2004]
#   정상 Lot RPM 범위: 1513 ~ 1983
#   RPM |z|>2 인 불량 Lot: 3 / 19
#   Gap  z>2 인 불량 Lot: 12 / 19
#   둘 중 하나라도 해당: 13 / 19

