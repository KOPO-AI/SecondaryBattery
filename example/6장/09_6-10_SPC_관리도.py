# -*- coding: utf-8 -*-
"""6장 예제 — 6.10 SPC 관리도 `필수`

교재 출처 : manuscript/61_ch6_최적화.md
실행 방법 : example 폴더에서  python "6장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 06_6_예제.py, 07_6-8_불량_발생_패턴_분석.py, 08_6-9_원인_추적_기법.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import numpy as np
import pandas as pd
df = pd.read_csv("data/battery_process_data.csv")
df["건조로_1구간_온도"] = df["건조로_1구간_온도"].replace(9999, np.nan)
df = df.fillna(df.median(numeric_only=True))   # 3.7절 clean_data() 규약

# --------------------------------------------------------------------
# [파이썬 구현 — 건조로 온도 관리도]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   {'CL': 109.99, 'UCL': 115.17, 'LCL': 104.81, 'CL_R': 8.98, 'UCL_R': 18.99, 'LCL_R': 0.0}
#   X-bar 관리 이탈 부분군: [] []
#   R 관리 이탈 부분군: [168] [20.3]

# --------------------------------------------------------------------
# [Western Electric 런 규칙 — 한계 안쪽의 이상 신호]
# --------------------------------------------------------------------
def rule_run8(xbar, cl):
    side = np.sign(xbar - cl)
    hits, run = [], 1
    for i in range(1, len(side)):
        run = run + 1 if side[i] == side[i - 1] else 1
        if run == 8:
            hits.append(i)
    return hits

print("연속 8점 편측 위반(종료 부분군):", rule_run8(xbar, lim["CL"]))

# 교재 실행 결과 ------------------------------------------------------
#   연속 8점 편측 위반(종료 부분군): [183]

