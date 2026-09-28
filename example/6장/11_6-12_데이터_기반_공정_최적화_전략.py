# -*- coding: utf-8 -*-
"""6장 예제 — 6.12 데이터 기반 공정 최적화 전략 `필수`

교재 출처 : manuscript/61_ch6_최적화.md
실행 방법 : example 폴더에서  python "6장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 08_6-9_원인_추적_기법.py, 09_6-10_SPC_관리도.py, 10_6-11_공정능력지수.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd
import numpy as np

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 6장 후반부 도입 셀·6.8절·6.9절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
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

# (6.9.3절) SHAP 교차 검증에서 만든 입력 X·정답 y와 분류기 클래스
from sklearn.ensemble import RandomForestClassifier

X, y = df[FEATS], df["불량_여부"]
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [그리드 탐색 실습 — 믹싱 RPM × 건조 온도]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#         믹싱_RPM  건조로_1구간_온도    예측_용량  예측_불량확률
#   552  1786.21      108.99  3601.96      0.0
#   553  1786.21      109.53  3601.96      0.0
#   549  1786.21      107.36  3601.87      0.0

