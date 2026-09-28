# -*- coding: utf-8 -*-
"""6장 예제 — 6.3 Isolation Forest `필수`

교재 출처 : manuscript/60_ch6_이상탐지.md
실행 방법 : example 폴더에서  python "6장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 01_6-1_이상탐지_문제의_정의.py, 02_6-2_통계_기반_기법.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd
import numpy as np

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 6.1절·6.2절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
# (6.1.4절) 표준 정제 규약 clean_data() + 센서 오류(9999) Lot 5건 제외 → df 995 Lot
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

# 이 장만의 추가 조치: 온도가 대체값인 Lot을 이상탐지 입력에서 제외
df = df[~df["Lot_ID"].isin(err_lots)].reset_index(drop=True)

# (6.2.3절) 이상탐지 입력 변수 8개
feats = ["믹싱_RPM", "믹싱_온도", "코팅_토출압력", "건조로_1구간_온도",
         "프레스_압력", "프레스_Gap",              # 공정 변수 6개
         "최종_용량_mAh", "전극_면저항_mOhmcm2"]    # 품질 변수 2개

# (6.2.4절) 긴 한 줄 출력을 접어 주는 표준 모듈
import textwrap   # 긴 한 줄 출력을 지정한 폭에서 접어 주는 파이썬 표준 모듈
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [실습: 정상 데이터만으로 학습 → 전체 적용]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   검출 Lot 수: 29
#   점수 범위: -0.115 ~ 0.175
#   ['L25141', 'L25166', 'L25183', 'L25246', 'L25253', 'L25279',
#    'L25293', 'L25318', 'L25326', 'L25375', 'L25411', 'L25429',
#    'L25476', 'L25546', 'L25620', 'L25627', 'L25711', 'L25723',
#    'L25732', 'L25759', 'L25774', 'L25785', 'L25827', 'L25847',
#    'L25892', 'L25922', 'L25943', 'L25949', 'L25950']

