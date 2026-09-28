# -*- coding: utf-8 -*-
"""6장 예제 — 6.5 검출 결과의 검증 `필수`

교재 출처 : manuscript/60_ch6_이상탐지.md
실행 방법 : example 폴더에서  python "6장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 02_6-2_통계_기반_기법.py, 03_6-3_Isolation_Forest.py, 04_6-4_One.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd
import numpy as np

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 6.1절~6.4절에서 만든 객체를 이 파일만으로 재현한 것이다.
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

# (6.3.4절) 정상 참조로 스케일러·Isolation Forest 학습 → 전체 적용
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

detected_if = df.loc[pred == -1, "Lot_ID"].tolist()

# (6.4.2절) One-Class SVM·LOF 검출 → 세 기법의 집합과 교집합
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
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [이상 판정과 실제 불량의 대조]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#                    기법  검출  적중(불량)  오탐  미탐    정밀도    재현율
#   0  Isolation Forest  29       9  20  10  0.310  0.474
#   1     One-Class SVM  49       9  40  10  0.184  0.474
#   2               LOF  20       3  17  16  0.150  0.158
#   3           3기법 교집합   6       3   3  16  0.500  0.158
#   IF가 적중한 불량 Lot: ['L25183', 'L25253', 'L25279', 'L25293', 'L25326', 'L25774', 'L25827', 'L25922', 'L25949']

