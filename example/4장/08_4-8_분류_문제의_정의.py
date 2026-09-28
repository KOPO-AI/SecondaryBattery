# -*- coding: utf-8 -*-
"""4장 예제 — 4.8 분류 문제의 정의 — "얼마나"에서 "무엇인가"로

교재 출처 : manuscript/41_ch4_분류.md
실행 방법 : example 폴더에서  python "4장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 05_4-5_교차검증.py, 06_4-6_하이퍼파라미터_튜닝.py, 07_4-7_종합_회귀_모델_구축_파이프라인.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 4.1절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
# ── 4.1.2절: 정제 규약 clean_data() ──
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
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [실습 데이터 준비 — 4장 전반부와 같은 `clean_data()` 규약]
# --------------------------------------------------------------------
import numpy as np
import pandas as pd

# clean_data()는 4.1.2절(= 3.7.1절)에서 정의한 함수를 그대로 쓴다.
# 새 노트북에서 시작했다면 4.1.2절의 정의 셀을 먼저 실행할 것.
df = clean_data("data/battery_process_data.csv")

print(df.shape)
print(df["불량_여부"].value_counts())

# 교재 실행 결과 ------------------------------------------------------
#   (1000, 10)
#   불량_여부
#   0    981
#   1     19
#   Name: count, dtype: int64

# --------------------------------------------------------------------
# [실습 데이터 준비 — 4장 전반부와 같은 `clean_data()` 규약]
# --------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 4.1.2절에서 정의한 것과 같은 6개 공정 변수 리스트, 같은 이름(FEATS)을 쓴다
FEATS = ["믹싱_RPM", "믹싱_온도", "코팅_토출압력",
         "건조로_1구간_온도", "프레스_압력", "프레스_Gap"]
X = df[FEATS]
y = df["불량_여부"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

print("학습:", y_train.value_counts().to_dict())
print("평가:", y_test.value_counts().to_dict())

# 로지스틱 회귀·SVM용 표준화 (트리 계열은 원본 사용)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 교재 실행 결과 ------------------------------------------------------
#   학습: {0: 736, 1: 14}
#   평가: {0: 245, 1: 5}

