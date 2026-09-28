# -*- coding: utf-8 -*-
"""3장 예제 — 3.5 스케일링 — 변수의 단위를 지우는 작업

교재 출처 : manuscript/30_ch3_전처리.md
실행 방법 : example 폴더에서  python "3장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 02_3-2_결측치_처리.py, 03_3-3_이상치_처리.py, 04_3-4_중복·형변환·단위_정리.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd
import numpy as np
df = pd.read_csv("data/battery_process_data.csv")
df["건조로_1구간_온도"] = df["건조로_1구간_온도"].replace(9999, np.nan)
df = df.fillna(df.median(numeric_only=True))   # 3.7절 clean_data() 규약

# --------------------------------------------------------------------
# [sklearn 실습 — 그리고 데이터 누수 경고]
# --------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

X = df[['믹싱_RPM', '믹싱_온도', '코팅_토출압력', '건조로_1구간_온도',
        '프레스_압력', '프레스_Gap']]
y = df['불량_여부']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

ss = StandardScaler()
X_train_s = ss.fit_transform(X_train)   # train에 fit + transform
X_test_s = ss.transform(X_test)         # test에는 transform만!

print("표준화 후 train 평균:", np.round(X_train_s.mean(axis=0), 4))
print("표준화 후 train 표준편차:", np.round(X_train_s.std(axis=0), 4))
print("표준화 후 test 평균:", np.round(X_test_s.mean(axis=0), 4))

# 교재 실행 결과 ------------------------------------------------------
#   표준화 후 train 평균: [-0. -0.  0.  0.  0.  0.]
#   표준화 후 train 표준편차: [1. 1. 1. 1. 1. 1.]
#   표준화 후 test 평균: [-0.0371 -0.0536  0.0708 -0.0604 -0.1032 -0.0273]

# --------------------------------------------------------------------
# [sklearn 실습 — 그리고 데이터 누수 경고]
# --------------------------------------------------------------------
mm = MinMaxScaler()
X_train_m = mm.fit_transform(X_train)
X_test_m = mm.transform(X_test)
print("MinMax 후 train 최소/최대:", X_train_m.min().round(3), X_train_m.max().round(3))
print("MinMax 후 test 최소/최대:", X_test_m.min().round(3), X_test_m.max().round(3))

# 교재 실행 결과 ------------------------------------------------------
#   MinMax 후 train 최소/최대: 0.0 1.0
#   MinMax 후 test 최소/최대: -0.034 1.0

# --------------------------------------------------------------------
# [sklearn 실습 — 그리고 데이터 누수 경고]
# --------------------------------------------------------------------
comp = pd.DataFrame({'원본': X_train.iloc[0],
                     '표준화': X_train_s[0],
                     'MinMax': X_train_m[0]}).round(3)
print(comp)

# 교재 실행 결과 ------------------------------------------------------
#                   원본    표준화  MinMax
#   믹싱_RPM      1784.0  0.455   0.597
#   믹싱_온도         26.6  1.138   0.689
#   코팅_토출압력      116.2 -0.495   0.403
#   건조로_1구간_온도   111.1  0.273   0.517
#   프레스_압력        40.8  0.203   0.533
#   프레스_Gap       83.7 -1.888   0.221

