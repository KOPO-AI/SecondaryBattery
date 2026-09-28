# -*- coding: utf-8 -*-
"""3장 예제 — 3.15 연습문제

교재 출처 : manuscript/31_ch3_eda_fe.md
실행 방법 : example 폴더에서  python "3장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 12_3-12_Feature_Engineering.py, 13_3-13_차원_축소_개요.py, 14_3-14_EDA_리포트_정리.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 3.7~3.13절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def clean_data(path):
    """배터리 공정 데이터 정제 파이프라인"""
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

df = clean_data('data/battery_process_data.csv')

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트
plt.rcParams['axes.unicode_minus'] = False      # 음수 부호 깨짐 방지
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [해답]
# --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df['프레스_압력'], kde=True, ax=axes[0])
sns.boxplot(y=df['프레스_압력'], ax=axes[1])
plt.tight_layout(); plt.show()

p = df['프레스_압력']
q1, q3 = p.quantile(0.25), p.quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out = df[(p < lower) | (p > upper)]
print(f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, 하한={lower:.2f}, 상한={upper:.2f}")
print("이상치 후보:", len(out), "건 / 그중 불량:", int(out['불량_여부'].sum()))
print("검출값:", sorted(out['프레스_압력'].unique().tolist()))

# 교재 실행 결과 ------------------------------------------------------
#   Q1=37.27, Q3=42.32, IQR=5.05, 하한=29.70, 상한=49.90
#   이상치 후보: 16 건 / 그중 불량: 0
#   검출값: [28.0, 28.8, 29.0, 29.2, 29.4, 29.6, 49.9, 50.1, 50.9, 51.4, 52.0]

# --------------------------------------------------------------------
# [해답]
# --------------------------------------------------------------------
df['Gap_구간4'] = pd.cut(df['프레스_Gap'], bins=[0, 85, 95, 105, 999],
                        labels=['매우얇음', '얇음', '정상', '두꺼움'])
print(df.groupby('Gap_구간4', observed=True)['불량_여부']
        .agg(건수='size', 불량='sum', 불량률='mean').round(4))

# 교재 실행 결과 ------------------------------------------------------
#             건수  불량     불량률
#   Gap_구간4
#   매우얇음      47   0  0.0000
#   얇음       428   0  0.0000
#   정상       474   3  0.0063
#   두꺼움       51  16  0.3137

# --------------------------------------------------------------------
# [해답]
# --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(data=df, x='불량_여부', y='전극_면저항_mOhmcm2', ax=axes[0])
sns.boxplot(data=df, x='불량_여부', y='최종_용량_mAh', ax=axes[1])
plt.tight_layout(); plt.show()

print(df.groupby('불량_여부')['전극_면저항_mOhmcm2'].mean().round(1))
print(df.groupby('불량_여부')['최종_용량_mAh'].mean().round(1))

# 교재 실행 결과 ------------------------------------------------------
#   불량_여부
#   0    1510.9
#   1    1674.3
#   Name: 전극_면저항_mOhmcm2, dtype: float64
#   불량_여부
#   0    3600.5
#   1    3521.2
#   Name: 최종_용량_mAh, dtype: float64

# --------------------------------------------------------------------
# [해답]
# --------------------------------------------------------------------
proc = ['믹싱_RPM', '믹싱_온도', '코팅_토출압력',
        '건조로_1구간_온도', '프레스_압력', '프레스_Gap']
Xp = StandardScaler().fit_transform(df[proc])
pca6 = PCA().fit(Xp)
print(np.round(pca6.explained_variance_ratio_, 3))
print(np.round(np.cumsum(pca6.explained_variance_ratio_), 3))

# 교재 실행 결과 ------------------------------------------------------
#   [0.181 0.179 0.168 0.163 0.157 0.15 ]
#   [0.181 0.361 0.529 0.692 0.85  1.   ]

