# -*- coding: utf-8 -*-
"""3장 예제 — 3.13 차원 축소 개요 — PCA로 데이터의 뼈대 찾기

교재 출처 : manuscript/31_ch3_eda_fe.md
실행 방법 : example 폴더에서  python "3장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 10_3-10_이변량·다변량_시각화.py, 11_3-11_상관분석.py, 12_3-12_Feature_Engineering.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 3.7~3.9절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
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

num_cols = df.select_dtypes('number').columns.drop('불량_여부')
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [실습 — 설명분산비와 2차원 시각화]
# --------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = StandardScaler().fit_transform(df[num_cols])   # 8개 수치 변수 표준화
pca = PCA()
Z = pca.fit_transform(X)

print(np.round(pca.explained_variance_ratio_, 3))
print(np.round(np.cumsum(pca.explained_variance_ratio_), 3))

# --------------------------------------------------------------------
# [실습 — 설명분산비와 2차원 시각화]
# --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(range(1, 9), pca.explained_variance_ratio_)
axes[0].plot(range(1, 9), np.cumsum(pca.explained_variance_ratio_), 'ro-')
axes[0].set_xlabel('주성분'); axes[0].set_title('설명분산비(막대)와 누적(선)')
axes[1].scatter(Z[:, 0], Z[:, 1], c=df['불량_여부'], cmap='coolwarm', s=12)
axes[1].set_xlabel('PC1 (33.2%)'); axes[1].set_ylabel('PC2 (13.7%)')
axes[1].set_title('주성분 공간의 로트 분포')
plt.tight_layout()
plt.show()

