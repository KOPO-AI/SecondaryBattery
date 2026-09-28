# -*- coding: utf-8 -*-
"""3장 예제 — 3.10 이변량·다변량 시각화 — 변수 사이의 관계를 그리다

교재 출처 : manuscript/31_ch3_eda_fe.md
실행 방법 : example 폴더에서  python "3장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 07_3-7_전처리_파이프라인_정리.py, 08_3-8_EDA의_목적과_절차.py, 09_3-9_단변량_시각화.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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
# [산점도 — 프레스_Gap과 최종 용량]
# --------------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='프레스_Gap', y='최종_용량_mAh',
                hue='불량_여부', palette={0: 'steelblue', 1: 'crimson'})
plt.title('프레스_Gap vs 최종 용량')
plt.show()

# --------------------------------------------------------------------
# [그룹별 박스플롯 — 연속형 변수 × 범주형 변수]
# --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(data=df, x='불량_여부', y='믹싱_RPM', ax=axes[0])
axes[0].set_title('불량 여부별 믹싱_RPM')
sns.boxplot(data=df, x='불량_여부', y='프레스_Gap', ax=axes[1])
axes[1].set_title('불량 여부별 프레스_Gap')
plt.tight_layout()
plt.show()

print(df.groupby('불량_여부')['믹싱_RPM'].describe().round(1))
print(df.groupby('불량_여부')['프레스_Gap'].describe().round(1))

# --------------------------------------------------------------------
# [pairplot — 다변량 관계의 파노라마]
# --------------------------------------------------------------------
cols = ['프레스_Gap', '최종_용량_mAh', '전극_면저항_mOhmcm2', '믹싱_RPM', '불량_여부']
sns.pairplot(df[cols], hue='불량_여부', corner=True)
plt.show()

# --------------------------------------------------------------------
# [상관 히트맵 — 관계 강도의 지도]
# --------------------------------------------------------------------
corr = df[num_cols].corr()          # 피어슨 상관행렬

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1)
plt.title('공정·품질 변수 피어슨 상관 히트맵')
plt.tight_layout()
plt.show()

