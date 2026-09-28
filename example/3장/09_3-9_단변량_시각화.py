# -*- coding: utf-8 -*-
"""3장 예제 — 3.9 단변량 시각화 — 변수 하나의 얼굴을 그리다

교재 출처 : manuscript/31_ch3_eda_fe.md
실행 방법 : example 폴더에서  python "3장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 06_3-6_시계열_데이터_다루기_기초.py, 07_3-7_전처리_파이프라인_정리.py, 08_3-8_EDA의_목적과_절차.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd
import numpy as np
df = pd.read_csv("data/battery_process_data.csv")
df["건조로_1구간_온도"] = df["건조로_1구간_온도"].replace(9999, np.nan)
df = df.fillna(df.median(numeric_only=True))   # 3.7절 clean_data() 규약

# --------------------------------------------------------------------
# [한글 폰트 설정부터]
# --------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트
plt.rcParams['axes.unicode_minus'] = False      # 음수 부호 깨짐 방지

# --------------------------------------------------------------------
# [히스토그램 — 분포의 전체 모양]
# --------------------------------------------------------------------
num_cols = df.select_dtypes('number').columns.drop('불량_여부')

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), num_cols):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# [히스토그램 — 분포의 전체 모양]
# --------------------------------------------------------------------
print(df[num_cols].skew().round(3))

# --------------------------------------------------------------------
# [박스플롯 — 사분위수와 이상치의 요약]
# --------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, col in zip(axes, ['믹싱_RPM', '프레스_Gap', '최종_용량_mAh', '전극_면저항_mOhmcm2']):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# [countplot — 타깃(불량) 분포의 확인]
# --------------------------------------------------------------------
sns.countplot(data=df, x='불량_여부')
plt.title('불량 여부 분포')
plt.show()

print(df['불량_여부'].value_counts())
print(f"불량률: {df['불량_여부'].mean():.2%}")

