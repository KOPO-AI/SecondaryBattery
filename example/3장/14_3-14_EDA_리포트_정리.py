# -*- coding: utf-8 -*-
"""3장 예제 — 3.14 EDA 리포트 정리 — 품질 영향 인자 후보의 도출

교재 출처 : manuscript/31_ch3_eda_fe.md
실행 방법 : example 폴더에서  python "3장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 11_3-11_상관분석.py, 12_3-12_Feature_Engineering.py, 13_3-13_차원_축소_개요.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# ====================================================================
# [선행 절 준비 코드 — 단독 실행용] 아래는 3.7절에서 만든 객체를 이 파일만으로 재현한 것이다.
#   교재 본문에서는 앞 절에서 이미 만들어져 있으므로 다시 실행할 필요가 없다.
# ====================================================================
import pandas as pd
import numpy as np

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
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [종합 코드 (배포 스크립트 실행·해설)]
# --------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""3장 후반 종합: EDA · 상관분석 · Feature Engineering · PCA"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 1. 로드 (3.8) — 정제는 3.7.1의 단일 규약 clean_data()를 호출 ───
# clean_data() 정의는 3.7.1절과 완전히 같으므로 지면에서는 생략했다.
# 배포 스크립트에는 이 함수 정의가 파일 맨 위에 그대로 들어 있다.
df = clean_data('data/battery_process_data.csv')     # 결과: 1,000행 × 10열, 결측 0
num_cols = df.select_dtypes('number').columns.drop('불량_여부')   # 수치 8개

# ── 2. 단변량 (3.9) ─────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, c in zip(axes.ravel(), num_cols):
    sns.histplot(df[c], kde=True, ax=ax); ax.set_title(c)
plt.tight_layout(); plt.show()
print(f"불량률: {df['불량_여부'].mean():.2%}")            # 1.90%

# ── 3. 이변량·다변량 (3.10) ─────────────────────────────────
sns.scatterplot(data=df, x='프레스_Gap', y='최종_용량_mAh', hue='불량_여부')
plt.show()
sns.boxplot(data=df, x='불량_여부', y='믹싱_RPM'); plt.show()

# ── 4. 상관분석 (3.11) ──────────────────────────────────────
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.tight_layout(); plt.show()
print(corr.loc['프레스_Gap', '최종_용량_mAh'].round(4))    # -0.8704

# ── 5. Feature Engineering (3.12) ───────────────────────────
df['압력_Gap_비율'] = df['프레스_압력'] / df['프레스_Gap']
df['Gap_구간'] = pd.cut(df['프레스_Gap'], bins=[0, 90, 100, 999],
                       labels=['얇음', '정상', '두꺼움'])
df = pd.concat([df, pd.get_dummies(df['Gap_구간'], prefix='Gap', dtype=int)], axis=1)
df['RPM_이탈'] = ((df['믹싱_RPM'] < 1600) | (df['믹싱_RPM'] > 1900)).astype(int)
print(df.groupby('Gap_구간', observed=True)['불량_여부'].mean())  # 두꺼움 0.0888
print(df.groupby('RPM_이탈')['불량_여부'].mean())                 # 이탈 0.0536

# ── 6. PCA (3.13) ───────────────────────────────────────────
X = StandardScaler().fit_transform(df[num_cols])
pca = PCA(); Z = pca.fit_transform(X)
print(np.round(pca.explained_variance_ratio_, 3))
# [0.332 0.137 0.127 0.124 0.121 0.118 0.03  0.011]
plt.scatter(Z[:, 0], Z[:, 1], c=df['불량_여부'], cmap='coolwarm', s=12)
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.show()

