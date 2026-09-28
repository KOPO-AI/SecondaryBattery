# -*- coding: utf-8 -*-
"""3장 — 장 전체 예제 실행본

교재 3장의 파이썬 코드를 등장 순서대로 이어 붙였다.
절별 파일은 앞 절에서 만든 변수를 이어 쓰는 경우가 있어 단독 실행이 안 될 수 있는데,
이 파일은 처음부터 끝까지 순서대로 실행되므로 그대로 돌려 볼 수 있다.

실행 : example 폴더에서  python "3장/_장전체_실행본.py"
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""
import matplotlib
matplotlib.use("Agg")   # 창을 띄우지 않고 그림을 파일로만 처리


# ====================================================================
# 왜 전처리가 절반인가 / 이상치 다섯 개가 통계 전체를 무너뜨린다 — 실계산 시연
# ====================================================================
import pandas as pd
import numpy as np

df = pd.read_csv('data/battery_process_data.csv')
print(df.shape)

temp = df['건조로_1구간_온도']
print("평균(원본, 9999 포함):", round(temp.mean(), 2))
print("중앙값(원본):", round(temp.median(), 2))

normal = temp[temp < 1000]          # 9999를 제외한 정상값
print("9999 제외 평균:", round(normal.mean(), 2))
print("9999 제외 중앙값:", round(normal.median(), 2))
print("9999 개수:", (temp == 9999).sum())

print("원본 상관계수(건조온도 vs 용량):",
      round(df['건조로_1구간_온도'].corr(df['최종_용량_mAh']), 4))

mask = df['건조로_1구간_온도'] < 1000
print("9999 제외 상관계수:",
      round(df.loc[mask, '건조로_1구간_온도'].corr(df.loc[mask, '최종_용량_mAh']), 4))


# ====================================================================
# 결측치 처리 / 결측치의 탐지
# ====================================================================
print(df.isna().sum())

print(df[df['믹싱_온도'].isna()][['Lot_ID', '믹싱_RPM', '믹싱_온도', '불량_여부']])


# ====================================================================
# 결측치 처리 / 실습 — 믹싱_온도 결측 5건 처리
# ====================================================================
mean_v = df['믹싱_온도'].mean()
med_v = df['믹싱_온도'].median()
print("평균:", round(mean_v, 3), " 중앙값:", round(med_v, 3))

print("불량_여부 그룹별 믹싱_온도 평균:")
print(df.groupby('불량_여부')['믹싱_온도'].mean().round(3))

df['믹싱_온도'] = df['믹싱_온도'].fillna(med_v)
print("결측치 개수(처리 후):", df['믹싱_온도'].isna().sum())


# ====================================================================
# 이상치 처리 / 에러 코드형 이상치의 식별 — describe가 알려주는 신호
# ====================================================================
temp = df['건조로_1구간_온도']
print(temp.describe().round(2))

print(temp.sort_values(ascending=False).head(5))

df['건조로_1구간_온도'] = df['건조로_1구간_온도'].replace(9999, np.nan)
print("NaN 개수:", df['건조로_1구간_온도'].isna().sum())

med_d = df['건조로_1구간_온도'].median()
df['건조로_1구간_온도'] = df['건조로_1구간_온도'].fillna(med_d)
print("중앙값으로 대체 완료, 대체값:", round(med_d, 2))
print(df['건조로_1구간_온도'].describe().round(2))


# ====================================================================
# 이상치 처리 / IQR 규칙 — 박스플롯의 수학
# ====================================================================
q1 = df['프레스_압력'].quantile(0.25)
q3 = df['프레스_압력'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print(f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
print(f"하한={lower:.2f}, 상한={upper:.2f}")

out = df[(df['프레스_압력'] < lower) | (df['프레스_압력'] > upper)]
print("IQR 기준 이상치 개수:", len(out))
print(out[['Lot_ID', '프레스_압력', '불량_여부']].head(10))


# ====================================================================
# 이상치 처리 / z-score — 표준편차 몇 배나 벗어났는가
# ====================================================================
z = (df['프레스_압력'] - df['프레스_압력'].mean()) / df['프레스_압력'].std()
print("|Z|>3 개수:", (z.abs() > 3).sum())
print("Z 최대/최소:", round(z.max(), 2), round(z.min(), 2))


# ====================================================================
# 이상치 처리 / 도메인 지식 기반 물리 범위 검사 — 온도는 200℃일 수 없다
# ====================================================================
rules = {'믹싱_온도': (10, 50), '건조로_1구간_온도': (80, 160), '프레스_압력': (20, 80)}
for col, (lo, hi) in rules.items():
    n = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"{col}: 물리 범위 ({lo}~{hi}) 위반 {n}건")


# ====================================================================
# 중복·형변환·단위 정리 / 중복 데이터 — duplicated
# ====================================================================
print("완전 중복 행:", df.duplicated().sum())
print("Lot_ID 중복:", df.duplicated(subset='Lot_ID').sum())


# ====================================================================
# 중복·형변환·단위 정리 / 형변환 — astype과 숫자 열이 문자가 되는 사고
# ====================================================================
print(df.dtypes)
df['불량_여부'] = df['불량_여부'].astype('int8')
print("변환 후 불량_여부:", df['불량_여부'].dtype)


# ====================================================================
# 스케일링 — 변수의 단위를 지우는 작업 / sklearn 실습 — 그리고 데이터 누수 경고
# ====================================================================
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

mm = MinMaxScaler()
X_train_m = mm.fit_transform(X_train)
X_test_m = mm.transform(X_test)
print("MinMax 후 train 최소/최대:", X_train_m.min().round(3), X_train_m.max().round(3))
print("MinMax 후 test 최소/최대:", X_test_m.min().round(3), X_test_m.max().round(3))

comp = pd.DataFrame({'원본': X_train.iloc[0],
                     '표준화': X_train_s[0],
                     'MinMax': X_train_m[0]}).round(3)
print(comp)


# ====================================================================
# 시계열 데이터 다루기 기초 / 합성 센서 로그 생성과 datetime 파싱
# ====================================================================
rng = np.random.default_rng(42)
idx = pd.date_range('2026-03-02 08:00', periods=600, freq='10s')
sensor = pd.DataFrame({
    'timestamp': idx,
    '건조로_온도': 120 + 3*np.sin(np.arange(600)/50) + rng.normal(0, 0.8, 600)
})
sensor.loc[300:305, '건조로_온도'] += 15   # 순간 스파이크 주입
print(sensor.head())
print("전체 구간:", sensor['timestamp'].min(), "~", sensor['timestamp'].max())

sensor['timestamp'] = pd.to_datetime(sensor['timestamp'])
s = sensor.set_index('timestamp')      # 시각을 인덱스로


# ====================================================================
# 시계열 데이터 다루기 기초 / 리샘플링 — 시간 해상도 바꾸기
# ====================================================================
res = s.resample('1min').mean()
print(res.head())
print("리샘플링 후 행 수:", len(res))


# ====================================================================
# 시계열 데이터 다루기 기초 / 이동평균 — 잡음 속에서 추세 읽기
# ====================================================================
s['이동평균_1min'] = s['건조로_온도'].rolling('60s').mean()
print(s.iloc[298:308].round(2))


# ====================================================================
# 시계열 데이터 다루기 기초 / 시계열 동기화 개요 — 로트와 센서 로그 잇기
# ====================================================================
lot_log = pd.DataFrame({
    'timestamp': pd.to_datetime(['2026-03-02 08:07:23',
                                 '2026-03-02 08:31:11',
                                 '2026-03-02 08:55:40']),
    'Lot_ID': ['L25101', 'L25102', 'L25103']
})
sync = pd.merge_asof(lot_log, sensor.sort_values('timestamp'),
                     on='timestamp', direction='nearest')
print(sync)


# ====================================================================
# 전처리 파이프라인 정리 / 재사용 가능한 clean_data() 함수
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

raw = pd.read_csv('data/battery_process_data.csv')
clean = clean_data('data/battery_process_data.csv')
print("정제 전:", raw.shape, "/ 정제 후:", clean.shape)


# ====================================================================
# 전처리 파이프라인 정리 / 처리 전후 비교 — describe로 검증하기
# ====================================================================
cols = ['믹싱_온도', '건조로_1구간_온도', '프레스_압력']
행 = ['count', 'mean', 'std', 'min', 'max']

print("[정제 전]")
print(raw[cols].describe().loc[행].round(2))
print()                                    # 빈 줄 하나 — 두 표를 눈으로 분리
print("[정제 후]")
print(clean[cols].describe().loc[행].round(2))


# ====================================================================
# 전처리 파이프라인 정리 / 중간 점검 문항 (필수, 10분)
# ====================================================================
raw = pd.read_csv('data/battery_process_data.csv')
print("평균 대체 후 평균:", round(raw['믹싱_온도'].fillna(raw['믹싱_온도'].mean()).mean(), 4))
print("중앙값 대체 후 평균:", round(raw['믹싱_온도'].fillna(raw['믹싱_온도'].median()).mean(), 4))

g = df['프레스_Gap']
q1, q3 = g.quantile(0.25), g.quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out = df[(g < lower) | (g > upper)]
print(f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
print(f"하한={lower:.2f}, 상한={upper:.2f}")
print("이상치 후보:", len(out), "건 / 그중 불량:", int(out['불량_여부'].sum()))


# ====================================================================
# 전처리 파이프라인 정리 / 실전 예고 — 이 규약이 처음 무너지는 곳 (필수, 5분)
# ====================================================================
wmg = pd.read_csv('data/wmg/wmg_cells_54.csv')
print("모양:", wmg.shape)
print("결측 총 개수:", int(wmg.isna().sum().sum()))
print("dropna() 후:", wmg.dropna().shape)

print(wmg['grav_dis_10c_mahg'].describe().round(4))


# ====================================================================
# EDA의 목적과 절차 — 모델링 전에 데이터에게 묻는 질문 목록
# ====================================================================
import pandas as pd
import numpy as np

# 3.7.1절의 clean_data() 정의를 먼저 실행해 둔 상태여야 한다
df = clean_data('data/battery_process_data.csv')

print(df.shape)
print(df.isna().sum().sum())     # 결측 완전 해소 확인


# ====================================================================
# 단변량 시각화 — 변수 하나의 얼굴을 그리다 / 한글 폰트 설정부터
# ====================================================================
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트
plt.rcParams['axes.unicode_minus'] = False      # 음수 부호 깨짐 방지


# ====================================================================
# 단변량 시각화 — 변수 하나의 얼굴을 그리다 / 히스토그램 — 분포의 전체 모양
# ====================================================================
num_cols = df.select_dtypes('number').columns.drop('불량_여부')

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), num_cols):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()

print(df[num_cols].skew().round(3))


# ====================================================================
# 단변량 시각화 — 변수 하나의 얼굴을 그리다 / 박스플롯 — 사분위수와 이상치의 요약
# ====================================================================
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, col in zip(axes, ['믹싱_RPM', '프레스_Gap', '최종_용량_mAh', '전극_면저항_mOhmcm2']):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()


# ====================================================================
# 단변량 시각화 — 변수 하나의 얼굴을 그리다 / countplot — 타깃(불량) 분포의 확인
# ====================================================================
sns.countplot(data=df, x='불량_여부')
plt.title('불량 여부 분포')
plt.show()

print(df['불량_여부'].value_counts())
print(f"불량률: {df['불량_여부'].mean():.2%}")


# ====================================================================
# 이변량·다변량 시각화 — 변수 사이의 관계를 그리다 / 산점도 — 프레스_Gap과 최종 용량
# ====================================================================
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='프레스_Gap', y='최종_용량_mAh',
                hue='불량_여부', palette={0: 'steelblue', 1: 'crimson'})
plt.title('프레스_Gap vs 최종 용량')
plt.show()


# ====================================================================
# 이변량·다변량 시각화 — 변수 사이의 관계를 그리다 / 그룹별 박스플롯 — 연속형 변수 × 범주형 변수
# ====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(data=df, x='불량_여부', y='믹싱_RPM', ax=axes[0])
axes[0].set_title('불량 여부별 믹싱_RPM')
sns.boxplot(data=df, x='불량_여부', y='프레스_Gap', ax=axes[1])
axes[1].set_title('불량 여부별 프레스_Gap')
plt.tight_layout()
plt.show()

print(df.groupby('불량_여부')['믹싱_RPM'].describe().round(1))
print(df.groupby('불량_여부')['프레스_Gap'].describe().round(1))


# ====================================================================
# 이변량·다변량 시각화 — 변수 사이의 관계를 그리다 / pairplot — 다변량 관계의 파노라마
# ====================================================================
cols = ['프레스_Gap', '최종_용량_mAh', '전극_면저항_mOhmcm2', '믹싱_RPM', '불량_여부']
sns.pairplot(df[cols], hue='불량_여부', corner=True)
plt.show()


# ====================================================================
# 이변량·다변량 시각화 — 변수 사이의 관계를 그리다 / 상관 히트맵 — 관계 강도의 지도
# ====================================================================
corr = df[num_cols].corr()          # 피어슨 상관행렬

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1)
plt.title('공정·품질 변수 피어슨 상관 히트맵')
plt.tight_layout()
plt.show()


# ====================================================================
# 상관분석 — 관계를 숫자로 말하다 / 스피어만 순위상관 — 비선형 단조 관계
# ====================================================================
pearson  = df[num_cols].corr()                    # 피어슨
spearman = df[num_cols].corr(method='spearman')   # 스피어만

print(pearson.loc['프레스_Gap', '최종_용량_mAh'])   # -0.8704
print(spearman.loc['프레스_Gap', '최종_용량_mAh'])  # -0.852


# ====================================================================
# 상관분석 — 관계를 숫자로 말하다 / 실습 — 전체 상관행렬의 계산과 해석
# ====================================================================
corr = df[num_cols].corr()
print(corr.round(3))

# 타깃(불량_여부)과의 상관 — 이진 변수와 연속 변수의 피어슨 상관은
# 포인트-이연 상관(point-biserial)과 수학적으로 동일하다
corr_target = df[list(num_cols) + ['불량_여부']].corr()['불량_여부'].drop('불량_여부')
print(corr_target.sort_values(key=abs, ascending=False).round(3))


# ====================================================================
# Feature Engineering — 도메인 지식을 변수로 빚다 / 파생변수 — 비율, 차이, 구간화
# ====================================================================
# 비율: 단위 Gap당 프레스 압력 — 압연 강도의 대리 지표
df['압력_Gap_비율'] = df['프레스_압력'] / df['프레스_Gap']

# 차이: 건조로와 믹싱 공정의 온도 차 — 공정 간 열 이력 변화
df['온도_차이'] = df['건조로_1구간_온도'] - df['믹싱_온도']

df['Gap_구간'] = pd.cut(df['프레스_Gap'], bins=[0, 90, 100, 999],
                       labels=['얇음', '정상', '두꺼움'])
print(df['Gap_구간'].value_counts())
print(df.groupby('Gap_구간', observed=True)['불량_여부'].mean().round(4))


# ====================================================================
# Feature Engineering — 도메인 지식을 변수로 빚다 / 범주 인코딩 — 원-핫
# ====================================================================
dummies = pd.get_dummies(df['Gap_구간'], prefix='Gap', dtype=int)
print(pd.concat([df['Gap_구간'], dummies], axis=1).head(3))


# ====================================================================
# Feature Engineering — 도메인 지식을 변수로 빚다 / 상호작용항
# ====================================================================
df['RPMxGap'] = df['믹싱_RPM'] * df['프레스_Gap']
print(df['RPMxGap'].corr(df['불량_여부']).round(3))   # 0.226


# ====================================================================
# Feature Engineering — 도메인 지식을 변수로 빚다 / 도메인 기반 파생 — RPM 관리범위 이탈 플래그의 실검증
# ====================================================================
df['RPM_이탈'] = ((df['믹싱_RPM'] < 1600) | (df['믹싱_RPM'] > 1900)).astype(int)

print(df['RPM_이탈'].value_counts())
print(pd.crosstab(df['RPM_이탈'], df['불량_여부']))
print(df.groupby('RPM_이탈')['불량_여부'].mean().round(4))


# ====================================================================
# Feature Engineering — 도메인 지식을 변수로 빚다 / 특징 선택 개요 — 분산 필터와 상관 필터
# ====================================================================
high = corr.abs().where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
print(high[high > 0.8].round(3))


# ====================================================================
# 차원 축소 개요 — PCA로 데이터의 뼈대 찾기 / 실습 — 설명분산비와 2차원 시각화
# ====================================================================
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = StandardScaler().fit_transform(df[num_cols])   # 8개 수치 변수 표준화
pca = PCA()
Z = pca.fit_transform(X)

print(np.round(pca.explained_variance_ratio_, 3))
print(np.round(np.cumsum(pca.explained_variance_ratio_), 3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(range(1, 9), pca.explained_variance_ratio_)
axes[0].plot(range(1, 9), np.cumsum(pca.explained_variance_ratio_), 'ro-')
axes[0].set_xlabel('주성분'); axes[0].set_title('설명분산비(막대)와 누적(선)')
axes[1].scatter(Z[:, 0], Z[:, 1], c=df['불량_여부'], cmap='coolwarm', s=12)
axes[1].set_xlabel('PC1 (33.2%)'); axes[1].set_ylabel('PC2 (13.7%)')
axes[1].set_title('주성분 공간의 로트 분포')
plt.tight_layout()
plt.show()


# ====================================================================
# EDA 리포트 정리 — 품질 영향 인자 후보의 도출 / 종합 코드 (배포 스크립트 실행·해설)
# ====================================================================
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


# ====================================================================
# 연습문제 / 해답
# ====================================================================
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

df['Gap_구간4'] = pd.cut(df['프레스_Gap'], bins=[0, 85, 95, 105, 999],
                        labels=['매우얇음', '얇음', '정상', '두꺼움'])
print(df.groupby('Gap_구간4', observed=True)['불량_여부']
        .agg(건수='size', 불량='sum', 불량률='mean').round(4))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(data=df, x='불량_여부', y='전극_면저항_mOhmcm2', ax=axes[0])
sns.boxplot(data=df, x='불량_여부', y='최종_용량_mAh', ax=axes[1])
plt.tight_layout(); plt.show()

print(df.groupby('불량_여부')['전극_면저항_mOhmcm2'].mean().round(1))
print(df.groupby('불량_여부')['최종_용량_mAh'].mean().round(1))

proc = ['믹싱_RPM', '믹싱_온도', '코팅_토출압력',
        '건조로_1구간_온도', '프레스_압력', '프레스_Gap']
Xp = StandardScaler().fit_transform(df[proc])
pca6 = PCA().fit(Xp)
print(np.round(pca6.explained_variance_ratio_, 3))
print(np.round(np.cumsum(pca6.explained_variance_ratio_), 3))


# ====================================================================
# 실전 정제 ① 현장 엑셀 워크북 해체하기 (필수, 35분) / 첫걸음 — 엑셀은 '격자'로 먼저 연다
# ====================================================================
import pandas as pd
import numpy as np

XL = 'data/wmg/raw/Half-cell (Cathode) Electrochemical Performance.xlsx'

xls = pd.ExcelFile(XL)
print("시트 개수:", len(xls.sheet_names))
print(xls.sheet_names[:5], "...", xls.sheet_names[-1])

grid = pd.read_excel(XL, sheet_name='Group1', header=None)
print("격자 모양:", grid.shape)
print("0행:", grid.iloc[0].tolist())

print(grid.iloc[[20, 32, 80], 0:6].to_string())

g1 = pd.read_excel(XL, sheet_name='Group1')
g2 = pd.read_excel(XL, sheet_name='Group2')
print("Group1 컬럼:", list(g1.columns))
print("Group2 컬럼:", list(g2.columns))
print("그냥 concat:", pd.concat([g1, g2], ignore_index=True).shape)


# ====================================================================
# 실전 정제 ① 현장 엑셀 워크북 해체하기 (필수, 35분) / (섹션, 항목, 셀ID) 3중키 파서 직접 짜기
# ====================================================================
def read_group_sheet(path, sheet):
    """시트 한 장 → (조건, 섹션, 항목, 셀ID, 값) 다섯 열의 tidy 표."""
    g = pd.read_excel(path, sheet_name=sheet, header=None)
    cell_ids = [g.iat[0, c] for c in (2, 3, 4)]   # 0행의 2~4열 = 셀 ID 세 개
    g[0] = g[0].ffill()                           # 병합으로 비어 있는 섹션명 채우기
    recs = []
    for _, row in g.iterrows():
        if pd.isna(row[1]):                       # 항목명이 없는 빈 줄은 건너뛴다
            continue
        for k, c in enumerate((2, 3, 4)):
            recs.append({'group': sheet, 'section': row[0], 'label': row[1],
                         'cell_id': cell_ids[k], 'value': row[c]})
    return pd.DataFrame(recs)

t1 = read_group_sheet(XL, 'Group1')
print("Group1 tidy:", t1.shape)
print(t1[t1['label'] == 'At C/20'].head(6).to_string(index=False))

sheets = ['Group%d' % i for i in range(1, 19)]
tidy = pd.concat([read_group_sheet(XL, s) for s in sheets], ignore_index=True)
tidy['value_num'] = pd.to_numeric(tidy['value'], errors='coerce')

print("tidy 전체:", tidy.shape)
print("셀 개수:", tidy['cell_id'].nunique(), " 섹션 개수:", tidy['section'].nunique())
print("숫자 변환 실패:", int(tidy['value_num'].isna().sum()),
      "건 (그중 원래 빈칸", int(tidy['value'].isna().sum()), "건)")

sel = tidy[(tidy['section'] == 'Gravimetric Discharge Capacity (mAh/g)') &
           (tidy['label'].isin(['At C/20', 'At 5C', 'At 10C']))]
wide = sel.pivot_table(index='cell_id', columns='label', values='value_num')
print(wide.shape)
print(wide.head(4).round(2))

ref = pd.read_csv('data/wmg/wmg_cells_54.csv').set_index('cell_id')
diff = (wide['At C/20'] - ref['grav_dis_c20_mahg']).abs()
print("배포 CSV와의 최대 차이:", float(diff.max()))
print("일치한 셀:", int((diff < 1e-9).sum()), "/", len(diff))


# ====================================================================
# 실전 정제 ① 현장 엑셀 워크북 해체하기 (필수, 35분) / 원본의 계산 오류를 스스로 찾아내기 — 변동계수 스캔
# ====================================================================
st = (tidy.groupby(['group', 'section', 'label'])['value_num']
          .agg(['mean', 'std', 'count']))
st = st[(st['count'] == 3) & (st['mean'].abs() > 0)]   # 3반복이 온전한 항목만
st['cv'] = st['std'] / st['mean'].abs()

print("검사 대상:", len(st), "개 (조건 × 항목)")
print("CV 중앙값:", round(float(st['cv'].median()), 4))
print(st['cv'].sort_values(ascending=False).head(5).round(3).to_string())

grid = pd.read_excel(XL, sheet_name='Group1', header=None)
sec = grid[0].ffill()
row = grid[(sec == 'Gravimetric Charge Capacity (mAh/g)') & (grid[1] == 'At C/20')].iloc[0]

vals = row[[2, 3, 4, 5]]
vals.index = ['DD001', 'DD002', 'DD056', '워크북 Mean']
print(vals.astype(float).round(4).to_string())

v3 = vals[['DD001', 'DD002', 'DD056']].astype(float)
print("0을 그대로 둔 평균      :", round(float(v3.mean()), 4))
print("0을 결측으로 되돌린 평균:", round(float(v3.replace(0, np.nan).mean()), 4))

row2 = grid[(sec == 'Capacity Charge (mAh)') & (grid[1] == 'At C/20')].iloc[0]
v2 = row2[[2, 3, 4, 5]]
v2.index = ['DD001', 'DD002', 'DD056', '워크북 Mean']
print(v2.astype(float).round(4).to_string())


# ====================================================================
# 실전 정제 ① 현장 엑셀 워크북 해체하기 (필수, 35분) / 병합 2단 헤더와 엑셀 날짜 — 두 번째 워크북
# ====================================================================
CAL = 'data/wmg/raw/Intermediate measurements during calendering.xlsx'
print(pd.ExcelFile(CAL).sheet_names)

c0 = pd.read_excel(CAL, sheet_name='Cathode')
print("모양:", c0.shape)
print("컬럼 앞 7개:", list(c0.columns)[:7])

c = pd.read_excel(CAL, sheet_name='Cathode', header=[0, 1])
print("모양:", c.shape)
for col in list(c.columns)[:7]:
    print("  ", col)

def flatten(col):
    top, bot = str(col[0]), str(col[1])
    top = '' if top.startswith('Unnamed') else top.strip()
    bot = '' if bot.startswith('Unnamed') else bot.strip()
    return (top + ' | ' + bot).strip(' |')

c.columns = [flatten(x) for x in c.columns]
for col in list(c.columns)[:7]:
    print("  ", col)

c = c.rename(columns={
    'Calendering conditions | Target coating weight (GSM)': 'gsm',
    'Calendering conditions | Roll temperature (oC)': 'roll_temp',
    'Calendering date': 'cal_date'})

print("행 수:", len(c), " / 설계상 조건 수: 18")
print(c[['No', 'gsm', 'roll_temp']].iloc[16:24].to_string())
print("목표 코팅중량 고유값:",
      sorted(pd.to_numeric(c['gsm'], errors='coerce').dropna().unique().tolist()))
print("39행 그대로 낸 평균:",
      round(float(pd.to_numeric(c['gsm'], errors='coerce').mean()), 2))

cond = c.iloc[:18].copy()
print("조건 표:", cond[['No', 'gsm', 'roll_temp', 'cal_date']].shape)
print("목표 코팅중량 고유값:", sorted(cond['gsm'].unique().tolist()))

print("dtype:", cond['cal_date'].dtype)
print("첫 값:", repr(cond['cal_date'].iloc[0]))

cond['cal_date'] = pd.to_datetime(cond['cal_date'])
print("변환 후 dtype:", cond['cal_date'].dtype)

serial = (cond['cal_date'] - pd.Timestamp('1899-12-30')).dt.days
print("엑셀 serial:", sorted(serial.unique().tolist()))

print(pd.crosstab(cond['cal_date'].dt.date, cond['roll_temp']).to_string())


# ====================================================================
# 실전 정제 ① 현장 엑셀 워크북 해체하기 (필수, 35분) / NaN이 아닌 실패값 — 0.0009 mAh/g 일곱 셀
# ====================================================================
cells = pd.read_csv('data/wmg/wmg_cells_54.csv')
y = 'grav_dis_10c_mahg'
print("결측:", int(cells[y].isna().sum()), "건")
print(cells[y].sort_values().head(9).round(4).to_string())

fail = cells[y] < 1.0
print("1 mAh/g 미만:", int(fail.sum()), "셀")
print(cells.loc[fail, ['cell_id', 'coat_weight_level', 'density_level',
                       'roll_temp_c', y]].to_string(index=False))

print("평균 (54셀 전부):", round(float(cells[y].mean()), 2))
print("평균 (실패 7셀 제외):", round(float(cells.loc[~fail, y].mean()), 2))
print("---")
print("전체 54셀 :", round(float(cells['porosity_pct'].corr(cells[y])), 4))
ok = cells[~fail]
print("실패 제외 47셀:", round(float(ok['porosity_pct'].corr(ok[y])), 4))


# ====================================================================
# 실전 정제 ① 현장 엑셀 워크북 해체하기 (필수, 35분) / 이론 한계를 넘는 값 — DD020을 어떻게 판정할 것인가
# ====================================================================
Y = 'grav_dis_c20_mahg'
print(cells[Y].describe().round(2).to_string())

print(cells.nlargest(3, Y)[['cell_id', 'group', Y]].round(2).to_string(index=False))

print(cells[cells['group'] == 3][['cell_id', 'coat_weight_level', 'roll_temp_c',
                                  'density_level', Y]].round(2).to_string(index=False))

sd = cells.groupby('group')[Y].std()
print("group 3 의 SD :", round(float(sd.loc[3]), 2))
print("나머지 17조건 SD 중앙값:", round(float(sd.drop(3).median()), 2))
print("배율:", round(float(sd.loc[3] / sd.drop(3).median()), 1), "배")

asi = [c for c in cells.columns if c.startswith('asi_')]
print("ASI 열:", len(asi), "개")
print("DD020 의 ASI 결측:",
      int(cells.loc[cells['cell_id'] == 'DD020', asi].isna().sum(axis=1).iloc[0]), "개")
print("나머지 53셀의 ASI 결측 합계:",
      int(cells.loc[cells['cell_id'] != 'DD020', asi].isna().sum().sum()), "개")


# ====================================================================
# 실전 정제 ① 현장 엑셀 워크북 해체하기 (필수, 35분) / 이 절의 정제 규약 — clean_wmg()
# ====================================================================
def clean_wmg(path):
    """WMG 하프셀 표 정제 — 실패값·불가값을 결측으로 되돌리고 플래그를 남긴다."""
    d = pd.read_csv(path)
    # 1) 10C 시험 실패(0.0009 mAh/g)를 결측으로 되돌리되, 사실을 플래그로 보존
    d['flag_10c_fail'] = (d['grav_dis_10c_mahg'] < 1.0).astype(int)
    d.loc[d['flag_10c_fail'] == 1, 'grav_dis_10c_mahg'] = np.nan
    # 2) 이론 한계 초과는 지우지 않고 플래그만 단다
    d['flag_over_theory'] = (d['grav_dis_c20_mahg'] > 200).astype(int)
    return d

w = clean_wmg('data/wmg/wmg_cells_54.csv')
print("모양:", w.shape)
print("10C 실패 플래그:", int(w['flag_10c_fail'].sum()), "셀")
print("이론 초과 플래그:", int(w['flag_over_theory'].sum()), "셀")
print("정제 후 10C 평균:", round(float(w['grav_dis_10c_mahg'].mean()), 2))
print("정제 후 10C 결측:", int(w['grav_dis_10c_mahg'].isna().sum()), "건")


# ====================================================================
# 실전 정제 ② 통계 규칙이 못 잡는 물리적 불가값 (필수, 25분) / 이 절에서 쓸 데이터 — 18650 셀 수명 로그
# ====================================================================
rul = pd.read_csv('data/rul/battery_rul_original.csv')
print("모양:", rul.shape)
print("결측:", int(rul.isna().sum().sum()), "건 · 중복 행:", int(rul.duplicated().sum()), "건")
print(list(rul.columns))


# ====================================================================
# 실전 정제 ② 통계 규칙이 못 잡는 물리적 불가값 (필수, 25분) / describe() 한 줄이 던지는 경고
# ====================================================================
t = ['Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Time at 4.15V (s)']
print(rul[t].describe().loc[['min', '50%', 'max']].round(1).to_string())


# ====================================================================
# 실전 정제 ② 통계 규칙이 못 잡는 물리적 불가값 (필수, 25분) / IQR도 z-score도 이것을 잡지 못한다
# ====================================================================
s = rul['Time at 4.15V (s)']
q1, q3 = s.quantile(0.25), s.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"IQR 경계: {lo:.1f} ~ {hi:.1f}")
print("IQR 규칙 검출:", int(((s < lo) | (s > hi)).sum()), "건")

z = (s - s.mean()) / s.std()
print("|z| > 3 검출:", int((z.abs() > 3).sum()), "건")
print("실제 음수:", int((s < 0).sum()), "건 (최솟값", round(float(s.min()), 1), "초)")
print("음수 중 IQR 규칙이 잡은 것:", int(((s < 0) & ((s < lo) | (s > hi))).sum()), "건")
print("음수 중 |z|>3 이 잡은 것 :", int(((s < 0) & (z.abs() > 3)).sum()), "건")


# ====================================================================
# 실전 정제 ② 통계 규칙이 못 잡는 물리적 불가값 (필수, 25분) / 도메인 규칙 다섯 줄이 이긴다
# ====================================================================
rules = {
    '음수 시간 · 전압강하 구간': rul['Decrement 3.6-3.4V (s)'] < 0,
    '음수 시간 · 4.15V 도달': rul['Time at 4.15V (s)'] < 0,
    '부분 > 전체 · 4.15V 도달 > 총 충전': rul['Time at 4.15V (s)'] > rul['Charging time (s)'],
    '부분 > 전체 · 전압강하 > 총 방전':
        rul['Decrement 3.6-3.4V (s)'].abs() > rul['Discharge Time (s)'],
    '부분 > 전체 · CC구간 > 총 충전':
        rul['Time constant current (s)'] > rul['Charging time (s)'],
}

bad = pd.Series(False, index=rul.index)
for name, m in rules.items():
    print(f"{name}: {int(m.sum())}건")
    bad = bad | m

print("---")
print("합집합:", int(bad.sum()), "행 =", round(100 * float(bad.mean()), 2), "%")
print("제거 후:", rul[~bad].shape)


# ====================================================================
# 실전 정제 ② 통계 규칙이 못 잡는 물리적 불가값 (필수, 25분) / ID 컬럼이 없는 로그에서 로트를 복원하기
# ====================================================================
reset = rul['Cycle_Index'].diff() < 0
print("사이클 번호가 되돌아간 지점:", int(reset.sum()), "곳")

rul['cell_id'] = reset.cumsum() + 1
print("복원된 셀 개수:", rul['cell_id'].nunique())

summ = rul.groupby('cell_id').agg(행수=('Cycle_Index', 'size'),
                                  EOL=('Cycle_Index', 'max'))
summ['EOL'] = summ['EOL'].astype(int)
summ['불가값행'] = bad.groupby(rul['cell_id']).sum()
print(summ.head(5).to_string())
print("행수:", int(summ['행수'].min()), "~", int(summ['행수'].max()),
      "· EOL:", int(summ['EOL'].min()), "~", int(summ['EOL'].max()))
print("불가값이 하나도 없는 셀:", int((summ['불가값행'] == 0).sum()), "개")

gap = summ['EOL'] - summ['행수']
print("셀별 누락 사이클:", int(gap.min()), "~", int(gap.max()), "· 합계:", int(gap.sum()))


# ====================================================================
# 실전 정제 ② 통계 규칙이 못 잡는 물리적 불가값 (필수, 25분) / 복원한 셀 ID가 곧바로 드러낸 것 — 타깃 누수 예고
# ====================================================================
rul['EOL'] = rul.groupby('cell_id')['Cycle_Index'].transform('max')
print("RUL == EOL − Cycle_Index 가 15,064행 전부에서 성립? ",
      bool((rul['RUL'] == rul['EOL'] - rul['Cycle_Index']).all()))
print("corr(Cycle_Index, RUL) =", round(float(rul['Cycle_Index'].corr(rul['RUL'])), 6))


# ====================================================================
# 실전 EDA — 파일명에서 공정 변수를 캐낸다 (필수, 25분) / 이 절에서 쓸 데이터 — 코팅 결함 이미지 라벨
# ====================================================================
lab = pd.read_csv('data/coatingvision/labels_original.csv')
print("모양:", lab.shape)
print(lab.head(3).to_string(index=False))


# ====================================================================
# 실전 EDA — 파일명에서 공정 변수를 캐낸다 (필수, 25분) / 먼저 라벨 구조부터 — 멀티라벨이라는 낯선 형식
# ====================================================================
labels = ['Surface_Crack', 'Delamination', 'Pinhole', 'unclassified']
print(pd.DataFrame({'양성수': lab[labels].sum(),
                    '비율(%)': (lab[labels].mean() * 100).round(1)}).to_string())
print()
print("한 장에 붙은 라벨 수:")
print(lab[labels].sum(axis=1).value_counts().sort_index().to_string())


# ====================================================================
# 실전 EDA — 파일명에서 공정 변수를 캐낸다 (필수, 25분) / 정규식 5분 — 문자열에서 규칙을 뽑아내는 도구
# ====================================================================
pat = (r'^R(?P<run>\d+)-(?P<gap>\d+)um-(?P<position>.+?)'
       r'_frame_(?P<frame>\d+)_patch_(?P<patch>\d+)\.png$')

ext = lab['original_file_name'].str.extract(pat)
print("추출 실패:", int(ext['run'].isna().sum()), "건")
print(ext.head(3).to_string(index=False))

lab['run'] = ext['run'].astype(int)
lab['gap'] = ext['gap'].astype(int)
lab['frame'] = ext['frame'].astype(int)
lab['position'] = ext['position'].str.replace('top-to-bottom-center', 'TBC')
lab['video_id'] = ('R' + lab['run'].astype(str) + '-'
                   + lab['gap'].astype(str) + '-' + lab['position'])
lab['frame_id'] = lab['video_id'] + '_f' + lab['frame'].astype(str)

print("런:", sorted(lab['run'].unique().tolist()))
print("코팅 갭:", sorted(lab['gap'].unique().tolist()))
print("영상:", lab['video_id'].nunique(), "개 · 프레임:", lab['frame_id'].nunique(),
      "개 · 패치:", len(lab), "개")


# ====================================================================
# 실전 EDA — 파일명에서 공정 변수를 캐낸다 (필수, 25분) / 갭별 결함률 크로스탭 — 그리고 즉시 따라붙는 경고
# ====================================================================
tab = lab.groupby('gap')[labels].mean().mul(100).round(1)
tab.insert(0, '패치수', lab.groupby('gap').size())
print(tab.to_string())

print(pd.crosstab(lab['gap'], lab['video_id']).to_string())


# ====================================================================
# 실전 EDA — 파일명에서 공정 변수를 캐낸다 (필수, 25분) / 결정적 반례 — 같은 700 µm, 89.6배 차이
# ====================================================================
g7 = lab[lab['gap'] == 700]
out = g7.groupby('run')['Delamination'].agg(패치수='size', 박리='sum', 검출률='mean')
out['검출률'] = (out['검출률'] * 100).round(2)
print(out.to_string())


# ====================================================================
# 실전 EDA — 파일명에서 공정 변수를 캐낸다 (필수, 25분) / 행은 독립이 아니다 — 패치·프레임·영상의 3층 구조
# ====================================================================
print("프레임당 패치 수 — 중앙값:", int(lab.groupby('frame_id').size().median()),
      "· 최대:", int(lab.groupby('frame_id').size().max()))


# ====================================================================
# 실전 EDA — 파일명에서 공정 변수를 캐낸다 (필수, 25분) / 결함을 눈으로 보기
# ====================================================================
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

idx = pd.read_csv('data/coatingvision/curated_index.csv')
pick = ['image_1019.jpg', 'image_1501.jpg', 'image_1729.jpg']
sel = idx[idx['file_name'].isin(pick)].set_index('file_name').loc[pick]
print(sel[['run', 'coating_gap_um', 'frame', 'patch', 'defect_set']].to_string())

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, fn in zip(axes, pick):
    img = mpimg.imread('data/coatingvision/images_curated/' + fn)
    ax.imshow(img)
    ax.set_title(f"{fn}\n{sel.loc[fn, 'defect_set']}", fontsize=9)
    ax.axis('off')
    print(fn, "→ 배열 크기:", img.shape)
plt.tight_layout()
plt.show()


# ====================================================================
# 고차원 데이터와의 첫 대면 — SECOM (선택 학습, 20분) / 이 절에서 쓸 데이터 — 그리고 중요한 경고
# ====================================================================
sec = pd.read_csv('data/secom/secom_merged.csv', parse_dates=['timestamp'])
sensors = [c for c in sec.columns if c.startswith('sensor_')]

print("모양:", sec.shape)
print("센서 컬럼 수:", len(sensors), " (문서 기재: 591)")
print(sec['label'].value_counts().to_string())
print("불량률:", round(100 * float(sec['fail'].mean()), 2), "%")


# ====================================================================
# 고차원 데이터와의 첫 대면 — SECOM (선택 학습, 20분) / dropna()가 데이터를 전멸시키는 순간
# ====================================================================
row_na = sec[sensors].isna().sum(axis=1)
print("전체 결측률:", round(100 * float(sec[sensors].isna().mean().mean()), 2), "%")
print("결측이 0인 행:", int((row_na == 0).sum()), "개")
print("행별 결측 센서 수 — 중앙값:", int(row_na.median()), "· 최대:", int(row_na.max()))
print("dropna() 결과:", sec.dropna().shape)


# ====================================================================
# 고차원 데이터와의 첫 대면 — SECOM (선택 학습, 20분) / 열부터 정리한다 — 정보가 없는 센서 골라내기
# ====================================================================
miss = sec[sensors].isna().mean()
nuniq = sec[sensors].nunique()

print("결측이 0인 컬럼:", int((miss == 0).sum()), "개")
print("결측 50% 초과 컬럼:", int((miss > 0.5).sum()), "개 (최대",
      round(100 * float(miss.max()), 2), "%)")
print("값이 한 종류뿐인(분산 0) 컬럼:", int((nuniq <= 1).sum()), "개 =",
      round(100 * float((nuniq <= 1).mean()), 1), "%")

drop = set(nuniq[nuniq <= 1].index) | set(miss[miss > 0.5].index)
keep = [c for c in sensors if c not in drop]
print("제거 대상 합집합:", len(drop), "개 → 남는 센서:", len(keep), "개")


# ====================================================================
# 고차원 데이터와의 첫 대면 — SECOM (선택 학습, 20분) / 남은 446개도 그대로는 못 쓴다
# ====================================================================
sd = sec[keep].std()
print("표준편차 최소:", round(float(sd.min()), 6), "· 최대:", round(float(sd.max()), 1))
print("최대/최소 비:", f"{float(sd.max() / sd.min()):,.0f}", "배")
print("---")
print("전부 정상으로 찍는 모델의 정확도:",
      round(100 * (1 - float(sec['fail'].mean())), 2), "%")
corr = sec[keep].corrwith(sec['fail']).abs().sort_values(ascending=False)
print("|corr(센서, 불량)| 상위 5개:")
print(corr.head(5).round(4).to_string())


# ====================================================================
# 고차원 데이터와의 첫 대면 — SECOM (선택 학습, 20분) / 시간이 흐르면 공정도 변한다
# ====================================================================
m = sec.groupby(sec['timestamp'].dt.to_period('M'))['fail'].agg(
    건수='size', 불량='sum', 불량률='mean')
m['불량률'] = (m['불량률'] * 100).round(2)
print(m.to_string())
print("중복 timestamp:", int(sec['timestamp'].duplicated().sum()), "건")

