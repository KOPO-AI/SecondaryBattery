# -*- coding: utf-8 -*-
"""4장 예제 — 4.18 실전 분류 — 극단 불균형·고차원·시간 드리프트 (UCI SECOM) `선택 학습(자율 복습)` 〔실습 30분〕

교재 출처 : manuscript/41_ch4_분류.md
실행 방법 : example 폴더에서  python "4장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 15_4-15_연습문제.py, 16_4-16_실전_회귀_①.py, 17_4-17_실전_회귀_②.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd
import numpy as np

# --------------------------------------------------------------------
# [첫 30초 — `dropna()`가 데이터를 전멸시킨다]
# --------------------------------------------------------------------
sec = pd.read_csv('data/secom/secom_merged.csv', parse_dates=['timestamp'])

print(sec.shape)
print(sec['fail'].value_counts())
print(f"불량률 = {sec['fail'].mean()*100:.2f}%")

# 교재 실행 결과 ------------------------------------------------------
#   (1567, 593)
#   fail
#   0    1463
#   1     104
#   Name: count, dtype: int64
#   불량률 = 6.64%

# --------------------------------------------------------------------
# [첫 30초 — `dropna()`가 데이터를 전멸시킨다]
# --------------------------------------------------------------------
print("결측이 하나도 없는 행 =", int((sec.isna().sum(axis=1) == 0).sum()), "행")
print("dropna() 결과        =", sec.dropna().shape)

# 교재 실행 결과 ------------------------------------------------------
#   결측이 하나도 없는 행 = 0 행
#   dropna() 결과        = (0, 593)

# --------------------------------------------------------------------
# [590개 익명 센서 정리하기]
# --------------------------------------------------------------------
센서 = [c for c in sec.columns if c.startswith('sensor_')]
print("센서 개수 :", len(센서))

분산0   = [c for c in 센서 if sec[c].nunique(dropna=True) <= 1]
결측많음 = [c for c in 센서 if sec[c].isna().mean() > 0.5]

print("분산이 0인 센서    :", len(분산0), "개")
print("결측 50% 초과 센서 :", len(결측많음), "개")

쓸센서 = [c for c in 센서 if c not in 분산0 and c not in 결측많음]
print("남은 센서          :", len(쓸센서), "개")

# 교재 실행 결과 ------------------------------------------------------
#   센서 개수 : 590
#   분산이 0인 센서    : 116 개
#   결측 50% 초과 센서 : 28 개
#   남은 센서          : 446 개

# --------------------------------------------------------------------
# [정확도 93.4%의 함정, 그리고 class_weight가 듣지 않을 때]
# --------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score, confusion_matrix)

X, y = sec[쓸센서], sec['fail']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)
print("학습:", y_tr.value_counts().to_dict())
print("평가:", y_te.value_counts().to_dict())
print(f"전부 정상이라 답할 때의 정확도 = {1 - y_te.mean():.4f}")

# 교재 실행 결과 ------------------------------------------------------
#   학습: {0: 1097, 1: 78}
#   평가: {0: 366, 1: 26}
#   전부 정상이라 답할 때의 정확도 = 0.9337

# --------------------------------------------------------------------
# [정확도 93.4%의 함정, 그리고 class_weight가 듣지 않을 때]
# --------------------------------------------------------------------
def 만들기(분류기):
    return Pipeline([('결측대체', SimpleImputer(strategy='median')),
                     ('표준화',   StandardScaler()),
                     ('분류기',   분류기)])


def 성적표(이름, y_true, y_pred, 점수=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    pr = f"  PR-AUC={average_precision_score(y_true, 점수):.3f}" if 점수 is not None else ""
    print(f"{이름:22s} 정확도={accuracy_score(y_true, y_pred):.3f} "
          f"정밀도={precision_score(y_true, y_pred, zero_division=0):.3f} "
          f"재현율={recall_score(y_true, y_pred):.3f} "
          f"F1={f1_score(y_true, y_pred):.3f}  검출 {tp}/{tp+fn}건, 과검 {fp}건{pr}")


성적표('전부-정상 더미', y_te, np.zeros(len(y_te), dtype=int))

로짓 = 만들기(LogisticRegression(max_iter=2000, random_state=42)).fit(X_tr, y_tr)
성적표('로지스틱(기본)', y_te, 로짓.predict(X_te), 로짓.predict_proba(X_te)[:, 1])

로짓B = 만들기(LogisticRegression(max_iter=2000, class_weight='balanced',
                              random_state=42)).fit(X_tr, y_tr)
성적표('로지스틱(balanced)', y_te, 로짓B.predict(X_te), 로짓B.predict_proba(X_te)[:, 1])

숲 = 만들기(RandomForestClassifier(n_estimators=300, random_state=42,
                                n_jobs=-1)).fit(X_tr, y_tr)
성적표('RF(기본)', y_te, 숲.predict(X_te), 숲.predict_proba(X_te)[:, 1])

숲B = 만들기(RandomForestClassifier(n_estimators=300,
                                class_weight='balanced_subsample',
                                random_state=42, n_jobs=-1)).fit(X_tr, y_tr)
성적표('RF(balanced)', y_te, 숲B.predict(X_te), 숲B.predict_proba(X_te)[:, 1])

print(f"\n무작위로 찍었을 때의 PR-AUC 기대값 = {y_te.mean():.3f}")

# 교재 실행 결과 ------------------------------------------------------
#   전부-정상 더미               정확도=0.934 정밀도=0.000 재현율=0.000 F1=0.000  검출 0/26건, 과검 0건
#   로지스틱(기본)               정확도=0.880 정밀도=0.161 재현율=0.192 F1=0.175  검출 5/26건, 과검 26건  PR-AUC=0.142
#   로지스틱(balanced)         정확도=0.834 정밀도=0.102 재현율=0.192 F1=0.133  검출 5/26건, 과검 44건  PR-AUC=0.155
#   RF(기본)                 정확도=0.934 정밀도=0.000 재현율=0.000 F1=0.000  검출 0/26건, 과검 0건  PR-AUC=0.193
#   RF(balanced)           정확도=0.934 정밀도=0.000 재현율=0.000 F1=0.000  검출 0/26건, 과검 0건  PR-AUC=0.209
#   무작위로 찍었을 때의 PR-AUC 기대값 = 0.066

# --------------------------------------------------------------------
# [판정선을 옮긴다 — 여기서 유일하게 듣는 손잡이]
# --------------------------------------------------------------------
확률 = 숲.predict_proba(X_te)[:, 1]        # RF(기본)의 불량 확률

print(" 임계값  판정건수  검출  과검  정밀도  재현율    F1")
for t in [0.50, 0.30, 0.20, 0.15, 0.10, 0.05]:
    판정 = (확률 >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, 판정).ravel()
    print(f"  {t:.2f}    {판정.sum():5d}   {tp:3d}  {fp:4d}   "
          f"{precision_score(y_te, 판정, zero_division=0):.3f}  "
          f"{recall_score(y_te, 판정):.3f}  {f1_score(y_te, 판정):.3f}")

# 교재 실행 결과 ------------------------------------------------------
#    임계값  판정건수  검출  과검  정밀도  재현율    F1
#     0.50        0     0     0   0.000  0.000  0.000
#     0.30       14     4    10   0.286  0.154  0.200
#     0.20       32     7    25   0.219  0.269  0.241
#     0.15       55    12    43   0.218  0.462  0.296
#     0.10      104    16    88   0.154  0.615  0.246
#     0.05      203    22   181   0.108  0.846  0.192

# --------------------------------------------------------------------
# [현장의 언어로 바꾸기 — "재검 예산이 하루 20건이라면"]
# --------------------------------------------------------------------
기저 = y_te.mean()

print(" 재검 예산   검출 불량   적중률   무작위 대비")
for k in [10, 20, 40, 80]:
    상위 = np.argsort(-확률)[:k]              # 확률 내림차순 상위 K개의 위치
    검출 = int(y_te.values[상위].sum())
    print(f"  상위 {k:3d}건   {검출:5d}건   {검출/k*100:5.1f}%   {(검출/k)/기저:.1f}배")

# 교재 실행 결과 ------------------------------------------------------
#    재검 예산   검출 불량   적중률   무작위 대비
#     상위  10건       4건    40.0%   6.0배
#     상위  20건       5건    25.0%   3.8배
#     상위  40건       9건    22.5%   3.4배
#     상위  80건      15건    18.8%   2.8배

# --------------------------------------------------------------------
# [시간순으로 나누면 전부 무너진다]
# --------------------------------------------------------------------
월별 = sec.groupby(sec['timestamp'].dt.to_period('M'))['fail'].agg(['size', 'sum', 'mean'])
월별['mean'] = (월별['mean'] * 100).round(2)
print(월별.to_string())

# 교재 실행 결과 ------------------------------------------------------
#              size  sum   mean
#   timestamp
#   2008-07      63   14  22.22
#   2008-08     555   51   9.19
#   2008-09     590   17   2.88
#   2008-10     359   22   6.13

# --------------------------------------------------------------------
# [시간순으로 나누면 전부 무너진다]
# --------------------------------------------------------------------
경계 = int(len(sec) * 0.75)          # 데이터는 시각 오름차순으로 정렬돼 있다
X_tr2, X_te2 = X.iloc[:경계], X.iloc[경계:]
y_tr2, y_te2 = y.iloc[:경계], y.iloc[경계:]

print(f"학습 구간: {sec['timestamp'].iloc[0]:%Y-%m-%d} ~ "
      f"{sec['timestamp'].iloc[경계-1]:%Y-%m-%d}  불량 {y_tr2.mean()*100:.2f}%")
print(f"평가 구간: {sec['timestamp'].iloc[경계]:%Y-%m-%d} ~ "
      f"{sec['timestamp'].iloc[-1]:%Y-%m-%d}  불량 {y_te2.mean()*100:.2f}%")

숲2 = 만들기(RandomForestClassifier(n_estimators=300, random_state=42,
                                n_jobs=-1)).fit(X_tr2, y_tr2)
확률2 = 숲2.predict_proba(X_te2)[:, 1]

print(f"시간순 분할 PR-AUC = {average_precision_score(y_te2, 확률2):.3f} "
      f"(기저 {y_te2.mean():.3f})")
for k in [10, 20, 40]:
    상위 = np.argsort(-확률2)[:k]
    검출 = int(y_te2.values[상위].sum())
    print(f"  상위 {k:3d}건   {검출:3d}건 검출   무작위 대비 {(검출/k)/y_te2.mean():.1f}배")

# 교재 실행 결과 ------------------------------------------------------
#   학습 구간: 2008-07-19 ~ 2008-09-29  불량 6.81%
#   평가 구간: 2008-09-29 ~ 2008-10-17  불량 6.12%
#   시간순 분할 PR-AUC = 0.072 (기저 0.061)
#     상위  10건     0건 검출   무작위 대비 0.0배
#     상위  20건     0건 검출   무작위 대비 0.0배
#     상위  40건     2건 검출   무작위 대비 0.8배

