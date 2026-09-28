# -*- coding: utf-8 -*-
"""3장 예제 — 3.11 상관분석 — 관계를 숫자로 말하다

교재 출처 : manuscript/31_ch3_eda_fe.md
실행 방법 : example 폴더에서  python "3장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 08_3-8_EDA의_목적과_절차.py, 09_3-9_단변량_시각화.py, 10_3-10_이변량·다변량_시각화.py
실행 상태 : 단독 실행 확인됨(선행 절 준비 코드 포함).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
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

num_cols = df.select_dtypes('number').columns.drop('불량_여부')
# ==================== [선행 절 준비 코드 끝] ==========================

# --------------------------------------------------------------------
# [스피어만 순위상관 — 비선형 단조 관계]
# --------------------------------------------------------------------
pearson  = df[num_cols].corr()                    # 피어슨
spearman = df[num_cols].corr(method='spearman')   # 스피어만

print(pearson.loc['프레스_Gap', '최종_용량_mAh'])   # -0.8704
print(spearman.loc['프레스_Gap', '최종_용량_mAh'])  # -0.852

# --------------------------------------------------------------------
# [실습 — 전체 상관행렬의 계산과 해석]
# --------------------------------------------------------------------
corr = df[num_cols].corr()
print(corr.round(3))

# 타깃(불량_여부)과의 상관 — 이진 변수와 연속 변수의 피어슨 상관은
# 포인트-이연 상관(point-biserial)과 수학적으로 동일하다
corr_target = df[list(num_cols) + ['불량_여부']].corr()['불량_여부'].drop('불량_여부')
print(corr_target.sort_values(key=abs, ascending=False).round(3))

