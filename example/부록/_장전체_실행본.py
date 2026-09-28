# -*- coding: utf-8 -*-
"""부록 — 장 전체 예제 실행본

교재 부록의 파이썬 코드를 등장 순서대로 이어 붙였다.
절별 파일은 앞 절에서 만든 변수를 이어 쓰는 경우가 있어 단독 실행이 안 될 수 있는데,
이 파일은 처음부터 끝까지 순서대로 실행되므로 그대로 돌려 볼 수 있다.

실행 : example 폴더에서  python "부록/_장전체_실행본.py"
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""
import matplotlib
matplotlib.use("Agg")   # 창을 띄우지 않고 그림을 파일로만 처리


# ====================================================================
# 부록 A. 주 실습 데이터셋 명세 — battery_process_data.csv / A.4 로드 예시
# ====================================================================
import pandas as pd
df = pd.read_csv("data/battery_process_data.csv")
print(df.shape)
print(df["불량_여부"].sum())
print(round(df["전극_면저항_mOhmcm2"].mean(), 1))
print(round(df["프레스_Gap"].corr(df["전극_면저항_mOhmcm2"]), 4))

