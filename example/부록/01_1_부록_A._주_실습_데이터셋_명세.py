# -*- coding: utf-8 -*-
"""부록 예제 — 부록 A. 주 실습 데이터셋 명세 — battery_process_data.csv

교재 출처 : manuscript/70_appendix.md
실행 방법 : example 폴더에서  python "부록/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --------------------------------------------------------------------
# [A.4 로드 예시]
# --------------------------------------------------------------------
import pandas as pd
df = pd.read_csv("data/battery_process_data.csv")
print(df.shape)
print(df["불량_여부"].sum())
print(round(df["전극_면저항_mOhmcm2"].mean(), 1))
print(round(df["프레스_Gap"].corr(df["전극_면저항_mOhmcm2"]), 4))

# 교재 실행 결과 ------------------------------------------------------
#   (1000, 10)
#   19
#   1514.0
#   0.8352

