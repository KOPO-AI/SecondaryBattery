# -*- coding: utf-8 -*-
"""체크포인트 ① — 데이터 단계 확인.  실행: python check_data.py  (같은 폴더에 data_io.py, cleaning.py)"""
import data_io, cleaning

raw = data_io.load_csv("../data/battery_process_data.csv")
print(data_io.describe_loaded(raw))
df = cleaning.clean_data(raw)
print(cleaning.cleaning_report(raw, df))
print(cleaning.eda_summary(df))
print("\n[기대값] 행 1000 / 결측 5→0 / 9999 5건 / 건조로 범위 위반 5건 / 상관 1위 프레스_Gap -0.870")
