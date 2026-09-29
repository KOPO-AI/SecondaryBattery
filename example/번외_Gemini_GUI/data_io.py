# -*- coding: utf-8 -*-
"""[프롬프트 2] 데이터 불러오기 — battery_process_data.csv 를 읽고 열 명세를 검증한다."""
import pandas as pd

EXPECTED_COLUMNS = [
    "Lot_ID", "믹싱_RPM", "믹싱_온도", "코팅_토출압력", "건조로_1구간_온도",
    "프레스_압력", "프레스_Gap", "최종_용량_mAh", "전극_면저항_mOhmcm2", "불량_여부",
]


def load_csv(path):
    """CSV 를 DataFrame 으로 읽는다. 열이 명세와 다르면 ValueError 를 낸다."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("필수 열이 없습니다: " + ", ".join(missing))
    return df


def describe_loaded(df):
    """불러온 직후 화면(로그)에 보여 줄 한 줄 요약."""
    return "행 %d개, 열 %d개, 결측 %d칸, 불량 %d건" % (
        df.shape[0], df.shape[1], int(df.isna().sum().sum()), int(df["불량_여부"].sum()))
