# -*- coding: utf-8 -*-
"""[프롬프트 3·4] 정제 규약(교재 3.7.1) 과 EDA 요약 — 3장에서 배운 내용."""
import numpy as np
import pandas as pd

# 교재 3.7.1 절의 물리 범위 규칙 (그대로 사용)
PHYSICAL_RULES = {
    "믹싱_온도": (10, 50),
    "건조로_1구간_온도": (80, 160),
    "프레스_압력": (20, 80),
}


def clean_data(df):
    """정제 4단계: ① 9999 → NaN ② 물리 범위 밖 → NaN ③ 결측 → 열 중앙값 ④ Lot_ID 중복 제거.
    원본을 바꾸지 않고 새 DataFrame 을 돌려준다."""
    df = df.copy()
    df["건조로_1구간_온도"] = df["건조로_1구간_온도"].replace(9999, np.nan)
    for col, (lo, hi) in PHYSICAL_RULES.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df = df.drop_duplicates(subset="Lot_ID").reset_index(drop=True)
    return df


def cleaning_report(before, after):
    """정제 전후 비교 문장(로그 박스용)."""
    lines = ["[정제 결과] %d행 → %d행, 결측 %d칸 → %d칸" % (
        len(before), len(after), int(before.isna().sum().sum()), int(after.isna().sum().sum()))]
    n9999 = int((before["건조로_1구간_온도"] == 9999).sum())
    lines.append("  에러 코드 9999: %d건 → NaN → 중앙값 대체" % n9999)
    for col, (lo, hi) in PHYSICAL_RULES.items():
        n = int(((before[col] < lo) | (before[col] > hi)).sum())
        lines.append("  %s 물리 범위(%s~%s) 위반: %d건" % (col, lo, hi, n))
    return "\n".join(lines)


def eda_summary(df, target="최종_용량_mAh"):
    """3장 EDA 요약: 수치 열 describe + 타깃과의 상관 상위 3개."""
    num = df.select_dtypes("number")
    desc = num.describe().T[["mean", "std", "min", "max"]].round(2)
    corr = num.corr()[target].drop(target).sort_values(key=abs, ascending=False).head(3).round(3)
    txt = ["[기술통계]", desc.to_string(), "", "[%s 와의 상관 상위 3]" % target, corr.to_string()]
    return "\n".join(txt)
