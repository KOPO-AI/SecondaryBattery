# -*- coding: utf-8 -*-
"""1장 — 장 전체 예제 실행본

교재 1장의 파이썬 코드를 등장 순서대로 이어 붙였다.
절별 파일은 앞 절에서 만든 변수를 이어 쓰는 경우가 있어 단독 실행이 안 될 수 있는데,
이 파일은 처음부터 끝까지 순서대로 실행되므로 그대로 돌려 볼 수 있다.

실행 : example 폴더에서  python "1장/_장전체_실행본.py"
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""
import matplotlib
matplotlib.use("Agg")   # 창을 띄우지 않고 그림을 파일로만 처리


# ====================================================================
# 실습 데이터셋 소개: battery_process_data.csv `필수` / 워밍업 실습 — 데이터를 직접 열어 보기 (필수, 실습 35분)
# ====================================================================
import sys
import pandas as pd

print("파이썬 버전:", sys.version.split()[0])
print("pandas 버전:", pd.__version__)
print("실습 준비 완료!")

df = pd.read_csv("data/battery_process_data.csv", encoding="utf-8-sig")

print("행 개수:", df.shape[0])
print("열 개수:", df.shape[1])
print(df.columns.tolist())

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

print(df.head())

df.info()

cols = ["믹싱_온도", "건조로_1구간_온도", "프레스_Gap", "최종_용량_mAh", "전극_면저항_mOhmcm2"]

print(df[cols].describe().round(1))

bad = df[df["건조로_1구간_온도"] == 9999]
print("9999 건수:", len(bad))
print(bad[["Lot_ID", "건조로_1구간_온도"]])


# ====================================================================
# 실습 데이터셋 소개: battery_process_data.csv `필수` / 실제 파일럿 라인 데이터는 어떻게 생겼는가 `선택 학습(자율 복습)` (강독 4분 · 자율 실습 20분)
# ====================================================================
import pandas as pd

cond = pd.read_csv("data/wmg/wmg_conditions_18.csv")

print("행 개수:", cond.shape[0])
print("열 개수:", cond.shape[1])

design = ["group", "electrode_id", "coat_weight_level",
          "roll_temp_c", "density_level", "roll_gap_um", "n_passes"]
print(cond[design].head(10).to_string(index=False))

print(pd.crosstab(cond["coat_weight_level"],
                  [cond["roll_temp_c"], cond["density_level"]]))

print(pd.crosstab(cond["calendering_date"], cond["roll_temp_c"]))

print(cond.groupby(["coat_weight_level", "density_level"])["roll_gap_um"]
      .agg(["min", "max"]).round(1))
print()
print(cond["n_passes"].value_counts().sort_index())

cells = pd.read_csv("data/wmg/wmg_cells_54.csv")

print("행 개수:", cells.shape[0])
print("열 개수:", cells.shape[1])

y = ["grav_dis_c20_mahg", "grav_dis_5c_mahg",
     "grav_dis_10c_mahg", "retention_50cyc_pct"]
print(cells[y].describe().round(4))

print(cells[y].isna().sum())

fail = cells[cells["grav_dis_10c_mahg"] < 1]
print("10C 용량이 1 mAh/g 미만인 셀:", len(fail))
print(fail[["cell_id", "coat_weight_level", "density_level",
            "grav_dis_10c_mahg"]].to_string(index=False))

print(cells.loc[cells["grav_dis_c20_mahg"] > 200,
                ["cell_id", "group", "grav_dis_c20_mahg",
                 "asi_dis_soc50_ohmcm2"]].to_string(index=False))


# ====================================================================
# 실습 데이터셋 소개: battery_process_data.csv `필수` / 제조공정 데이터 vs 사용단계 데이터 `선택 학습(자율 복습)` (강독 3분 · 자율 실습 12분)
# ====================================================================
import pandas as pd

files = {
    "WMG 캘린더링 DoE": "data/wmg/wmg_cells_54.csv",
    "CoatingVision 코팅결함": "data/coatingvision/coating_labels_tidy.csv",
    "Battery RUL 셀수명": "data/rul/battery_rul_clean.csv",
    "EIS 임피던스": "data/eis/eis_wide.csv",
}

for name, path in files.items():
    df = pd.read_csv(path)
    print(name, "→", df.shape)

groups = {
    "WMG 캘린더링 DoE": ("data/wmg/wmg_cells_54.csv", "group"),
    "CoatingVision 코팅결함": ("data/coatingvision/coating_labels_tidy.csv", "video_id"),
    "Battery RUL 셀수명": ("data/rul/battery_rul_clean.csv", "cell_id"),
    "EIS 임피던스": ("data/eis/eis_wide.csv", "battery_id"),
}

for name, (path, key) in groups.items():
    df = pd.read_csv(path)
    print(name, "→ 행", len(df), "/", key, "고유값", df[key].nunique())

