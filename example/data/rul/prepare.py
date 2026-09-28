# -*- coding: utf-8 -*-
"""
prepare.py — Battery RUL (HNEI 파생본) 교재 실습용 가공본 생성 스크립트

원본 : https://www.kaggle.com/api/v1/datasets/download/ignaciovinuales/battery-remaining-useful-life-rul
       (archive.zip -> Battery_RUL.csv, 15,064 rows x 9 cols)
원출처: Hawaii Natural Energy Institute (HNEI), 14 x NMC-LCO 18650 (2.8 Ah 공칭),
        25 degC, CC-CV C/2 충전 / 1.5C 방전, 1000+ 사이클
        Battery Archive 파일명 HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_<letter>_timeseries.csv

실행:
    set PYTHONUTF8=1
    python prepare.py --src <원본 Battery_RUL.csv 경로> --out <출력 폴더>

생성물:
    battery_rul_original.csv  원본 그대로 복사본
    battery_rul_cells.csv     cell_id / cell_code / 물리검증 플래그 추가 (수업 메인 파일)
    battery_rul_clean.csv     물리적 불가능 행 제거 + snake_case 컬럼 (모델링용)
    cell_summary.csv          셀 14개 요약
    data_dictionary.csv       컬럼 사전
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. 상수
# ---------------------------------------------------------------------------
# 원저자 join_dataframes.py 의 concat 순서 (14개 HNEI 원본 파일의 letter)
#   files = ['a','b','c','d','e','f','g','j','l','n','o','p','s','t']
# CSV 행 순서가 이 concat 순서를 그대로 보존하므로 블록 i -> letter i 로 매핑한다.
# (CSV 자체에는 셀 식별자가 없으므로 이 매핑은 '추정'임 — data_dictionary.csv 에 명시)
CELL_LETTERS = ["a", "b", "c", "d", "e", "f", "g", "j", "l", "n", "o", "p", "s", "t"]

C_CYC = "Cycle_Index"
C_DIS = "Discharge Time (s)"
C_DEC = "Decrement 3.6-3.4V (s)"
C_MXV = "Max. Voltage Dischar. (V)"
C_MNV = "Min. Voltage Charg. (V)"
C_415 = "Time at 4.15V (s)"
C_TCC = "Time constant current (s)"
C_CHG = "Charging time (s)"
C_RUL = "RUL"

SNAKE = {
    C_CYC: "cycle_index",
    C_DIS: "discharge_time_s",
    C_DEC: "decrement_36_34v_s",
    C_MXV: "max_volt_discharge_v",
    C_MNV: "min_volt_charge_v",
    C_415: "time_at_415v_s",
    C_TCC: "time_cc_s",
    C_CHG: "charging_time_s",
    C_RUL: "rul",
}


# ---------------------------------------------------------------------------
# 1. cell_id 복원 : Cycle_Index 가 되돌아가는(감소하는) 지점 = 새 셀의 시작
# ---------------------------------------------------------------------------
def restore_cell_id(df: pd.DataFrame) -> pd.Series:
    cyc = df[C_CYC].to_numpy()
    starts = np.where(np.diff(cyc) < 0)[0] + 1          # 새 블록 첫 행 위치
    bounds = [0] + starts.tolist() + [len(df)]
    cell_id = np.zeros(len(df), dtype=int)
    for i in range(len(bounds) - 1):
        cell_id[bounds[i]:bounds[i + 1]] = i + 1
    return pd.Series(cell_id, index=df.index, name="cell_id")


# ---------------------------------------------------------------------------
# 2. 물리적 타당성 검사
# ---------------------------------------------------------------------------
def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    out = df.copy()
    # HARD : 물리적으로 불가능 (측정/전처리 오류가 확실)
    out["flag_neg_time"] = (out[C_DEC] < 0) | (out[C_415] < 0)
    out["flag_415_gt_charge"] = out[C_415] > out[C_CHG] + eps
    out["flag_dec_gt_discharge"] = out[C_DEC].abs() > out[C_DIS] + eps
    out["flag_cc_gt_charge"] = out[C_TCC] > out[C_CHG] + eps
    out["is_valid"] = ~(
        out["flag_neg_time"]
        | out["flag_415_gt_charge"]
        | out["flag_dec_gt_discharge"]
        | out["flag_cc_gt_charge"]
    )
    # SOFT : 규격 위반 의심 (셀 상한 4.2 V 기준) — 제거하지 않고 표시만
    out["flag_volt_over_spec"] = out[C_MXV] > 4.2
    out["flag_minchg_gt_maxdis"] = out[C_MNV] > out[C_MXV]
    return out


# ---------------------------------------------------------------------------
def main(src: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(src)
    print(f"[load] {src}  shape={df.shape}")

    # (1) 원본 사본
    dst_raw = os.path.join(out_dir, "battery_rul_original.csv")
    shutil.copyfile(src, dst_raw)

    # (2) cell_id 복원
    df.insert(0, "cell_id", restore_cell_id(df))
    n_cells = df["cell_id"].nunique()
    print(f"[cell] restored cells = {n_cells}")
    assert n_cells == len(CELL_LETTERS), f"기대 14개, 실제 {n_cells}개"
    df.insert(1, "cell_code", df["cell_id"].map(lambda i: CELL_LETTERS[i - 1]))

    # (3) 누수 검증 : RUL == (셀별 최대 Cycle_Index) - Cycle_Index 인가?
    leak_exact = (
        df.groupby("cell_id")
        .apply(lambda g: bool((g[C_RUL] == g[C_CYC].max() - g[C_CYC]).all()),
               include_groups=False)
    )
    print(f"[leak] RUL == maxCycle - Cycle_Index 성립 셀 수: "
          f"{int(leak_exact.sum())}/{n_cells}")

    # (4) 플래그
    df = add_flags(df)
    n_hard = int((~df["is_valid"]).sum())
    print(f"[flag] 물리 불가능(hard) 행 = {n_hard}  "
          f"(neg={int(df.flag_neg_time.sum())}, "
          f"415>chg={int(df.flag_415_gt_charge.sum())}, "
          f"dec>dis={int(df.flag_dec_gt_discharge.sum())}, "
          f"cc>chg={int(df.flag_cc_gt_charge.sum())})")
    print(f"[flag] 규격의심(soft): >4.2V={int(df.flag_volt_over_spec.sum())}, "
          f"minChg>maxDis={int(df.flag_minchg_gt_maxdis.sum())}")

    df.to_csv(os.path.join(out_dir, "battery_rul_cells.csv"), index=False)

    # (5) 모델링용 clean
    clean = df.loc[df["is_valid"], ["cell_id"] + list(SNAKE.keys())].rename(columns=SNAKE)
    clean = clean.reset_index(drop=True)
    clean.to_csv(os.path.join(out_dir, "battery_rul_clean.csv"), index=False)
    print(f"[clean] shape={clean.shape}")

    # (6) 셀 요약
    g = df.groupby(["cell_id", "cell_code"])
    summary = pd.DataFrame({
        "n_rows": g.size(),
        "cycle_min": g[C_CYC].min().astype(int),
        "cycle_max": g[C_CYC].max().astype(int),
        "n_missing_cycles": (g[C_CYC].max() - g[C_CYC].min() + 1 - g.size()).astype(int),
        "rul_first": g[C_RUL].first().astype(int),
        "rul_last": g[C_RUL].last().astype(int),
        "n_invalid_rows": g["is_valid"].apply(lambda s: int((~s).sum())),
        "max_volt_dis_mean": g[C_MXV].mean().round(4),
        "min_volt_chg_mean": g[C_MNV].mean().round(4),
        "discharge_time_median_s": g[C_DIS].median().round(2),
    }).reset_index()
    summary.to_csv(os.path.join(out_dir, "cell_summary.csv"), index=False)

    # (7) 데이터 사전
    dd = [
        ("cell_id", "int", "-", "복원된 셀 번호 1~14. Cycle_Index 가 감소(리셋)하는 지점을 셀 경계로 판단해 부여. 원본 CSV에는 없던 컬럼."),
        ("cell_code", "str", "-", "원 HNEI 파일 letter(a,b,c,d,e,f,g,j,l,n,o,p,s,t) 추정 매핑. 원저자 join_dataframes.py 의 concat 순서 기반이며 CSV만으로는 검증 불가."),
        ("Cycle_Index", "float", "회", "충방전 사이클 번호. RUL 과 결정론적 관계(누수) — 모델 입력으로 쓰면 안 됨."),
        ("Discharge Time (s)", "float", "s", "해당 사이클 방전 구간 소요 시간."),
        ("Decrement 3.6-3.4V (s)", "float", "s", "방전 중 전압이 3.6V에서 3.4V로 내려가는 데 걸린 시간. 음수 24건 존재."),
        ("Max. Voltage Dischar. (V)", "float", "V", "방전 구간 최대 전압. RUL 과 상관 +0.783 으로 가장 유용한 물리 특징."),
        ("Min. Voltage Charg. (V)", "float", "V", "충전 구간 최소 전압. RUL 과 상관 -0.760."),
        ("Time at 4.15V (s)", "float", "s", "충전 중 4.15V 도달 시각. 음수 9건 존재."),
        ("Time constant current (s)", "float", "s", "CC(정전류) 충전 구간 지속 시간."),
        ("Charging time (s)", "float", "s", "충전 전체 소요 시간(CC+CV)."),
        ("RUL", "int", "회", "타깃. 원저자 정의 RUL = (해당 셀 마지막 Cycle_Index) - (현재 Cycle_Index)."),
        ("flag_neg_time", "bool", "-", "시간 컬럼에 음수가 있는 행."),
        ("flag_415_gt_charge", "bool", "-", "Time at 4.15V > Charging time (충전 총시간보다 중간 시각이 큼)."),
        ("flag_dec_gt_discharge", "bool", "-", "|Decrement 3.6-3.4V| > Discharge Time (부분 구간이 전체보다 김)."),
        ("flag_cc_gt_charge", "bool", "-", "Time constant current > Charging time."),
        ("is_valid", "bool", "-", "위 4개 hard 플래그가 모두 False 인 행. battery_rul_clean.csv 는 True 행만 포함."),
        ("flag_volt_over_spec", "bool", "-", "Max. Voltage Dischar. > 4.2V (셀 충전 상한 초과 의심). 제거하지 않음."),
        ("flag_minchg_gt_maxdis", "bool", "-", "Min. Voltage Charg. > Max. Voltage Dischar. 인 역전 행. 제거하지 않음."),
    ]
    pd.DataFrame(dd, columns=["column", "dtype", "unit", "description"]).to_csv(
        os.path.join(out_dir, "data_dictionary.csv"), index=False, encoding="utf-8-sig"
    )
    print("[done]", out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="원본 Battery_RUL.csv 경로")
    p.add_argument("--out", required=True, help="출력 폴더")
    a = p.parse_args()
    main(a.src, a.out)
