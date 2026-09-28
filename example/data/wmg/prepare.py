# -*- coding: utf-8 -*-
"""
WMG NMC622 캘린더링 DoE 데이터 -> 교재 실습용 tidy CSV 생성 스크립트

원본 : Faraji-Niri, M. et al. "Characteristics of Electrodes and Lithium-ion Cells
        at Pilot-Plant Manufacturing Scale", Mendeley Data V1, DOI 10.17632/wwhm2frfmy.1
        (WMG, University of Warwick / Faraday Institution).  License: CC0 1.0
입력 : ./raw/ 안의 엑셀 3개 (원본 아카이브 '1- Tables' 폴더 그대로)
출력 : 같은 폴더의 CSV 5개 + data_dictionary.csv

실행 : python prepare.py      (pandas, openpyxl 필요)
"""
import os
import sys
import csv

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

F_CAL = os.path.join(RAW, "Intermediate measurements during calendering.xlsx")
F_HALF = os.path.join(RAW, "Half-cell (Cathode) Electrochemical Performance.xlsx")
F_CYC = os.path.join(RAW, "Half-cell Electrochemical Performance Cycling Performance.xlsx")


def num(v):
    """엑셀 셀 값을 float 또는 None(결측)으로. 빈칸/문자열은 None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def txt(v):
    if v is None:
        return ""
    return str(v).strip()


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {os.path.basename(path):<28} rows={len(rows):>6}  cols={len(header):>3}  "
          f"bytes={os.path.getsize(path):,}")


# ---------------------------------------------------------------- 1. 조건 18개
# 'Cathode' 시트: 1~2행 병합 2단 헤더, 3~20행 = 평균값 블록, 22~23행 헤더, 24~41행 = 표준편차 블록
CAL_COLS = [
    (1,  "no"),
    (2,  "electrode_id"),
    (3,  "target_coat_weight_gsm"),
    (4,  "roll_temp_c"),
    (5,  "target_density_g_cm3"),
    (6,  "target_porosity_pct"),
    (7,  "calendering_date"),
    (8,  "precal_thickness_sheet_um"),
    (9,  "precal_thickness_discs_um"),
    (10, "precal_coat_weight_gsm"),
    (11, "precal_density_g_cm3"),
    (12, "precal_porosity_pct"),
    (13, "precal_tensile_strength_kpa"),
    (14, "semeds_performed"),
    (15, "precal_grad_carbon"),
    (16, "precal_grad_fluorine"),
    (17, "precal_z11_carbon"),
    (18, "precal_z11_fluorine"),
    (19, "precal_moran_carbon"),
    (20, "precal_moran_fluorine"),
    (21, "precal_cracks"),
    (22, "roll_gap_um"),
    (23, "n_passes"),
    (24, "cal_thickness_sheet_um"),
    (25, "cal_thickness_discs_um"),
    (26, "cal_coat_weight_gsm"),
    (27, "cal_density_g_cm3"),
    (28, "cal_porosity_pct"),
    (29, "cal_tensile_strength_kpa"),
    (30, "cal_grad_carbon"),
    (31, "cal_grad_fluorine"),
    (32, "cal_z11_carbon"),
    (33, "cal_z11_fluorine"),
    (34, "cal_moran_carbon"),
    (35, "cal_moran_fluorine"),
    (36, "cal_cracks"),
]
# 표준편차 블록에서 실제로 값이 채워진 열만 sd_ 접두어로 내보낸다.
SD_COLS = [3, 8, 9, 10, 11, 12, 13, 24, 25, 26, 27, 28, 29]
TEXT_COLS = {"electrode_id", "semeds_performed", "precal_cracks", "cal_cracks", "calendering_date"}


def build_conditions():
    wb = openpyxl.load_workbook(F_CAL, data_only=True)
    ws = wb["Cathode"]

    mean = {}
    for r in range(3, 21):
        rec = {}
        for ci, name in CAL_COLS:
            v = ws.cell(r, ci).value
            if name == "calendering_date":
                rec[name] = v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else txt(v)
            elif name in TEXT_COLS:
                rec[name] = txt(v)
            else:
                rec[name] = num(v)
        mean[int(rec["no"])] = rec

    for r in range(24, 42):
        no = int(ws.cell(r, 1).value)
        for ci in SD_COLS:
            name = dict(CAL_COLS)[ci]
            mean[no]["sd_" + name] = num(ws.cell(r, ci).value)

    # 파생 설계인자: 코팅중량 수준(L/H), 목표밀도 수준(P/M/D)
    for no, rec in mean.items():
        parts = rec["electrode_id"].split("_")   # NEX_CAT_240_L_85_P
        rec["coat_weight_level"] = parts[3]
        rec["density_level"] = parts[5]
        rec["group"] = no

    header = (["group", "electrode_id", "coat_weight_level", "density_level"]
              + [n for _, n in CAL_COLS if n not in ("no", "electrode_id")]
              + ["sd_" + dict(CAL_COLS)[c] for c in SD_COLS])
    rows = [[mean[no].get(h) for h in header] for no in range(1, 19)]
    return header, rows, mean


# ------------------------------------------------- 2. Group 시트 (전치 레이아웃)
def sheet_index(ws):
    """(섹션라벨, 행라벨, 등장순번) -> 행번호.  A열은 병합되어 있으므로 forward-fill."""
    idx = {}
    sec = None
    for r in range(1, ws.max_row + 1):
        av = ws.cell(r, 1).value
        if av not in (None, ""):
            sec = str(av).strip()
        bv = ws.cell(r, 2).value
        if bv in (None, ""):
            continue
        lab = str(bv).strip()
        n = 1
        while (sec, lab, n) in idx:
            n += 1
        idx[(sec, lab, n)] = r
    return idx


CAP_D = "Capacity Discharge (mAh)"
CAP_C = "Capacity Charge (mAh)"
GRV_D = "Gravimetric Discharge Capacity (mAh/g)"
GRV_C = "Gravimetric Charge Capacity (mAh/g)"
VOL_D = "Volumetric Discharge Capacity (mAh/cm3)"
VOL_C = "Volumetric Charge Capacity (mAh/cm3)"
ASI_C = "ASI Charge (Ohm cm2)"
ASI_D = "ASI Discharge (Ohm cm2)"
DET = "Electrode and cell details"

# rate test 12 스텝. (스텝번호, 방전 C-rate, 충전 C-rate, 방전행라벨, 충전행라벨+등장순번)
STEPS = [
    (1,  "C/20", "C/20", ("At C/20", 1),   ("At C/20", 1)),
    (2,  "C/5",  "C/5",  ("At C/5- 1", 1), ("At C/5- 1", 1)),
    (3,  "C/5",  "C/5",  ("At C/5- 2", 1), ("At C/5- 2", 1)),
    (4,  "C/5",  "C/5",  ("At C/5- 3", 1), ("At C/5- 3", 1)),
    (5,  "C/5",  "C/5",  ("At C/5- 4", 1), ("At C/5- 4", 1)),
    (6,  "C/5",  "C/5",  ("At C/5- 5", 1), ("At C/5- 5", 1)),
    (7,  "C/5",  "C/5",  ("At C/5- 6", 1), ("At C/5- 6", 1)),
    (8,  "C/2",  "C/5",  ("At C/2", 1),    ("At C/5", 1)),
    (9,  "1C",   "C/5",  ("At C", 1),      ("At C/5", 2)),
    (10, "2C",   "C/5",  ("At 2C", 1),     ("At C/5", 3)),
    (11, "5C",   "C/5",  ("At 5C", 1),     ("At C/5", 4)),
    (12, "10C",  "C/5",  ("At 10C", 1),    ("At C/5", 5)),
]
# ASI 스윕: 라벨은 0.9 ~ 0.1 (SOC/DOD 분율) + 원본에 설명 없는 0.5, 0.2 재등장 2개
ASI_KEYS = [("0.9", 1), ("0.8", 1), ("0.7", 1), ("0.6", 1), ("0.5", 1),
            ("0.4", 1), ("0.3", 1), ("0.2", 1), ("0.1", 1),
            ("0.5", 2), ("0.2", 2)]


def gv(ws, idx, sec, lab, occ, col):
    """섹션/라벨/순번으로 지정된 행의 col열 값(숫자)."""
    r = idx.get((sec, lab, occ))
    if r is None:
        return None
    return num(ws.cell(r, col).value)


def gt(ws, idx, sec, lab, occ, col):
    r = idx.get((sec, lab, occ))
    if r is None:
        return ""
    return txt(ws.cell(r, col).value)


def build_cells(cond):
    wb = openpyxl.load_workbook(F_HALF, data_only=True)
    cells, rate_rows, asi_rows = [], [], []

    for g in range(1, 19):
        ws = wb[f"Group{g}"]
        idx = sheet_index(ws)
        # 1행: A,B 라벨 뒤 C..E = 반복 셀, F = Mean, G = Standard deviation
        ids = [txt(ws.cell(1, c).value) for c in (3, 4, 5)]
        assert txt(ws.cell(1, 6).value).lower().startswith("mean"), f"Group{g} 레이아웃 불일치"
        c0 = cond[g]

        for rep, (col, cid) in enumerate(zip((3, 4, 5), ids), start=1):
            rec = {
                "group": g,
                "electrode_id": c0["electrode_id"],
                "cell_id": cid,
                "replicate": rep,
                "coating_id": gt(ws, idx, DET, "Cathode coating ID", 1, col),
                # --- 설계인자 X (조건 수준) ---
                "coat_weight_level": c0["coat_weight_level"],
                "target_coat_weight_gsm": c0["target_coat_weight_gsm"],
                "roll_temp_c": c0["roll_temp_c"],
                "density_level": c0["density_level"],
                "target_density_g_cm3": c0["target_density_g_cm3"],
                "target_porosity_pct": c0["target_porosity_pct"],
                "roll_gap_um": c0["roll_gap_um"],
                "n_passes": c0["n_passes"],
                # --- 실측 전극 물성 (셀 단위) ---
                "coating_thickness_um": gv(ws, idx, DET, "Coating thickness (um)", 1, col),
                "coating_weight_gsm": gv(ws, idx, DET, "Coating weight (g/m2)", 1, col),
                "density_g_cm3": gv(ws, idx, DET, "Density (g/cm3)", 1, col),
                "porosity_pct": gv(ws, idx, DET, "Porosity (%)", 1, col),
                "area_cm2": gv(ws, idx, DET, "Area (cm2)", 1, col),
                "active_mass_g": gv(ws, idx, DET, "Active material mass (g)", 1, col),
                "coating_mass_g": gv(ws, idx, DET, "Coating mass (g)", 1, col),
                "total_thickness_um": gv(ws, idx, DET, "Total thickness (um)", 1, col),
                "expected_capacity_mah": gv(ws, idx, DET, "Expected capacity (mAh)", 1, col),
                # --- 조건 수준 전극 물성 (캘린더링 표에서 결합) ---
                "cal_tensile_strength_kpa": c0["cal_tensile_strength_kpa"],
                "precal_tensile_strength_kpa": c0["precal_tensile_strength_kpa"],
                "cal_grad_carbon": c0["cal_grad_carbon"],
                "cal_grad_fluorine": c0["cal_grad_fluorine"],
                "cal_moran_carbon": c0["cal_moran_carbon"],
                "cal_moran_fluorine": c0["cal_moran_fluorine"],
                # --- 시험 조건 (잡음인자) ---
                "test_equipment": gt(ws, idx, DET, "Testing equipment", 1, col),
                "channel": gt(ws, idx, DET, "Channel", 1, col),
                "v_after_assembly_v": gv(ws, idx, DET, "V after assembly (V)", 1, col),
            }

            # --- y: 방전용량 / 그램당 방전용량 ---
            rec["cap_dis_c20_mah"] = gv(ws, idx, CAP_D, "At C/20", 1, col)
            rec["cap_dis_c5_mah"] = gv(ws, idx, CAP_D, "At C/5- 6", 1, col)
            rec["cap_dis_c2_mah"] = gv(ws, idx, CAP_D, "At C/2", 1, col)
            rec["cap_dis_1c_mah"] = gv(ws, idx, CAP_D, "At C", 1, col)
            rec["cap_dis_2c_mah"] = gv(ws, idx, CAP_D, "At 2C", 1, col)
            rec["cap_dis_5c_mah"] = gv(ws, idx, CAP_D, "At 5C", 1, col)
            rec["cap_dis_10c_mah"] = gv(ws, idx, CAP_D, "At 10C", 1, col)

            rec["grav_dis_c20_mahg"] = gv(ws, idx, GRV_D, "At C/20", 1, col)
            rec["grav_dis_c5_mahg"] = gv(ws, idx, GRV_D, "At C/5- 6", 1, col)
            rec["grav_dis_c2_mahg"] = gv(ws, idx, GRV_D, "At C/2", 1, col)
            rec["grav_dis_1c_mahg"] = gv(ws, idx, GRV_D, "At C", 1, col)
            rec["grav_dis_2c_mahg"] = gv(ws, idx, GRV_D, "At 2C", 1, col)
            rec["grav_dis_5c_mahg"] = gv(ws, idx, GRV_D, "At 5C", 1, col)
            rec["grav_dis_10c_mahg"] = gv(ws, idx, GRV_D, "At 10C", 1, col)

            rec["rate_ratio_5c_over_0p2c"] = gv(ws, idx, GRV_D, "Rate charge 5C:0.2C", 1, col)
            rec["first_cycle_loss_pct"] = gv(ws, idx, GRV_D, "First cycle loss (%)", 1, col)

            # --- y: ASI 대표값 ---
            for lab, nm in (("0.9", "soc90"), ("0.5", "soc50"), ("0.2", "soc20")):
                rec[f"asi_chg_{nm}_ohmcm2"] = gv(ws, idx, ASI_C, lab, 1, col)
                rec[f"asi_dis_{nm}_ohmcm2"] = gv(ws, idx, ASI_D, lab, 1, col)

            # --- y: 사이클 요약 ---
            rec["cyc_cap_c10_cycle0_mah"] = gv(ws, idx, "Cycling", "Capacity at C/10 (cycle 0)", 1, col)
            rec["cyc_cap_c2_cycle1_mah"] = gv(ws, idx, "Cycling", "Capacity at C/2 (cycle 1)", 1, col)
            rec["cyc_cap_c2_cycle50_mah"] = gv(ws, idx, "Cycling", "Capacity at C/2 (cycle 50)", 1, col)
            rec["cyc_cap_c10_cycle51_mah"] = gv(ws, idx, "Cycling", "Capacity at C/10 (cycle 51)", 1, col)

            cells.append(rec)

            # --- 부속 long 테이블: rate test 12스텝 ---
            for step, dr, cr, (dl, do), (cl, co) in STEPS:
                rate_rows.append([
                    g, c0["electrode_id"], cid, rep, step, dr, cr,
                    gv(ws, idx, CAP_D, dl, do, col),
                    gv(ws, idx, CAP_C, cl, co, col),
                    gv(ws, idx, GRV_D, dl, do, col),
                    gv(ws, idx, GRV_C, cl, co, col),
                    gv(ws, idx, VOL_D, dl, do, col),
                    gv(ws, idx, VOL_C, cl, co, col),
                ])

            # --- 부속 long 테이블: ASI 스윕 ---
            for lab, occ in ASI_KEYS:
                asi_rows.append([
                    g, c0["electrode_id"], cid, rep, float(lab), occ,
                    gv(ws, idx, ASI_C, lab, occ, col),
                    gv(ws, idx, ASI_D, lab, occ, col),
                ])
    return cells, rate_rows, asi_rows


# ------------------------------------------------------------- 3. 사이클 long
def build_cycling():
    wb = openpyxl.load_workbook(F_CYC, data_only=True)
    ws = wb["Sheet1"]
    ncol = ws.max_column
    ids = [txt(ws.cell(1, c).value) for c in range(3, ncol + 1)]
    grp = [num(ws.cell(2, c).value) for c in range(3, ncol + 1)]

    # 블록 시작행: 방전용량 4~55(cycle 0~51), 충전용량 59~110(0~51),
    #              용량유지율 114~163(1~50), 쿨롱효율 168~217(1~50)
    dis = {int(ws.cell(r, 1).value): r for r in range(4, 56)}
    chg = {int(ws.cell(r, 1).value): r for r in range(59, 111)}
    ret = {int(ws.cell(r, 2).value): r for r in range(114, 164)}
    cef = {int(ws.cell(r, 2).value): r for r in range(168, 218)}
    rate = {c: txt(ws.cell(r, 2).value) for c, r in dis.items()}

    rows = []
    for j, (cid, g) in enumerate(zip(ids, grp)):
        col = 3 + j
        for cyc in range(0, 52):
            rows.append([
                int(g), cid, cyc, rate[cyc],
                num(ws.cell(dis[cyc], col).value),
                num(ws.cell(chg[cyc], col).value),
                num(ws.cell(ret[cyc], col).value) if cyc in ret else None,
                num(ws.cell(cef[cyc], col).value) if cyc in cef else None,
            ])
    return ids, rows


# ------------------------------------------------------------ 4. 컬럼 사전
# (컬럼명, 역할, 단위, 한글설명, 원본 라벨/출처)
D_CELLS = [
    ("group", "id", "-", "캘린더링 조건 번호 1~18 (Group 시트 번호 = 조건표 No)", "Group sheet no."),
    ("electrode_id", "id", "-", "전극 조건 ID. NEX_CAT_240_{L|H}_{롤온도}_{P|M|D}", "Electrode ID"),
    ("cell_id", "id", "-", "하프셀 고유번호 (DDxxx). 총 54개", "Cathode cell ID"),
    ("replicate", "id", "-", "같은 조건 안의 반복 셀 번호 1~3", "Group 시트 C/D/E열 순서"),
    ("coating_id", "id", "-", "코팅 시트에서 잘라낸 디스크 위치 ID", "Cathode coating ID"),
    ("coat_weight_level", "X_design", "-", "코팅중량 수준. L=저중량, H=고중량", "Electrode ID에서 파생"),
    ("target_coat_weight_gsm", "X_design", "g/m2", "목표 코팅중량. L=122.48, H=182.73", "Target coating weight (GSM)"),
    ("roll_temp_c", "X_design", "degC", "캘린더링 롤 온도. 85 / 120 / 145", "Roll temperature (oC)"),
    ("density_level", "X_design", "-", "목표밀도 수준. P=다공(porous), M=중간, D=치밀(dense)", "Electrode ID에서 파생"),
    ("target_density_g_cm3", "X_design", "g/cm3", "목표 전극 밀도. 2.7 / 2.95 / 3.2", "Target density (g/cm3)"),
    ("target_porosity_pct", "X_design", "%", "목표밀도로부터 계산한 목표 공극률. 39.44 / 33.83 / 28.22", "Calculated target porosity (%)"),
    ("roll_gap_um", "X_design", "um", "설정 롤 갭. 약 500um 심(shim) 2장 포함값. 390~495", "Roll gap (um)- includes 2 shims of ~500 um"),
    ("n_passes", "X_design", "회", "목표 밀도 도달까지 실제 통과 횟수 1~3 (결과이자 공정 조건)", "Number of passes"),
    ("coating_thickness_um", "X_measured", "um", "해당 셀 디스크의 코팅 두께(집전체 제외)", "Coating thickness (um)"),
    ("coating_weight_gsm", "X_measured", "g/m2", "해당 셀 디스크의 실측 코팅중량", "Coating weight (g/m2)"),
    ("density_g_cm3", "X_measured", "g/cm3", "해당 셀 디스크의 실측 전극 밀도", "Density (g/cm3)"),
    ("porosity_pct", "X_measured", "%", "해당 셀 디스크의 실측 공극률. 진밀도 4.458 g/cm3 기준", "Porosity (%)"),
    ("area_cm2", "X_measured", "cm2", "전극 디스크 면적. 전 셀 1.7195 고정", "Area (cm2)"),
    ("active_mass_g", "X_measured", "g", "활물질(NMC622) 질량. 그램당 용량 환산 기준", "Active material mass (g)"),
    ("coating_mass_g", "X_measured", "g", "코팅층 질량(전체질량 - Al박 질량)", "Coating mass (g)"),
    ("total_thickness_um", "X_measured", "um", "코팅 + Al 집전체 총 두께", "Total thickness (um)"),
    ("expected_capacity_mah", "X_measured", "mAh", "활물질량 x 165 mAh/g 로 계산한 기대용량", "Expected capacity (mAh)"),
    ("cal_tensile_strength_kpa", "X_measured", "kPa", "캘린더링 후 전극 최대 인장강도(조건 단위 값)", "Electrode max. tensile strength (kPa), Calendered"),
    ("precal_tensile_strength_kpa", "X_measured", "kPa", "캘린더링 전 인장강도. 코팅중량 수준별 1개 값뿐(L=728.62, H=683.25)", "Electrode max. tensile strength (kPa), Before"),
    ("cal_grad_carbon", "X_measured", "%/%", "캘린더링 후 두께방향 탄소 농도 구배 (SEM/EDS)", "Gradient in carbon (% %-1)"),
    ("cal_grad_fluorine", "X_measured", "%/%", "캘린더링 후 두께방향 불소(바인더) 농도 구배 (SEM/EDS)", "Gradient in fluorine (% %-1)"),
    ("cal_moran_carbon", "X_measured", "-", "캘린더링 후 탄소 분포의 Moran's I (공간 자기상관, 0~1)", "Moran I score in carbon"),
    ("cal_moran_fluorine", "X_measured", "-", "캘린더링 후 불소 분포의 Moran's I", "Moran I score in fluorine"),
    ("test_equipment", "meta", "-", "충방전기 장비명(BCS1~BCS8). 조건과 교락 가능한 잡음인자", "Testing equipment"),
    ("channel", "meta", "-", "충방전기 채널 번호", "Channel"),
    ("v_after_assembly_v", "meta", "V", "셀 조립 직후 개회로 전압", "V after assembly (V)"),
    ("cap_dis_c20_mah", "y", "mAh", "C/20 화성(formation) 1회차 방전용량", "Capacity Discharge, At C/20"),
    ("cap_dis_c5_mah", "y", "mAh", "C/5 반복 6회차 방전용량(안정화된 기준값)", "Capacity Discharge, At C/5- 6"),
    ("cap_dis_c2_mah", "y", "mAh", "C/2 방전용량 (충전은 항상 C/5)", "Capacity Discharge, At C/2"),
    ("cap_dis_1c_mah", "y", "mAh", "1C 방전용량", "Capacity Discharge, At C"),
    ("cap_dis_2c_mah", "y", "mAh", "2C 방전용량", "Capacity Discharge, At 2C"),
    ("cap_dis_5c_mah", "y", "mAh", "5C 방전용량", "Capacity Discharge, At 5C"),
    ("cap_dis_10c_mah", "y", "mAh", "10C 방전용량. 일부 셀은 사실상 0 (레이트 시험 실패)", "Capacity Discharge, At 10C"),
    ("grav_dis_c20_mahg", "y", "mAh/g", "C/20 그램당 방전용량 = cap_dis_c20_mah / active_mass_g", "Gravimetric Discharge Capacity, At C/20"),
    ("grav_dis_c5_mahg", "y", "mAh/g", "C/5 6회차 그램당 방전용량", "Gravimetric Discharge Capacity, At C/5- 6"),
    ("grav_dis_c2_mahg", "y", "mAh/g", "C/2 그램당 방전용량", "Gravimetric Discharge Capacity, At C/2"),
    ("grav_dis_1c_mahg", "y", "mAh/g", "1C 그램당 방전용량", "Gravimetric Discharge Capacity, At C"),
    ("grav_dis_2c_mahg", "y", "mAh/g", "2C 그램당 방전용량", "Gravimetric Discharge Capacity, At 2C"),
    ("grav_dis_5c_mahg", "y", "mAh/g", "5C 그램당 방전용량", "Gravimetric Discharge Capacity, At 5C"),
    ("grav_dis_10c_mahg", "y", "mAh/g", "10C 그램당 방전용량 (레이트 특성 핵심 지표)", "Gravimetric Discharge Capacity, At 10C"),
    ("rate_ratio_5c_over_0p2c", "y", "-", "5C 용량 / 0.2C(=C/5) 용량 비. 레이트 유지율(0~1)", "Rate charge 5C:0.2C"),
    ("first_cycle_loss_pct", "y", "%", "초기 비가역 용량 손실률", "First cycle loss (%)"),
    ("asi_chg_soc90_ohmcm2", "y", "Ohm cm2", "충전 중 상태분율 0.9 지점의 면적비저항", "ASI Charge (Ohm cm2), 0.9"),
    ("asi_dis_soc90_ohmcm2", "y", "Ohm cm2", "방전 중 상태분율 0.9 지점의 면적비저항", "ASI Discharge (Ohm cm2), 0.9"),
    ("asi_chg_soc50_ohmcm2", "y", "Ohm cm2", "충전 중 상태분율 0.5 지점의 면적비저항", "ASI Charge (Ohm cm2), 0.5"),
    ("asi_dis_soc50_ohmcm2", "y", "Ohm cm2", "방전 중 상태분율 0.5 지점의 면적비저항", "ASI Discharge (Ohm cm2), 0.5"),
    ("asi_chg_soc20_ohmcm2", "y", "Ohm cm2", "충전 중 상태분율 0.2 지점의 면적비저항", "ASI Charge (Ohm cm2), 0.2"),
    ("asi_dis_soc20_ohmcm2", "y", "Ohm cm2", "방전 중 상태분율 0.2 지점의 면적비저항", "ASI Discharge (Ohm cm2), 0.2"),
    ("cyc_cap_c10_cycle0_mah", "y", "mAh", "수명시험 0사이클(C/10) 방전용량", "Cycling, Capacity at C/10 (cycle 0)"),
    ("cyc_cap_c2_cycle1_mah", "y", "mAh", "수명시험 1사이클(C/2) 방전용량", "Cycling, Capacity at C/2 (cycle 1)"),
    ("cyc_cap_c2_cycle50_mah", "y", "mAh", "수명시험 50사이클(C/2) 방전용량", "Cycling, Capacity at C/2 (cycle 50)"),
    ("cyc_cap_c10_cycle51_mah", "y", "mAh", "수명시험 51사이클(C/10) 방전용량", "Cycling, Capacity at C/10 (cycle 51)"),
    ("retention_50cyc_pct", "y", "%", "50사이클 용량유지율 = cycle50 / cycle1 x 100", "Cycling 파일 retention 블록 cycle 50"),
]

D_COND_EXTRA = {
    "calendering_date": ("meta", "YYYY-MM-DD", "캘린더링 실시일 (2021-10-15 / 10-18 / 10-27). 롤온도와 완전 교락", "Calendering date"),
    "semeds_performed": ("meta", "-", "SEM/EDS 분석 수행 여부. 전 조건 Y", "SEM/EDS analysis performed (Y/N)"),
    "precal_cracks": ("X_measured", "-", "캘린더링 전 입자 균열 관찰 여부. 전 조건 N", "Observed cracks in particles (Y/N), Before"),
    "cal_cracks": ("y", "-", "캘린더링 후 입자 균열 관찰 여부. 전 조건 Y", "Observed cracks in particles (Y/N), Calendered"),
}
_COND_KO = {
    "thickness_sheet_um": ("um", "코팅 시트에서 측정한 코팅 두께"),
    "thickness_discs_um": ("um", "펀칭한 디스크에서 측정한 코팅 두께"),
    "coat_weight_gsm": ("g/m2", "디스크 실측 코팅중량"),
    "density_g_cm3": ("g/cm3", "디스크 실측 전극 밀도"),
    "porosity_pct": ("%", "디스크 실측 공극률"),
    "tensile_strength_kpa": ("kPa", "전극 최대 인장강도"),
    "grad_carbon": ("%/%", "두께방향 탄소 농도 구배 (SEM/EDS)"),
    "grad_fluorine": ("%/%", "두께방향 불소(바인더) 농도 구배 (SEM/EDS)"),
    "z11_carbon": ("-", "탄소 분포의 Z1-1 통계량 (SEM/EDS)"),
    "z11_fluorine": ("-", "불소 분포의 Z1-1 통계량 (SEM/EDS)"),
    "moran_carbon": ("-", "탄소 분포의 Moran's I 공간 자기상관"),
    "moran_fluorine": ("-", "불소 분포의 Moran's I 공간 자기상관"),
}

D_RATE = [
    ("group", "id", "-", "조건 번호 1~18", "-"),
    ("electrode_id", "id", "-", "전극 조건 ID", "-"),
    ("cell_id", "id", "-", "하프셀 번호", "-"),
    ("replicate", "id", "-", "반복 셀 번호 1~3", "-"),
    ("step", "id", "-", "레이트 시험 스텝 순번 1~12", "Group 시트 행 순서"),
    ("discharge_c_rate", "X_design", "-", "해당 스텝의 방전 C-rate", "행 라벨"),
    ("charge_c_rate", "X_design", "-", "해당 스텝의 충전 C-rate (8~12스텝은 항상 C/5)", "행 라벨"),
    ("cap_dis_mah", "y", "mAh", "방전용량", "Capacity Discharge (mAh)"),
    ("cap_chg_mah", "y", "mAh", "충전용량", "Capacity Charge (mAh)"),
    ("grav_dis_mahg", "y", "mAh/g", "그램당 방전용량", "Gravimetric Discharge Capacity (mAh/g)"),
    ("grav_chg_mahg", "y", "mAh/g", "그램당 충전용량", "Gravimetric Charge Capacity (mAh/g)"),
    ("vol_dis_mahcm3", "y", "mAh/cm3", "부피당 방전용량", "Volumetric Discharge Capacity (mAh/cm3)"),
    ("vol_chg_mahcm3", "y", "mAh/cm3", "부피당 충전용량", "Volumetric Charge Capacity (mAh/cm3)"),
]

D_ASI = [
    ("group", "id", "-", "조건 번호 1~18", "-"),
    ("electrode_id", "id", "-", "전극 조건 ID", "-"),
    ("cell_id", "id", "-", "하프셀 번호", "-"),
    ("replicate", "id", "-", "반복 셀 번호 1~3", "-"),
    ("soc_fraction", "X_design", "-", "원본 행 라벨 0.9~0.1 (충전/방전 진행 상태 분율). 원본에 SOC/DOD 정의 명시 없음", "ASI 섹션 행 라벨"),
    ("occurrence", "id", "-", "같은 라벨의 등장 순번. 1=0.9~0.1 스윕, 2=원본에 설명 없는 0.5/0.2 재측정", "행 중복 구분자"),
    ("asi_chg_ohmcm2", "y", "Ohm cm2", "충전 면적비저항", "ASI Charge (Ohm cm2)"),
    ("asi_dis_ohmcm2", "y", "Ohm cm2", "방전 면적비저항", "ASI Discharge (Ohm cm2)"),
]

D_CYC = [
    ("group", "id", "-", "조건 번호 1~18", "Group 행"),
    ("cell_id", "id", "-", "하프셀 번호", "Cell ID 행"),
    ("cycle", "id", "회", "사이클 번호 0~51", "A열/B열 사이클 인덱스"),
    ("discharge_c_rate", "X_design", "-", "방전 C-rate. cycle 0과 51은 C/10, 1~50은 C/2", "C rate for discharge"),
    ("cap_dis_mah", "y", "mAh", "해당 사이클 방전용량", "Discharge capacity (mAh)"),
    ("cap_chg_mah", "y", "mAh", "해당 사이클 충전용량. 충전은 C/5 (cycle 0은 원본 결측)", "Charge capacity (mAh)"),
    ("retention_pct", "y", "%", "1사이클(C/2) 대비 용량유지율. cycle 1~50만 존재", "Capacity retention (%)"),
    ("coulombic_eff_pct", "y", "%", "쿨롱효율 = 방전용량/충전용량 x 100. cycle 1~50만 존재", "Coulombic effieciency (%)"),
]


def build_dictionary(headers):
    rows = []
    for c, role, unit, ko, src in D_CELLS:
        rows.append(["wmg_cells_54.csv", c, role, unit, ko, src])

    for c in headers["cond"]:
        if c in ("group", "electrode_id", "coat_weight_level", "density_level",
                 "target_coat_weight_gsm", "roll_temp_c", "target_density_g_cm3",
                 "target_porosity_pct", "roll_gap_um", "n_passes"):
            base = dict((x[0], x) for x in D_CELLS)[c]
            rows.append(["wmg_conditions_18.csv", c, base[1], base[2], base[3], base[4]])
        elif c in D_COND_EXTRA:
            role, unit, ko, src = D_COND_EXTRA[c]
            rows.append(["wmg_conditions_18.csv", c, role, unit, ko, src])
        else:
            sd = c.startswith("sd_")
            body = c[3:] if sd else c
            if body.startswith("precal_"):
                stage, key = "캘린더링 전", body[len("precal_"):]
            elif body.startswith("cal_"):
                stage, key = "캘린더링 후", body[len("cal_"):]
            else:
                stage, key = "", body
            if key == "target_coat_weight_gsm":
                # 표준편차 블록의 이 열은 '목표값'이 아니라 미캘린더 코팅중량 실측의 산포다.
                unit = "g/m2"
                ko = ("표준편차: 캘린더링 전 코팅중량 실측 산포 "
                      "(원본 헤더는 'Target coating weight'이지만 목표값의 편차가 아님)")
                rows.append(["wmg_conditions_18.csv", c, "meta", unit, ko,
                             "Intermediate measurements during calendering.xlsx, Cathode 시트 표준편차 블록"])
                continue
            unit, ko = _COND_KO[key]
            pre = "표준편차: " if sd else ""
            desc = " ".join(f"{pre}{stage} {ko} (조건당 반복 측정 요약)".split())
            rows.append(["wmg_conditions_18.csv", c, "X_measured" if not sd else "meta",
                         unit, desc,
                         "Intermediate measurements during calendering.xlsx, Cathode 시트"])

    for tag, defs in (("wmg_rate_long.csv", D_RATE), ("wmg_asi_long.csv", D_ASI),
                      ("wmg_cycling_long.csv", D_CYC)):
        for c, role, unit, ko, src in defs:
            rows.append([tag, c, role, unit, ko, src])

    # 모든 컬럼이 사전에 등록되었는지 검증
    covered = {}
    for f, c, *_ in rows:
        covered.setdefault(f, set()).add(c)
    for f, hs in (("wmg_cells_54.csv", headers["cells"]), ("wmg_conditions_18.csv", headers["cond"]),
                  ("wmg_rate_long.csv", headers["rate"]), ("wmg_asi_long.csv", headers["asi"]),
                  ("wmg_cycling_long.csv", headers["cyc"])):
        miss = set(hs) - covered.get(f, set())
        extra = covered.get(f, set()) - set(hs)
        assert not miss, f"{f} 사전 누락: {sorted(miss)}"
        assert not extra, f"{f} 사전에만 있는 컬럼: {sorted(extra)}"
    return rows


# --------------------------------------------------------------------- main
def main():
    print("[1/6] 캘린더링 조건표 (18조건)")
    ch, cr, cond = build_conditions()
    write_csv(os.path.join(HERE, "wmg_conditions_18.csv"), ch, cr)

    print("[2/6] 하프셀 단위 메인 테이블 (54셀)")
    cells, rate_rows, asi_rows = build_cells(cond)

    print("[3/6] 사이클 long 테이블")
    cyc_ids, cyc_rows = build_cycling()

    # 사이클 50회 후 유지율을 메인 테이블에 결합
    ret50 = {}
    for g, cid, cyc, rt, d, c, r, e in cyc_rows:
        if cyc == 50:
            ret50[cid] = r
    for rec in cells:
        rec["retention_50cyc_pct"] = ret50.get(rec["cell_id"])

    # --- 결측 처리: 용량 0 은 물리적으로 불가능 -> 결측으로 변환 ---
    zero_fixed = 0
    for rec in cells:
        for k, v in list(rec.items()):
            if isinstance(v, float) and v == 0.0 and ("cap_" in k or "grav_" in k or "asi_" in k):
                rec[k] = None
                zero_fixed += 1
    for row in rate_rows:
        for i in range(7, 13):
            if row[i] == 0.0:
                row[i] = None
                zero_fixed += 1
    print(f"       용량/ASI 열의 0 -> 결측 변환: {zero_fixed}건")

    main_header = list(cells[0].keys())
    write_csv(os.path.join(HERE, "wmg_cells_54.csv"),
              main_header, [[r.get(h) for h in main_header] for r in cells])

    print("[4/6] 부속 long 테이블")
    write_csv(os.path.join(HERE, "wmg_rate_long.csv"),
              ["group", "electrode_id", "cell_id", "replicate", "step", "discharge_c_rate",
               "charge_c_rate", "cap_dis_mah", "cap_chg_mah", "grav_dis_mahg",
               "grav_chg_mahg", "vol_dis_mahcm3", "vol_chg_mahcm3"], rate_rows)
    write_csv(os.path.join(HERE, "wmg_asi_long.csv"),
              ["group", "electrode_id", "cell_id", "replicate", "soc_fraction", "occurrence",
               "asi_chg_ohmcm2", "asi_dis_ohmcm2"], asi_rows)
    write_csv(os.path.join(HERE, "wmg_cycling_long.csv"),
              ["group", "cell_id", "cycle", "discharge_c_rate", "cap_dis_mah",
               "cap_chg_mah", "retention_pct", "coulombic_eff_pct"], cyc_rows)

    print("[5/6] 컬럼 사전")
    dict_rows = build_dictionary({
        "cells": main_header,
        "cond": ch,
        "rate": ["group", "electrode_id", "cell_id", "replicate", "step", "discharge_c_rate",
                 "charge_c_rate", "cap_dis_mah", "cap_chg_mah", "grav_dis_mahg",
                 "grav_chg_mahg", "vol_dis_mahcm3", "vol_chg_mahcm3"],
        "asi": ["group", "electrode_id", "cell_id", "replicate", "soc_fraction", "occurrence",
                "asi_chg_ohmcm2", "asi_dis_ohmcm2"],
        "cyc": ["group", "cell_id", "cycle", "discharge_c_rate", "cap_dis_mah",
                "cap_chg_mah", "retention_pct", "coulombic_eff_pct"],
    })
    write_csv(os.path.join(HERE, "data_dictionary.csv"),
              ["file", "column", "role", "unit", "description_ko", "source_label_en"], dict_rows)

    print("[6/6] 정합성 검증")
    ok = True
    main_ids = [r["cell_id"] for r in cells]
    if len(main_ids) != 54:
        print("  ! 셀 수 불일치:", len(main_ids)); ok = False
    if sorted(main_ids) != sorted(cyc_ids):
        print("  ! 메인/사이클 셀ID 불일치"); ok = False
    if len(set(main_ids)) != 54:
        print("  ! 중복 셀ID 존재"); ok = False
    print("  검증 결과:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
