# -*- coding: utf-8 -*-
"""
prepare.py — 리튬이온 배터리 EIS(전기화학 임피던스 분광) 데이터 교재 실습용 가공 스크립트

원자료
    Buchicchio, E.; De Angelis, A.; Santoni, F.; Carbone, P. (2022),
    "Dataset on broadband Electrochemical Impedance Spectroscopy of Lithium-Ion
     Batteries for Different Values of the State of Charge", Mendeley Data, V3.
    DOI: 10.17632/mbv3bx847g.3   /   License: CC BY 4.0

실험 설계 (원자료 실측 확인값)
    Samsung ICR18650-26J 원통형 셀 4개(ID 02, 03, 05, 06)
    셀당 EIS 측정 6회 반복 -> 총 24 measurement
    측정 1회당 SOC 10단계(10~100%, 10% 간격)
    SOC 1단계당 주파수 14점 (0.05 Hz ~ 1000 Hz)
    4 x 6 x 10 x 14 = 3,360 행

핵심 가공 포인트
    1) IMPEDANCE_VALUE 가 복소수 "문자열"  예: "(0.110973570048518-0.00547305228273139j)"
       -> 파이썬 내장 complex() 로 파싱해 실수부/허수부 분리
    2) 주파수는 ID 로만 저장 -> frequencies.csv 룩업 테이블과 merge 필요
    3) long -> wide 피벗: 스펙트럼 1개(= measurement x SOC)를 1행으로
       -> 240 행 x 28 피처(주파수 14점 x 실수부/허수부)

산출물
    impedance_raw.csv  원본 그대로 (파싱 실습 입력)
    frequencies.csv    원본 그대로 (merge 실습 입력)
    eis_long.csv       파싱 + merge 완료 tidy 형식, 3,360 행 (EDA / Nyquist 플롯용)
    eis_wide.csv       모델링용 피벗 테이블, 240 행 (SOC 회귀/분류, GroupKFold 실습용)

실행:  python prepare.py
"""

import os
import urllib.request

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 0. 경로 설정
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

# Mendeley Data 공개 파일 API 다운로드 URL (v3)
SOURCES = {
    "frequencies.csv": (
        "https://data.mendeley.com/public-files/datasets/mbv3bx847g/files/"
        "c1972cb9-dc05-4d6a-932e-e29d306b613d/file_downloaded"
    ),
    "impedance_raw.csv": (
        "https://data.mendeley.com/public-files/datasets/mbv3bx847g/files/"
        "1ada879d-afc4-4805-b9e8-036afa0604fa/file_downloaded"
    ),
}


def load_source(fname):
    """폴더에 원본이 있으면 그대로 읽고, 없으면 Mendeley에서 내려받아 저장한다."""
    path = os.path.join(HERE, fname)
    if os.path.exists(path):
        print(f"[read ] {fname}")
        return path
    print(f"[fetch] {fname} <- Mendeley")
    req = urllib.request.Request(SOURCES[fname], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)
    return path


# ----------------------------------------------------------------------------
# 1. 원본 적재
#    ★ dtype=str 로 읽는 이유:
#      SOC('010','090'), BATTERY_ID('02') 가 0으로 채워진 문자열이라
#      기본 설정으로 읽으면 pandas 가 정수 10, 90, 2 로 바꿔버린다.
# ----------------------------------------------------------------------------
imp_path = load_source("impedance_raw.csv")
frq_path = load_source("frequencies.csv")

imp = pd.read_csv(imp_path, dtype=str)
frq = pd.read_csv(frq_path)  # FREQUENCY_ID(int), FREQUENCY_VALUE(float)

assert list(imp.columns) == [
    "MEASURE_ID", "SOC", "BATTERY_ID", "FREQUENCY_ID", "IMPEDANCE_VALUE",
], imp.columns
assert len(imp) == 3360, len(imp)
assert len(frq) == 14, len(frq)

# ----------------------------------------------------------------------------
# 2. 복소수 문자열 파싱  "(0.1109735-0.0054730j)" -> real / imag
#    파이썬 내장 complex() 가 괄호 표기까지 그대로 처리한다.
#    (원자료 3,360건 전부 예외 없이 파싱됨 - 실측 확인)
# ----------------------------------------------------------------------------
z = imp["IMPEDANCE_VALUE"].map(complex)

df = pd.DataFrame({
    "measure_id":   imp["MEASURE_ID"],                       # '02_4' 등 24종
    "battery_id":   "B" + imp["BATTERY_ID"],                 # 'B02','B03','B05','B06'
    # 반복 회차: measure_id 의 '_' 뒤 숫자. 셀마다 시작 번호가 다르다
    # (B02/B03 = 4~9,  B05/B06 = 3~8) -> 절대값 비교 금지, 그룹 구분용으로만 사용
    "repeat_id":    imp["MEASURE_ID"].str.split("_").str[1].astype(int),
    "soc":          imp["SOC"].astype(int),                  # '090' -> 90
    "frequency_id": imp["FREQUENCY_ID"].astype(int),
    "z_real":       z.map(lambda c: c.real),                 # Re{Z} [Ohm]
    "z_imag":       z.map(lambda c: c.imag),                 # Im{Z} [Ohm] (전부 음수)
})

# ----------------------------------------------------------------------------
# 3. 주파수 룩업 테이블 merge
# ----------------------------------------------------------------------------
df = df.merge(
    frq.rename(columns={"FREQUENCY_ID": "frequency_id", "FREQUENCY_VALUE": "frequency_hz"}),
    on="frequency_id", how="left", validate="many_to_one",
)
assert df["frequency_hz"].notna().all(), "주파수 merge 실패"

# 파생 컬럼 (Bode 플롯용)
df["minus_z_imag"] = -df["z_imag"]                                   # Nyquist 선도 세로축
df["z_mag"] = (df["z_real"] ** 2 + df["z_imag"] ** 2) ** 0.5          # |Z| [Ohm]
df["z_phase_deg"] = np.degrees(np.arctan2(df["z_imag"], df["z_real"]))  # 위상 [deg]

long_df = df[[
    "measure_id", "battery_id", "repeat_id", "soc",
    "frequency_id", "frequency_hz",
    "z_real", "z_imag", "minus_z_imag", "z_mag", "z_phase_deg",
]].sort_values(["battery_id", "measure_id", "soc", "frequency_id"]).reset_index(drop=True)

# ----------------------------------------------------------------------------
# 4. long -> wide 피벗
#    샘플 1개 = EIS 스펙트럼 1개 = (measure_id, soc) 조합
#    피처   = 주파수 14점 x {실수부, 허수부} = 28개
# ----------------------------------------------------------------------------
def hz_label(v):
    """1000.0 -> '1000',  0.05 -> '0.05'  (컬럼명용)"""
    return f"{v:g}"


wide = long_df.pivot_table(
    index=["measure_id", "battery_id", "soc"],
    columns="frequency_hz",
    values=["z_real", "z_imag"],
)

# MultiIndex 컬럼 -> 'Zre_0.05Hz' / 'Zim_0.05Hz' 형태로 평탄화
prefix = {"z_real": "Zre", "z_imag": "Zim"}
wide.columns = [f"{prefix[a]}_{hz_label(b)}Hz" for a, b in wide.columns]

# 컬럼 순서: 실수부 14개(주파수 오름차순) -> 허수부 14개
freq_sorted = sorted(long_df["frequency_hz"].unique())
ordered = ([f"Zre_{hz_label(f)}Hz" for f in freq_sorted]
           + [f"Zim_{hz_label(f)}Hz" for f in freq_sorted])
wide = wide[ordered].reset_index()

# sample_id 를 맨 앞에
wide.insert(0, "sample_id", wide["measure_id"] + "_soc" + wide["soc"].astype(str).str.zfill(3))
wide.insert(3, "repeat_id", wide["measure_id"].str.split("_").str[1].astype(int))
wide = wide.sort_values(["battery_id", "measure_id", "soc"]).reset_index(drop=True)

# ----------------------------------------------------------------------------
# 5. 검증 (실측 기준값과 대조)
# ----------------------------------------------------------------------------
assert len(long_df) == 3360, len(long_df)
assert len(wide) == 240, len(wide)
# 식별자 5개(sample_id, measure_id, battery_id, repeat_id, soc) + 피처 28개
assert wide.shape[1] == 5 + 28, wide.shape
assert wide.isna().sum().sum() == 0, "결측 발생"
assert wide["battery_id"].nunique() == 4
assert wide["measure_id"].nunique() == 24
assert wide["soc"].nunique() == 10
assert (long_df["z_imag"] < 0).all(), "Im{Z} 부호 이상"

# ----------------------------------------------------------------------------
# 6. 저장
# ----------------------------------------------------------------------------
long_out = os.path.join(HERE, "eis_long.csv")
wide_out = os.path.join(HERE, "eis_wide.csv")
long_df.to_csv(long_out, index=False, encoding="utf-8", float_format="%.10g")
wide.to_csv(wide_out, index=False, encoding="utf-8", float_format="%.10g")

# 데이터 사전 — 실제 산출된 컬럼에서 자동 생성한다(수기 관리 시 내용이 어긋나므로).
DESC = {
    "sample_id":    ("문자열", "-",   "스펙트럼 1개의 고유 키. measure_id + SOC"),
    "measure_id":   ("문자열", "-",   "EIS 측정 1회(셀 1개의 1회 스윕)의 원본 코드. 24종"),
    "battery_id":   ("문자열", "-",   "셀 식별자 B02/B03/B05/B06. GroupKFold 의 groups 로 사용"),
    "repeat_id":    ("정수",   "-",   "반복 회차(원본 번호). 셀마다 시작값이 다름(B02·B03=4~9, B05·B06=3~8)"),
    "soc":          ("정수",   "%",   "충전 상태 10~100(10 간격). 예측 대상(target)"),
    "frequency_id": ("정수",   "-",   "주파수 인덱스 0~13"),
    "frequency_hz": ("실수",   "Hz",  "가진 주파수 0.05~1000"),
    "z_real":       ("실수",   "Ohm", "임피던스 실수부 Re{Z}"),
    "z_imag":       ("실수",   "Ohm", "임피던스 허수부 Im{Z} (본 데이터는 전 구간 음수)"),
    "minus_z_imag": ("실수",   "Ohm", "-Im{Z}. Nyquist 선도의 세로축"),
    "z_mag":        ("실수",   "Ohm", "임피던스 크기 |Z|"),
    "z_phase_deg":  ("실수",   "deg", "임피던스 위상각"),
}
rows = []
for col in long_df.columns:
    t, u, d = DESC[col]
    rows.append(["eis_long.csv", col, t, u, d])
for col in wide.columns:
    if col in DESC:
        t, u, d = DESC[col]
    else:
        part, hz = ("실수부 Re{Z}", col[4:-2]) if col.startswith("Zre_") else ("허수부 Im{Z}", col[4:-2])
        t, u, d = "실수", "Ohm", f"{hz} Hz 에서의 임피던스 {part} (피처)"
    rows.append(["eis_wide.csv", col, t, u, d])

dict_out = os.path.join(HERE, "eis_data_dictionary.csv")
pd.DataFrame(rows, columns=["파일", "컬럼명", "자료형", "단위", "설명"]).to_csv(
    dict_out, index=False, encoding="utf-8-sig"
)

print()
print(f"eis_long.csv : {long_df.shape[0]:,} 행 x {long_df.shape[1]} 열"
      f"  ({os.path.getsize(long_out)/1024:.1f} KB)")
print(f"eis_wide.csv : {wide.shape[0]:,} 행 x {wide.shape[1]} 열"
      f"  ({os.path.getsize(wide_out)/1024:.1f} KB)")
print(f"  샘플 {len(wide)}개 = 셀 {wide['battery_id'].nunique()}개"
      f" x 측정 {wide.groupby('battery_id')['measure_id'].nunique().iloc[0]}회"
      f" x SOC {wide['soc'].nunique()}단계")
print(f"  피처 28개 = 주파수 {len(freq_sorted)}점 x (실수부, 허수부)")
print(f"  GroupKFold 그룹 후보: battery_id({wide['battery_id'].nunique()}개),"
      f" measure_id({wide['measure_id'].nunique()}개)")
