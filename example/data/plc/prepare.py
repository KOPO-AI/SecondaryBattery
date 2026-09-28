# -*- coding: utf-8 -*-
"""양극 전극 라인 PLC 태그 로그(모사) 생성 스크립트 — data/plc/

원본: 현장에서 제공한 PLC 태그 목록(양극 라인 믹서·코팅·프레스·노칭, 60 태그 + 믹서 상세 176 태그).
      태그 이름·설명·최소/최대 범위는 그 목록에서 그대로 가져왔고, 시계열 값은 이 스크립트가 만든 것이다.
      (실제 PLC 주소·통신 설정은 싣지 않는다.)

산출 파일
    plc_tag_list.csv   태그 사전 25행 — 태그명·설비·구분(SV/PV)·설명·단위·최소·최대
    plc_line_log.csv   1분 주기 7일치 로그 10,080행 × 25열 (timestamp + 태그 24개)
    plc_events.csv     심어 둔 사건 8건 — 시작·종료·태그·유형·이상 여부 (정답지)

재현: python prepare.py   (seed 42 고정, 어느 폴더에서 실행해도 이 파일 옆에 저장한다)
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)

# ── 1. 태그 사전 ───────────────────────────────────────────────────────
TAGS = [
    # 태그명, 설비, 구분, 설명, 단위, 최소, 최대
    ("EC_M_MMF_SV",          "믹서", "SV", "메인 모터 고속 RPM 설정",     "rpm",   0, 3008),
    ("EC_M_MMF_PV",          "믹서", "PV", "메인 모터 고속 RPM 현재값",   "rpm",   0, 3008),
    ("EC_M_MMF_A_PV",        "믹서", "PV", "메인 모터 고속 전류",         "A",     0,  200),
    ("EC_M_TPD_TEMP_PV",     "믹서", "PV", "탱크(PD) 온도 현재값",        "℃",   -40,  100),
    ("EC_C_LINE_SPEED_SV",   "코팅", "SV", "코터 라인 속도 설정",         "m/min", 0,  100),
    ("EC_C_LINE_SPEED_PV",   "코팅", "PV", "코터 라인 속도 현재값",       "m/min", 0,  100),
    ("EC_C_DR1_TEMP_SV",     "코팅", "SV", "건조로 1구간 온도 설정",      "℃",     0,  200),
    ("EC_C_DR1_TEMP_PV",     "코팅", "PV", "건조로 1구간 온도 현재값",    "℃",     0,  200),
    ("EC_C_DR2_TEMP_SV",     "코팅", "SV", "건조로 2구간 온도 설정",      "℃",     0,  200),
    ("EC_C_DR2_TEMP_PV",     "코팅", "PV", "건조로 2구간 온도 현재값",    "℃",     0,  200),
    ("EC_C_DR3_TEMP_SV",     "코팅", "SV", "건조로 3구간 온도 설정",      "℃",     0,  200),
    ("EC_C_DR3_TEMP_PV",     "코팅", "PV", "건조로 3구간 온도 현재값",    "℃",     0,  200),
    ("EC_C_DR4_TEMP_SV",     "코팅", "SV", "건조로 4구간 온도 설정",      "℃",     0,  200),
    ("EC_C_DR4_TEMP_PV",     "코팅", "PV", "건조로 4구간 온도 현재값",    "℃",     0,  200),
    ("EC_C_UNWIN_TEN_SV",    "코팅", "SV", "언와인더 장력 설정",          "N",     0,  300),
    ("EC_C_UNWIN_TEN_PV",    "코팅", "PV", "언와인더 장력 현재값",        "N",     0,  300),
    ("EC_C_REWIN_TEN_SV",    "코팅", "SV", "리와인더 장력 설정",          "N",     0,  300),
    ("EC_C_REWIN_TEN_PV",    "코팅", "PV", "리와인더 장력 현재값",        "N",     0,  300),
    ("EC_P_DRIV_LINE_SPEED_PV", "프레스", "PV", "프레스 라인 속도 현재값", "m/min", 0,  100),
    ("EC_P_UPPER_ROLL_TEMP_PV", "프레스", "PV", "상부 롤 온도 현재값",     "℃",     0,  200),
    ("EC_P_LOWER_ROLL_TEMP_PV", "프레스", "PV", "하부 롤 온도 현재값",     "℃",     0,  200),
    ("EC_P_UNWIN_TEN_PV",    "프레스", "PV", "프레스 언와인더 장력 현재값", "N",   0,  300),
    ("EC_N_금형횟수_현재값",   "노칭", "PV", "노칭 금형 타발 횟수 누적",    "회",    0, 9999999),
    ("EC_N_금형횟수_경보_설정", "노칭", "SV", "금형 교체 경보 횟수 설정",   "회",    0, 9999999),
]
tag_df = pd.DataFrame(TAGS, columns=["태그명", "설비", "구분", "설명", "단위", "최소", "최대"])
tag_df.to_csv(os.path.join(HERE, "plc_tag_list.csv"), index=False, encoding="utf-8-sig")

# ── 2. 시간축: 2026-09-01 00:00 ~ 09-07 23:59, 1분 ────────────────────
idx = pd.date_range("2026-09-01 00:00", periods=7 * 1440, freq="1min")
n = len(idx)
t = np.arange(n)
day = t // 1440
minute_of_day = t % 1440
ambient = 1.0 * np.sin(2 * np.pi * (minute_of_day - 6 * 60) / 1440)   # 하루 주기 외기 영향(±1℃)


def ar1(n, sd, phi=0.9):
    """느리게 출렁이는 잡음(AR(1))"""
    e = rng.normal(0, sd, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x * np.sqrt(1 - phi ** 2)


def rows(ts_from, ts_to):
    """시각 문자열 구간 → 불리언 마스크 (끝 시각 포함)"""
    return (idx >= pd.Timestamp(ts_from)) & (idx <= pd.Timestamp(ts_to))


df = pd.DataFrame(index=idx)
df.index.name = "timestamp"

# ── 3. 코터(코팅기) ───────────────────────────────────────────────────
speed_sv = np.where(idx < pd.Timestamp("2026-09-04 12:00"), 30.0, 35.0)   # 4일차 정오 레시피 변경(정상)
running = ~((minute_of_day >= 6 * 60) & (minute_of_day < 6 * 60 + 40))    # 매일 06:00~06:40 계획 정지
speed_pv = np.where(running, speed_sv + ar1(n, 0.25), 0.0)
df["EC_C_LINE_SPEED_SV"] = speed_sv
df["EC_C_LINE_SPEED_PV"] = speed_pv.round(2)

dr_sv_base = {1: 90.0, 2: 110.0, 3: 120.0, 4: 100.0}
for k, sv in dr_sv_base.items():
    sv_arr = np.full(n, sv)
    if k in (2, 3):                                   # 레시피 변경 시 2·3구간 설정 +5℃
        sv_arr = np.where(idx < pd.Timestamp("2026-09-04 12:00"), sv, sv + 5.0)
    sv_lag = pd.Series(sv_arr).ewm(halflife=10).mean().to_numpy()   # 설정 변경 후 실제 온도는 몇 분에 걸쳐 따라간다
    pv = sv_lag + ar1(n, 0.6) + 0.3 * ambient + rng.normal(0, 0.15, n)
    df[f"EC_C_DR{k}_TEMP_SV"] = sv_arr
    df[f"EC_C_DR{k}_TEMP_PV"] = pv

# 사건 C1·C2: 2구간 온도 스파이크(점 이상)
df.loc[rows("2026-09-03 14:20", "2026-09-03 14:24"), "EC_C_DR2_TEMP_PV"] += 9.0
df.loc[rows("2026-09-05 03:10", "2026-09-05 03:12"), "EC_C_DR2_TEMP_PV"] += 7.0
# 사건 C3: 3구간 온도 완만한 하락(히터 열화, 맥락 이상) — 6시간 동안 서서히 -5℃, 이후 복구
m = rows("2026-09-06 09:00", "2026-09-06 15:00")
ramp = np.linspace(0, -5.0, m.sum())
df.loc[m, "EC_C_DR3_TEMP_PV"] += ramp

ten_sv_u, ten_sv_r = 60.0, 80.0
df["EC_C_UNWIN_TEN_SV"] = ten_sv_u
df["EC_C_UNWIN_TEN_PV"] = np.where(running, ten_sv_u + ar1(n, 1.2), 0.0)
df["EC_C_REWIN_TEN_SV"] = ten_sv_r
df["EC_C_REWIN_TEN_PV"] = np.where(running, ten_sv_r + ar1(n, 1.4), 0.0)
# 사건 C4: 언와인더 장력 30분간 -15 N (롤 교체 후 세팅 오류)
df.loc[rows("2026-09-03 20:00", "2026-09-03 20:30"), "EC_C_UNWIN_TEN_PV"] -= 15.0

# ── 4. 프레스(압연기) ─────────────────────────────────────────────────
p_running = running
df["EC_P_DRIV_LINE_SPEED_PV"] = np.where(p_running, 25.0 + ar1(n, 0.2), 0.0).round(2)
df["EC_P_UPPER_ROLL_TEMP_PV"] = 80.0 + ar1(n, 0.4) + 0.2 * ambient
df["EC_P_LOWER_ROLL_TEMP_PV"] = 80.0 + ar1(n, 0.4) + 0.2 * ambient
df["EC_P_UNWIN_TEN_PV"] = np.where(p_running, 50.0 + ar1(n, 1.0), 0.0)
# 사건 P1: 상부 롤 온도가 2시간에 걸쳐 +4℃ 상승 후 유지(냉각수 이상)
m = rows("2026-09-07 10:00", "2026-09-07 12:00")
df.loc[m, "EC_P_UPPER_ROLL_TEMP_PV"] += np.linspace(0, 4.0, m.sum())
df.loc[idx > pd.Timestamp("2026-09-07 12:00"), "EC_P_UPPER_ROLL_TEMP_PV"] += 4.0

# ── 5. 믹서: 150분 배치 주기(저속 20분 → 고속 90분 → 대기 40분) ────────
cycle = t % 150
phase_high = (cycle >= 20) & (cycle < 110)
phase_low = cycle < 20
batch_no = t // 150
mmf_sv = np.where(phase_high, 2400.0, 0.0)
mmf_pv = np.where(phase_high, 2400.0 + rng.normal(0, 12, n), 0.0)
min_in_high = np.clip(cycle - 20, 0, 90)
# 전류: RPM 비례 + 배치 진행에 따른 점도 상승분 + 잡음
amp = np.where(phase_high, 0.04 * mmf_pv + 0.25 * min_in_high + rng.normal(0, 1.5, n), 0.0)
# 사건 M1: 배치 35번(4일차 오후)에서 같은 RPM인데 전류가 25% 높음(슬러리 점도 이상)
amp = np.where((batch_no == 35) & phase_high, amp * 1.25, amp)
tank = 25.0 + 0.07 * min_in_high + ar1(n, 0.3) + 0.5 * ambient
tank = np.where(cycle >= 110, 25.0 + 6.3 * np.exp(-(cycle - 110) / 12.0) + 0.5 * ambient + ar1(n, 0.3), tank)
df["EC_M_MMF_SV"] = mmf_sv
df["EC_M_MMF_PV"] = np.clip(mmf_pv, 0, None).round(1)
df["EC_M_MMF_A_PV"] = np.clip(amp, 0, None).round(1)
df["EC_M_TPD_TEMP_PV"] = tank

# ── 6. 노칭: 금형 타발 횟수 누적(운전 중 분당 110회), 경보 설정 600,000회 ─
punch = np.where(p_running, 110 + rng.integers(-3, 4, n), 0)
count = np.zeros(n, dtype=np.int64)
c = 590_000                                           # 1일차 시작 시점의 누적 횟수
for i in range(n):
    c += int(punch[i])
    if c >= 600_000:                                  # 경보 도달 → 금형 교체 후 0에서 다시 센다
        c = 0
    count[i] = c
df["EC_N_금형횟수_현재값"] = count
df["EC_N_금형횟수_경보_설정"] = 600_000

# ── 7. 소수점 정리·저장 ───────────────────────────────────────────────
temp_cols = [c for c in df.columns if "TEMP" in c]
ten_cols = [c for c in df.columns if "TEN_PV" in c]
df[temp_cols] = df[temp_cols].round(2)
df[ten_cols] = df[ten_cols].round(2)
df = df[[t[0] for t in TAGS]]                          # 태그 사전 순서로 열 정렬
df.to_csv(os.path.join(HERE, "plc_line_log.csv"), encoding="utf-8-sig")

events = pd.DataFrame([
    ["C1", "코팅",   "EC_C_DR2_TEMP_PV",        "2026-09-03 14:20", "2026-09-03 14:24", "점 이상",   "건조로 2구간 순간 과열 +9℃",           "Y"],
    ["C4", "코팅",   "EC_C_UNWIN_TEN_PV",       "2026-09-03 20:00", "2026-09-03 20:30", "집단 이상", "언와인더 장력 -15 N (롤 교체 후 세팅 오류)", "Y"],
    ["M1", "믹서",   "EC_M_MMF_A_PV",           "2026-09-04 15:20", "2026-09-04 16:50", "맥락 이상", "같은 RPM에서 전류 +25% (슬러리 점도 이상)", "Y"],
    ["R1", "코팅",   "EC_C_LINE_SPEED_SV",      "2026-09-04 12:00", "2026-09-07 23:59", "레시피 변경", "라인 속도 30→35 m/min, 2·3구간 온도 설정 +5℃ (정상 운전 변경)", "N"],
    ["C2", "코팅",   "EC_C_DR2_TEMP_PV",        "2026-09-05 03:10", "2026-09-05 03:12", "점 이상",   "건조로 2구간 순간 과열 +7℃",           "Y"],
    ["C3", "코팅",   "EC_C_DR3_TEMP_PV",        "2026-09-06 09:00", "2026-09-06 15:00", "맥락 이상", "건조로 3구간 6시간에 걸쳐 -5℃ 완만 하락 (히터 열화)", "Y"],
    ["P1", "프레스", "EC_P_UPPER_ROLL_TEMP_PV", "2026-09-07 10:00", "2026-09-07 23:59", "집단 이상", "상부 롤 온도 2시간에 걸쳐 +4℃ 상승 후 유지 (냉각수 이상)", "Y"],
    ["S0", "코팅·프레스", "EC_C_LINE_SPEED_PV", "매일 06:00", "매일 06:40", "계획 정지", "라인 속도 0 — 이상이 아니라 정지. 분석에서 제외해야 한다", "N"],
], columns=["사건", "설비", "태그", "시작", "종료", "유형", "설명", "이상여부"])
events.to_csv(os.path.join(HERE, "plc_events.csv"), index=False, encoding="utf-8-sig")

print("plc_tag_list.csv", tag_df.shape, "| plc_line_log.csv", df.shape, "| plc_events.csv", events.shape)
