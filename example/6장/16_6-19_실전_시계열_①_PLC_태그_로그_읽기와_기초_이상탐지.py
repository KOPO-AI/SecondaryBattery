# -*- coding: utf-8 -*-
"""6장 예제 — 6.19 실전 시계열 ① PLC 태그 로그 읽기와 기초 이상탐지 (실습 40분) `필수`

교재 출처 : manuscript/61_ch6_최적화.md
포함 소절 : 데이터 소개 — 태그 사전과 로그 / 운전·정지 구분과 전체 조망 / 규칙 ① PV−SV 편차 — 가장 단순하고 가장 현장적인 기준 / 규칙 ② 이동 z-score와 연속 N회 규칙 — 설정값이 없는 태그를 위해 / 채점 — 정답지와 대조 / 정리 — 기초 규칙 세 줄
실행 방법 : example 폴더에서  python "6장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --------------------------------------------------------------------
# [데이터 소개 — 태그 사전과 로그]
# --------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

tags = pd.read_csv("data/plc/plc_tag_list.csv")
print(tags.shape)
print(tags[["태그명", "설비", "구분", "설명", "단위"]].to_string(index=False))

# 교재 실행 결과 ------------------------------------------------------
#   (24, 7)
#                       태그명  설비 구분               설명    단위
#               EC_M_MMF_SV  믹서 SV  메인 모터 고속 RPM 설정   rpm
#               EC_M_MMF_PV  믹서 PV 메인 모터 고속 RPM 현재값   rpm
#             EC_M_MMF_A_PV  믹서 PV      메인 모터 고속 전류     A
#          EC_M_TPD_TEMP_PV  믹서 PV    탱크(PD) 온도 현재값     ℃
#        EC_C_LINE_SPEED_SV  코팅 SV      코터 라인 속도 설정 m/min
#        EC_C_LINE_SPEED_PV  코팅 PV     코터 라인 속도 현재값 m/min
#          EC_C_DR1_TEMP_SV  코팅 SV    건조로 1구간 온도 설정     ℃
#          EC_C_DR1_TEMP_PV  코팅 PV   건조로 1구간 온도 현재값     ℃
#          EC_C_DR2_TEMP_SV  코팅 SV    건조로 2구간 온도 설정     ℃
#          EC_C_DR2_TEMP_PV  코팅 PV   건조로 2구간 온도 현재값     ℃
#          EC_C_DR3_TEMP_SV  코팅 SV    건조로 3구간 온도 설정     ℃
#          EC_C_DR3_TEMP_PV  코팅 PV   건조로 3구간 온도 현재값     ℃
#          EC_C_DR4_TEMP_SV  코팅 SV    건조로 4구간 온도 설정     ℃
#          EC_C_DR4_TEMP_PV  코팅 PV   건조로 4구간 온도 현재값     ℃
#         EC_C_UNWIN_TEN_SV  코팅 SV       언와인더 장력 설정     N
#         EC_C_UNWIN_TEN_PV  코팅 PV      언와인더 장력 현재값     N
#         EC_C_REWIN_TEN_SV  코팅 SV       리와인더 장력 설정     N
#         EC_C_REWIN_TEN_PV  코팅 PV      리와인더 장력 현재값     N

# --------------------------------------------------------------------
# [데이터 소개 — 태그 사전과 로그]
# --------------------------------------------------------------------
log = pd.read_csv("data/plc/plc_line_log.csv", parse_dates=["timestamp"], index_col="timestamp")
print(log.shape)
print(log.index.min(), "~", log.index.max(), "| 간격:", log.index[1] - log.index[0])
print(log[["EC_C_LINE_SPEED_PV", "EC_C_DR2_TEMP_PV", "EC_C_UNWIN_TEN_PV", "EC_N_금형횟수_현재값"]].describe().round(2))

# 교재 실행 결과 ------------------------------------------------------
#   (10080, 24)
#   2026-09-01 00:00:00 ~ 2026-09-07 23:59:00 | 간격: 0 days 00:01:00
#          EC_C_LINE_SPEED_PV  EC_C_DR2_TEMP_PV  EC_C_UNWIN_TEN_PV  EC_N_금형횟수_현재값
#   count            10080.00          10080.00           10080.00       10080.00
#   mean                31.60            112.48              58.18      272948.09
#   std                  5.89              2.56               9.94      163219.61
#   min                  0.00            107.55               0.00           0.00
#   25%                 29.95            109.97              58.94      134130.50
#   50%                 30.48            112.81              59.83      268367.50
#   75%                 34.98            114.97              60.67      402599.50
#   max                 35.94            121.69              64.34      599893.00

# --------------------------------------------------------------------
# [운전·정지 구분과 전체 조망]
# --------------------------------------------------------------------
운전중 = log["EC_C_LINE_SPEED_PV"] > 1          # 라인 속도가 0 근처면 정지
print("정지 분 수:", (~운전중).sum(), "/", len(log),
      "→ 운전 중 비율", round(운전중.mean() * 100, 1), "%")

fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
log["EC_C_LINE_SPEED_PV"].plot(ax=axes[0], lw=0.8)
axes[0].set_ylabel("라인 속도\n(m/min)")
log["EC_C_DR2_TEMP_PV"].plot(ax=axes[1], lw=0.8, label="PV(현재값)")
log["EC_C_DR2_TEMP_SV"].plot(ax=axes[1], lw=1.2, ls="--", color="k", label="SV(설정값)")
axes[1].set_ylabel("건조로 2구간\n온도(℃)"); axes[1].legend(loc="upper left")
log["EC_C_UNWIN_TEN_PV"].plot(ax=axes[2], lw=0.8)
axes[2].set_ylabel("언와인더\n장력(N)")
log["EC_N_금형횟수_현재값"].plot(ax=axes[3], lw=1.0)
axes[3].axhline(600000, color="r", ls=":", label="경보 설정 600,000")
axes[3].set_ylabel("금형 타발\n누적 횟수"); axes[3].legend(loc="upper left")
plt.tight_layout(); plt.savefig("fig_6_39_plc_overview.png", dpi=150); plt.close()

# 교재 실행 결과 ------------------------------------------------------
#   정지 분 수: 280 / 10080 → 운전 중 비율 97.2 %

# --------------------------------------------------------------------
# [규칙 ① PV−SV 편차 — 가장 단순하고 가장 현장적인 기준]
# --------------------------------------------------------------------
편차 = log["EC_C_DR2_TEMP_PV"] - log["EC_C_DR2_TEMP_SV"]
규칙1 = (편차.abs() > 3.0) & 운전중
print("|PV-SV| > 3℃ 인 분 수:", 규칙1.sum())
print(편차[규칙1].round(2).to_string())

# 교재 실행 결과 ------------------------------------------------------
#   |PV-SV| > 3℃ 인 분 수: 10
#   timestamp
#   2026-09-03 14:20:00    8.61
#   2026-09-03 14:21:00    8.49
#   2026-09-03 14:22:00    8.88
#   2026-09-03 14:23:00    8.58
#   2026-09-03 14:24:00    9.01
#   2026-09-04 12:00:00   -3.24
#   2026-09-04 12:02:00   -3.02
#   2026-09-05 03:10:00    6.69
#   2026-09-05 03:11:00    6.41
#   2026-09-05 03:12:00    6.43

# --------------------------------------------------------------------
# [규칙 ① PV−SV 편차 — 가장 단순하고 가장 현장적인 기준]
# --------------------------------------------------------------------
설정변경 = log["EC_C_DR2_TEMP_SV"].diff().abs() > 0
안정화중 = 설정변경.rolling("15min").max().fillna(0) > 0      # 변경 후 15분간 True
규칙1 = (편차.abs() > 3.0) & 운전중 & ~안정화중
print("안정화 15분 제외 후:", 규칙1.sum(), "분")

# 교재 실행 결과 ------------------------------------------------------
#   안정화 15분 제외 후: 8 분

# --------------------------------------------------------------------
# [규칙 ② 이동 z-score와 연속 N회 규칙 — 설정값이 없는 태그를 위해]
# --------------------------------------------------------------------
def rolling_z(s, window="60min", min_periods=30):
    """직전 window 구간(현재값 제외)의 평균·표준편차로 표준화한 z-score"""
    m = s.rolling(window, min_periods=min_periods).mean().shift(1)
    sd = s.rolling(window, min_periods=min_periods).std().shift(1)
    return (s - m) / sd

z2 = rolling_z(log.loc[운전중, "EC_C_DR2_TEMP_PV"])
초과 = z2.abs() > 3
연속3 = 초과.rolling(3).sum() >= 3                    # 3분 연속 초과일 때만
print("2구간 온도 |z| > 3:", 초과.sum(), "분 → 연속 3분 규칙 후:", 연속3.sum(), "분")
print(z2[연속3].round(1).to_string())
z속도 = rolling_z(log.loc[운전중, "EC_C_LINE_SPEED_PV"])
print("라인 속도 12:00 전후 z:", z속도.loc["2026-09-04 11:59":"2026-09-04 12:02"].round(1).tolist())

# 교재 실행 결과 ------------------------------------------------------
#   2구간 온도 |z| > 3: 76 분 → 연속 3분 규칙 후: 6 분
#   timestamp
#   2026-09-03 14:22:00    5.3
#   2026-09-03 14:23:00    4.1
#   2026-09-03 14:24:00    3.8
#   2026-09-05 03:12:00    4.6
#   2026-09-06 14:37:00   -4.1
#   2026-09-06 14:38:00   -3.0
#   라인 속도 12:00 전후 z: [-1.2, 33.6, 7.3, 5.3]

# --------------------------------------------------------------------
# [규칙 ② 이동 z-score와 연속 N회 규칙 — 설정값이 없는 태그를 위해]
# --------------------------------------------------------------------
z_상부롤 = rolling_z(log.loc[운전중, "EC_P_UPPER_ROLL_TEMP_PV"])
연속3_롤 = (z_상부롤.abs() > 3).rolling(3).sum() >= 3
print("상부 롤 온도 연속 3분 검출:", 연속3_롤.sum(), "분 —", 연속3_롤[연속3_롤].index.strftime("%m-%d %H:%M").tolist())

# 교재 실행 결과 ------------------------------------------------------
#   상부 롤 온도 연속 3분 검출: 14 분 — ['09-01 10:08', '09-01 10:09', '09-01 13:25', '09-01 13:26', '09-02 19:36', '09-02 19:37', '09-02 19:38', '09-02 19:39', '09-02 20:43', '09-04 17:17', '09-04 17:18', '09-05 13:50', '09-06 18:31', '09-07 17:02']

# --------------------------------------------------------------------
# [채점 — 정답지와 대조]
# --------------------------------------------------------------------
events = pd.read_csv("data/plc/plc_events.csv")
print(events[["사건", "설비", "태그", "시작", "종료", "유형", "이상여부"]].to_string(index=False))

def 사건별_검출(검출, events):
    """검출(True/False 시계열)을 사건 구간과 대조 — 사건별 검출 분 수와 첫 검출까지 지연(분)"""
    rows = []
    for _, e in events[events["이상여부"] == "Y"].iterrows():
        구간 = 검출.loc[e["시작"]:e["종료"]]
        첫검출 = 구간[구간].index.min() if 구간.any() else pd.NaT
        지연 = (첫검출 - pd.Timestamp(e["시작"])).total_seconds() / 60 if 구간.any() else np.nan
        rows.append([e["사건"], e["태그"], len(구간), int(구간.sum()), 지연])
    표 = pd.DataFrame(rows, columns=["사건", "태그", "구간(분)", "검출(분)", "첫 검출 지연(분)"])
    사건구간 = pd.Series(False, index=검출.index)
    for _, e in events[events["이상여부"] == "Y"].iterrows():
        사건구간.loc[e["시작"]:e["종료"]] = True
    표.attrs["오탐(분)"] = int((검출 & ~사건구간).sum())
    return 표

표1 = 사건별_검출(규칙1, events)
print(표1.to_string(index=False)); print("사건 밖 검출(오탐):", 표1.attrs["오탐(분)"], "분")

# 교재 실행 결과 ------------------------------------------------------
#   사건     설비                      태그               시작               종료     유형 이상여부
#   C1     코팅        EC_C_DR2_TEMP_PV 2026-09-03 14:20 2026-09-03 14:24   점 이상    Y
#   C4     코팅       EC_C_UNWIN_TEN_PV 2026-09-03 20:00 2026-09-03 20:30  집단 이상    Y
#   M1     믹서           EC_M_MMF_A_PV 2026-09-04 15:20 2026-09-04 16:50  맥락 이상    Y
#   R1     코팅      EC_C_LINE_SPEED_SV 2026-09-04 12:00 2026-09-07 23:59 레시피 변경    N
#   C2     코팅        EC_C_DR2_TEMP_PV 2026-09-05 03:10 2026-09-05 03:12   점 이상    Y
#   C3     코팅        EC_C_DR3_TEMP_PV 2026-09-06 09:00 2026-09-06 15:00  맥락 이상    Y
#   P1    프레스 EC_P_UPPER_ROLL_TEMP_PV 2026-09-07 10:00 2026-09-07 23:59  집단 이상    Y
#   S0 코팅·프레스      EC_C_LINE_SPEED_PV         매일 06:00         매일 06:40  계획 정지    N
#   사건                      태그  구간(분)  검출(분)  첫 검출 지연(분)
#   C1        EC_C_DR2_TEMP_PV      5      5         0.0
#   C4       EC_C_UNWIN_TEN_PV     31      0         NaN
#   M1           EC_M_MMF_A_PV     91      0         NaN
#   C2        EC_C_DR2_TEMP_PV      3      3         0.0
#   C3        EC_C_DR3_TEMP_PV    361      0         NaN
#   P1 EC_P_UPPER_ROLL_TEMP_PV    840      0         NaN
#   사건 밖 검출(오탐): 0 분

# --------------------------------------------------------------------
# [채점 — 정답지와 대조]
# --------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
편차[운전중].plot(ax=axes[0], lw=0.7, color="gray")
편차[규칙1].plot(ax=axes[0], style="r.", ms=6, label="|PV-SV|>3℃ 검출(안정화 제외)")
axes[0].axhline(3, color="r", ls=":"); axes[0].axhline(-3, color="r", ls=":")
axes[0].set_ylabel("건조로 2구간\nPV-SV (℃)"); axes[0].legend(loc="upper left")
z2.plot(ax=axes[1], lw=0.7, color="gray", label="이동 z-score")
z2[연속3].plot(ax=axes[1], style="r.", ms=6, label="|z|>3 연속 3분")
axes[1].axhline(3, color="r", ls=":"); axes[1].axhline(-3, color="r", ls=":")
axes[1].set_ylabel("건조로 2구간\n이동 z-score"); axes[1].legend(loc="upper left")
plt.tight_layout(); plt.savefig("fig_6_40_plc_rules.png", dpi=150); plt.close()

