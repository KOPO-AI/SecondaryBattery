# -*- coding: utf-8 -*-
"""2장 — 장 전체 예제 실행본

교재 2장의 파이썬 코드를 등장 순서대로 이어 붙였다.
절별 파일은 앞 절에서 만든 변수를 이어 쓰는 경우가 있어 단독 실행이 안 될 수 있는데,
이 파일은 처음부터 끝까지 순서대로 실행되므로 그대로 돌려 볼 수 있다.

실행 : example 폴더에서  python "2장/_장전체_실행본.py"
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""
import matplotlib
matplotlib.use("Agg")   # 창을 띄우지 않고 그림을 파일로만 처리


# ====================================================================
# 파이썬 시작하기 `필수` / 첫 코드 실행 `필수`
# ====================================================================
print("Hello, Battery World!")
print(3.7 * 50)


# ====================================================================
# 변수와 자료형 `필수` / 변수와 자료형 — 데이터에 이름과 종류가 있다 `필수`
# ====================================================================
voltage = 3.72          # 셀 전압 (V)
capacity = 2850         # 셀 용량 (mAh)
lot_id = "LOT-2408-A17" # 로트 번호
is_pass = True          # 합격 여부
print(voltage, capacity, lot_id, is_pass)
print(type(voltage), type(capacity), type(lot_id), type(is_pass))


# ====================================================================
# 변수와 자료형 `필수` / 실수 오차와 문자열 다루기 `필수`
# ====================================================================
print(3.65 - 3.7)              # 부동소수점 오차
lot_id = "LOT-2408-A17"
print(lot_id.replace("-", "_"))
print(lot_id.split("-"))
print(len(lot_id))


# ====================================================================
# 변수와 자료형 `필수` / f-string: 보고서 문장 만들기 `필수`
# ====================================================================
voltage = 3.7215
lot_id = "LOT-2408-A17"
print(f"로트 {lot_id}의 평균 전압은 {voltage:.2f} V입니다.")
print(f"용량: {2850:,} mAh")
print(f"|{voltage:>8.2f}|{'합격':>6}|")   # 폭 8칸 오른쪽 정렬


# ====================================================================
# 변수와 자료형 `필수` / 형변환 `필수`
# ====================================================================
raw = "3.72"            # 장비가 넘겨준 문자열
voltage = float(raw)    # 실수로 변환
print(voltage + 0.05)
print(int(3.99))        # 소수점 이하 버림
print(str(2850) + " mAh")


# ====================================================================
# 연산자와 제어문 / 산술·비교·논리 연산자 `필수`
# ====================================================================
capacity = 2850
n_cells = 4
print(capacity / n_cells, capacity // n_cells, capacity % n_cells)  # 나눗셈·몫·나머지
voltage = 3.72
temp = 45.3
print(voltage > 3.7, voltage == 3.72, voltage != 3.72)
print(voltage > 3.6 and temp < 40)   # and: 둘 다 참이어야 참
print(voltage > 3.6 or temp < 40)    # or: 하나만 참이어도 참
print(not voltage > 3.6)             # not: 참거짓 뒤집기


# ====================================================================
# 연산자와 제어문 / if/elif/else: 불량 판정 로직 `필수`
# ====================================================================
voltage = 3.42

if voltage >= 3.6:
    print("정상")
elif voltage >= 3.3:
    print("재검사 대상")
else:
    print("불량")

voltage = 3.72
temp = 45.3

if voltage >= 3.6 and temp <= 40:
    result = "합격"
else:
    result = "불합격"

print(f"전압 {voltage} V, 온도 {temp} ℃ → {result}")


# ====================================================================
# 연산자와 제어문 / for 반복문: 로트 순회 `필수`
# ====================================================================
voltages = [3.71, 3.68, 3.65, 3.42, 3.70]
fail_count = 0

for v in voltages:
    print(f"측정 전압: {v} V")
    if v < 3.6:
        fail_count += 1

print(f"불량 셀 수: {fail_count}개")

for cycle in range(1, 4):
    print(f"{cycle}회차 충방전 시험 진행")


# ====================================================================
# 연산자와 제어문 / while·break·continue `선택 학습(자율 복습)`
# ====================================================================
soc = 20                       # 초기 충전율 (%)
while soc < 100:               # 조건이 참인 동안 반복
    soc += 30
print(f"충전 종료 SOC: {soc}%")

for v in [3.71, None, 2.95, 3.70]:
    if v is None:
        continue               # 결측치는 이번 회차만 건너뛴다
    if v < 3.0:
        print(f"{v} V: 심각한 저전압 발견, 검사 중단")
        break                  # 반복 자체를 즉시 끝낸다
    print(f"{v} V: 검사 통과")


# ====================================================================
# 자료구조 4종 / 리스트(list): 순서 있는 측정값 목록 `필수`
# ====================================================================
voltages = [3.71, 3.68, 3.65, 3.42, 3.70]
print(voltages[0], voltages[3], voltages[-1])   # 첫째, 넷째, 마지막
print(voltages[1:4])   # 인덱스 1,2,3 (4는 미포함)
print(voltages[:2])    # 앞에서 2개
print(voltages[3:])    # 네 번째부터 끝까지

voltages = [3.71, 3.68]
voltages.append(3.65)          # 맨 뒤에 추가
voltages[0] = 3.70             # 값 수정
voltages.remove(3.68)          # 값으로 삭제
print(voltages, len(voltages))
print(sorted([3.71, 3.42, 3.68]))                # 새 리스트 반환(원본 보존)
print(sorted([3.71, 3.42, 3.68], reverse=True))  # 내림차순


# ====================================================================
# 자료구조 4종 / 딕셔너리(dict): 키-값으로 묶는 공정 조건 `필수`
# ====================================================================
lot_info = {"lot_id": "LOT-2408-A17", "line": 2, "temp": 45.3}
print(lot_info["lot_id"], lot_info["temp"])
lot_info["operator"] = "김엔지니어"        # 새 항목 추가
lot_info["temp"] = 44.8                   # 값 수정
print(lot_info.get("humidity", "미측정"))  # 없는 키도 안전하게 조회
for key, value in lot_info.items():
    print(f"{key}: {value}")


# ====================================================================
# 자료구조 4종 / 튜플과 셋 `선택 학습(자율 복습)`
# ====================================================================
spec = (3.6, 4.2)              # 튜플: 수정 불가한 규격 (하한, 상한)
print(spec[0], spec[1])

lot_ids = ["LOT-A01", "LOT-A02", "LOT-A01", "LOT-A03", "LOT-A02"]
print(sorted(set(lot_ids)), len(set(lot_ids)))       # 셋: 중복 제거
line1 = {"LOT-A01", "LOT-A02", "LOT-A03"}
line2 = {"LOT-A02", "LOT-A03", "LOT-A04"}
print(sorted(line1 & line2), sorted(line1 - line2))  # 교집합, 차집합


# ====================================================================
# 자료구조 4종 / 자료구조 선택 기준과 종합 예제 `필수`
# ====================================================================
lots = [
    {"lot_id": "LOT-A01", "voltage": 3.71, "result": "합격"},
    {"lot_id": "LOT-A02", "voltage": 3.42, "result": "불합격"},
    {"lot_id": "LOT-A03", "voltage": 3.68, "result": "합격"},
]
pass_count = 0
for lot in lots:
    print(f"{lot['lot_id']} | 전압 {lot['voltage']} V | {lot['result']}")
    if lot["result"] == "합격":
        pass_count += 1
print(f"전체 {len(lots)}개 로트 중 합격 {pass_count}개")


# ====================================================================
# 리스트 컴프리헨션과 유용한 내장함수 / 자주 쓰는 내장함수 `필수`
# ====================================================================
voltages = [3.71, 3.68, 3.65, 3.42, 3.70]
print(len(voltages), sum(voltages), min(voltages), max(voltages))
print(sum(voltages) / len(voltages))   # 평균

lot_ids = ["LOT-A01", "LOT-A02", "LOT-A03"]
voltages = [3.71, 3.42, 3.68]

for i, (lot, v) in enumerate(zip(lot_ids, voltages), start=1):
    print(f"{i}번 {lot}: {v} V")


# ====================================================================
# 리스트 컴프리헨션과 유용한 내장함수 / 리스트 컴프리헨션 `필수`
# ====================================================================
voltages = [3.71, 3.68, 3.65, 3.42, 3.70]
print([v * 1000 for v in voltages])        # 변환: V → mV
print([v for v in voltages if v < 3.6])    # 필터: 규격 미달만

lots = [
    {"lot_id": "LOT-A01", "voltage": 3.71},
    {"lot_id": "LOT-A02", "voltage": 3.42},
    {"lot_id": "LOT-A03", "voltage": 3.68},
    {"lot_id": "LOT-A04", "voltage": 3.55},
]
print([lot["lot_id"] for lot in lots if lot["voltage"] < 3.6])


# ====================================================================
# 리스트 컴프리헨션과 유용한 내장함수 / 실제 공개 데이터 맛보기 — 파일명에서 공정 조건 읽어내기 (12분) `선택 학습(자율 복습)`
# ====================================================================
files = [
    "R1-600um-top-to-bottom-center_frame_569_patch_5.png",
    "R1-700um-top-to-bottom-center_frame_597_patch_3.png",
    "R1-800um-top-to-bottom-center_frame_46_patch_2.png",
    "R1-900um-top-to-bottom-center_frame_251_patch_2.png",
    "R1-1000um-top-to-bottom-center_frame_173_patch_1.png",
    "R1-1100um-top-to-bottom-center-1_frame_220_patch_7.png",
    "R7-700um-middle_frame_489_patch_1.png",
]
print(len(files), "개")

name = files[0]
head, tail = name.split("_frame_")
print(head)
print(tail)

run_txt, gap_txt, position = head.split("-", 2)
print(run_txt, "|", gap_txt, "|", position)
print(int(run_txt[1:]), int(gap_txt[:-2]))

records = []
for name in files:
    head, tail = name.split("_frame_")
    run_txt, gap_txt, position = head.split("-", 2)
    records.append({"run": int(run_txt[1:]),
                    "gap_um": int(gap_txt[:-2]),
                    "position": position,
                    "frame": int(tail.split("_patch_")[0])})

for r in records:
    print(r)

gaps = sorted(set([r["gap_um"] for r in records]))
print("코팅 갭 수준:", gaps)
print("run 7에서 찍힌 갭:", [r["gap_um"] for r in records if r["run"] == 7])

count = {}
for r in records:
    count[r["gap_um"]] = count.get(r["gap_um"], 0) + 1
print(count)


# ====================================================================
# 연습문제 `필수(문제 1~2) / 과제(문제 3~5)` / 해답
# ====================================================================
capacity_initial = 2850   # 초기 용량 (mAh)
capacity_now = 2565       # 현재 용량 (mAh)
print(f"용량 유지율: {capacity_now / capacity_initial * 100:.1f}%")

voltages = [3.71, 3.55, 3.68, 3.42, 3.70, 3.58]
fail = [v for v in voltages if v < 3.6]
print(f"불량 {len(fail)}개 / 전체 {len(voltages)}개")
print(f"불량률: {len(fail) / len(voltages) * 100:.1f}%")


# ====================================================================
# 함수 — 반복되는 계산에 이름을 붙이다 `필수` / def로 함수 정의하기 `필수`
# ====================================================================
def calc_defect_rate(defect_count, total_count):
    """불량률(%)을 계산한다. 전체 개수가 0이면 0.0을 돌려준다."""
    if total_count == 0:
        return 0.0
    return defect_count / total_count * 100

rate = calc_defect_rate(3, 250)
print(f"불량률: {rate:.2f}%")


# ====================================================================
# 함수 — 반복되는 계산에 이름을 붙이다 `필수` / 기본값 인자 — 자주 쓰는 값은 미리 정해 둔다 `필수`
# ====================================================================
def judge_capacity(capacity, lower=3550, upper=3680):
    if capacity < lower:
        return "용량 미달"
    if capacity > upper:
        return "용량 초과"
    return "정상"

print(judge_capacity(3607.4), judge_capacity(3500.0))   # 표준 규격으로 판정
print(judge_capacity(3607.4, lower=3610))               # 하한만 3610으로 강화


# ====================================================================
# 함수 — 반복되는 계산에 이름을 붙이다 `필수` / 여러 값을 한꺼번에 돌려주기 `선택 학습(자율 복습)`
# ====================================================================
def summarize(values):
    return len(values), sum(values) / len(values), min(values), max(values)

caps = [3607.4, 3603.1, 3635.7, 3598.3]
n, avg, lo, hi = summarize(caps)
print(f"개수={n}, 평균={avg:.1f}, 최소={lo}, 최대={hi}")


# ====================================================================
# 모듈과 패키지 — 남의 코드, 나의 코드를 가져다 쓰기 / import 방식 3종 `필수`
# ====================================================================
import math                      # 방식 1: 모듈 전체 → "math.함수" 형태로 사용
from datetime import datetime    # 방식 2: 필요한 것만 → 이름만으로 사용
import pandas as pd              # 방식 3: 별명(alias)을 붙여 짧게

print(math.sqrt(2))
print(datetime(2026, 8, 7))


# ====================================================================
# 모듈과 패키지 — 남의 코드, 나의 코드를 가져다 쓰기 / 자주 쓰는 표준 라이브러리 네 가지 `선택 학습(자율 복습)`
# ====================================================================
import math, statistics, os
from datetime import datetime, timedelta
data = [3607.4, 3603.1, 3635.7, 3598.3]
print(math.floor(3.7), math.ceil(3.2))                                 # 내림, 올림
print(f"평균 {statistics.mean(data):.2f} / 표준편차 {statistics.stdev(data):.2f}")
now = datetime(2026, 8, 7, 14, 30)
print(now.strftime("%Y-%m-%d %H:%M"), "→ 8시간 뒤", (now + timedelta(hours=8)).strftime("%H:%M"))
print(os.path.exists("data/battery_process_data.csv"))                 # 파일이 있는가?


# ====================================================================
# 예외 처리와 파일 입출력 — 죽지 않는 프로그램 만들기 / 대표 사례 1 — 파일이 없을 때 (FileNotFoundError) `필수`
# ====================================================================
try:
    f = open("no_such_file.csv", "r")
except FileNotFoundError:
    print("파일이 없습니다. 경로를 확인하세요.")


# ====================================================================
# 예외 처리와 파일 입출력 — 죽지 않는 프로그램 만들기 / 대표 사례 2 — 형변환 실패 (ValueError) `필수`
# ====================================================================
values = ["3607.4", "3603.1", "오류", "3598.3"]
clean = []
for v in values:
    try:
        clean.append(float(v))
    except ValueError:
        print(f"숫자로 바꿀 수 없는 값 발견: {v!r} → 건너뜀")
print(clean)


# ====================================================================
# 예외 처리와 파일 입출력 — 죽지 않는 프로그램 만들기 / with open — 파일을 안전하게 열고 닫기 `필수`
# ====================================================================
with open("data/battery_process_data.csv", "r", encoding="utf-8-sig") as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i >= 2:
            break


# ====================================================================
# pandas 기초 — 표 데이터를 다루는 표준 도구 / Series와 DataFrame `필수`
# ====================================================================
import pandas as pd

s = pd.Series([3607.4, 3603.1, 3635.7],
              index=["L25000", "L25001", "L25002"], name="최종_용량_mAh")
print(s)


# ====================================================================
# pandas 기초 — 표 데이터를 다루는 표준 도구 / read_csv로 데이터 불러오기 `필수`
# ====================================================================
df = pd.read_csv("data/battery_process_data.csv")
print(df.head())
print(df.shape)


# ====================================================================
# pandas 기초 — 표 데이터를 다루는 표준 도구 / info와 describe — 데이터 건강검진 `필수`
# ====================================================================
df.info()
print(df.describe().round(2))


# ====================================================================
# pandas 기초 — 표 데이터를 다루는 표준 도구 / 열 선택 `필수`
# ====================================================================
print(df["최종_용량_mAh"].head(3))                          # 한 열 → Series
print(df[["Lot_ID", "최종_용량_mAh", "불량_여부"]].head(3))   # 여러 열 → DataFrame


# ====================================================================
# pandas 기초 — 표 데이터를 다루는 표준 도구 / 행 필터링 — 불량 로트만 추출하기 `필수`
# ====================================================================
defect = df[df["불량_여부"] == 1]
print(defect.shape)
print(defect[["Lot_ID", "최종_용량_mAh", "전극_면저항_mOhmcm2"]].head())
print(f"전체 불량률: {df['불량_여부'].mean() * 100:.2f}%")
cond = (df["프레스_압력"] > 48) & (df["불량_여부"] == 1)   # AND는 &, 조건마다 괄호
print(df[cond].shape[0], "개 로트")


# ====================================================================
# pandas 기초 — 표 데이터를 다루는 표준 도구 / 새 열 계산하기 `필수`
# ====================================================================
df["용량_면저항_비"] = df["최종_용량_mAh"] / df["전극_면저항_mOhmcm2"]
print(df[["Lot_ID", "최종_용량_mAh", "전극_면저항_mOhmcm2", "용량_면저항_비"]].head(3).round(3))


# ====================================================================
# pandas 기초 — 표 데이터를 다루는 표준 도구 / groupby — 그룹별 집계 `필수`
# ====================================================================
grp = df.groupby("불량_여부")[["최종_용량_mAh", "전극_면저항_mOhmcm2", "프레스_압력"]].mean()
print(grp.round(2))
agg = df.groupby("불량_여부")["최종_용량_mAh"].agg(["count", "mean", "std", "min", "max"])
print(agg.round(2))


# ====================================================================
# pandas 기초 — 표 데이터를 다루는 표준 도구 / to_csv — 결과 저장 `필수`
# ====================================================================
defect_only = df[df["불량_여부"] == 1]
defect_only.to_csv("data/defect_lots.csv", index=False, encoding="utf-8-sig")
print("저장 완료:", len(defect_only), "행")


# ====================================================================
# pandas 기초 — 표 데이터를 다루는 표준 도구 / .loc와 .iloc — 이름으로, 번호로 고르기 `필수`
# ====================================================================
print(df.iloc[0:3, 0:3])                                             # 번호로: 0~2행, 0~2열
print(df.loc[0:2, ["Lot_ID", "프레스_Gap", "전극_면저항_mOhmcm2"]])     # 이름으로
print(df.loc[df["불량_여부"] == 1, "전극_면저항_mOhmcm2"].mean().round(2))


# ====================================================================
# pandas 기초 — 표 데이터를 다루는 표준 도구 / NumPy 배열 — pandas를 떠받치는 엔진 `필수`
# ====================================================================
import numpy as np

arr = np.array([1428.4, 1444.9, np.nan, 1545.2])
print(arr * 2)                       # 벡터 연산: 원소 전체에 한 번에
print(np.isnan(arr).sum(), "개 결측")
print(np.nanmean(arr).round(2))      # 결측을 빼고 평균
X = df[["프레스_Gap", "전극_면저항_mOhmcm2"]].to_numpy()
print(X.shape, X.mean(axis=0).round(2))   # axis=0은 열 방향(위→아래) 집계
print(np.corrcoef(X[:, 0], X[:, 1])[0, 1].round(4))


# ====================================================================
# SQLite 데이터베이스 연결 — 파일을 넘어 이력 관리로 / sqlite3로 DB 만들고 pandas로 적재하기 `필수`
# ====================================================================
import sqlite3

conn = sqlite3.connect("data/process.db")    # 파일이 없으면 새로 만들어진다
df = pd.read_csv("data/battery_process_data.csv")
df.to_sql("process_data", conn, if_exists="replace", index=False)
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM process_data")
print("적재된 행 수:", cur.fetchone()[0])


# ====================================================================
# SQLite 데이터베이스 연결 — 파일을 넘어 이력 관리로 / SQL 기초 — SELECT, WHERE, GROUP BY `필수`
# ====================================================================
cur.execute("SELECT Lot_ID, 최종_용량_mAh, 불량_여부 FROM process_data LIMIT 3")
print(cur.fetchall())                                          # SELECT + LIMIT
cur.execute("SELECT Lot_ID, 전극_면저항_mOhmcm2 FROM process_data "
            "WHERE 불량_여부 = 1 LIMIT 3")
print(cur.fetchall())                                          # WHERE
cur.execute("SELECT 불량_여부, COUNT(*), ROUND(AVG(최종_용량_mAh), 2), "
            "ROUND(AVG(전극_면저항_mOhmcm2), 2) FROM process_data GROUP BY 불량_여부")
print(cur.fetchall())                                          # GROUP BY + 집계함수


# ====================================================================
# SQLite 데이터베이스 연결 — 파일을 넘어 이력 관리로 / read_sql — SQL 결과를 DataFrame으로 받기 `필수`
# ====================================================================
query = """SELECT 불량_여부, COUNT(*) AS lot_수, AVG(최종_용량_mAh) AS 평균_용량
           FROM process_data GROUP BY 불량_여부"""
result = pd.read_sql(query, conn)
print(result.round(2))
conn.close()   # 작업이 끝나면 연결을 닫는다


# ====================================================================
# 연습문제 `필수(문제 1~2) / 과제(문제 3~5)` / 해답
# ====================================================================
def sheet_resistance_status(r, limit=1550):
    """면저항이 limit을 넘으면 '초과', 아니면 '정상'을 돌려준다."""
    if r > limit:
        return "초과"
    return "정상"

print(sheet_resistance_status(1428.4), sheet_resistance_status(1600.0),
      sheet_resistance_status(1500.0, limit=1450))

raw = ["1774", "1667", "abc", "1825", ""]
rpms = []
for v in raw:
    try:
        rpms.append(int(v))
    except ValueError:
        print(f"변환 실패: {v!r}")
print(f"{rpms} → 평균 RPM: {sum(rpms)/len(rpms):.1f}")


# ====================================================================
# 실전 확장 ① 엑셀 파일 읽기 — 현장 데이터의 실제 모습 (25분) `선택 학습(자율 복습)` / read_excel과 openpyxl `선택 학습(자율 복습)`
# ====================================================================
import pandas as pd

XLSX = "data/wmg/raw/Intermediate measurements during calendering.xlsx"

xls = pd.ExcelFile(XLSX)
print("시트 개수:", len(xls.sheet_names))
print(xls.sheet_names)


# ====================================================================
# 실전 확장 ① 엑셀 파일 읽기 — 현장 데이터의 실제 모습 (25분) `선택 학습(자율 복습)` / 첫 줄이 열 이름이 아니다 — header 인자 `선택 학습(자율 복습)`
# ====================================================================
bad = pd.read_excel(XLSX, sheet_name="Cathode")
print(bad.shape)
print(list(bad.columns)[:6])

grid = pd.read_excel(XLSX, sheet_name="Cathode", header=None)
print("엑셀 격자 그대로:", grid.shape)
print(list(grid.iloc[0, 0:6]))
print(list(grid.iloc[1, 2:6]))

cal = pd.read_excel(XLSX, sheet_name="Cathode", header=1, nrows=18)
print(cal.shape)
print(cal[["No", "Roll temperature (oC)", "Target density (g/cm3)"]].head(3))


# ====================================================================
# 실전 확장 ① 엑셀 파일 읽기 — 현장 데이터의 실제 모습 (25분) `선택 학습(자율 복습)` / 한 시트에 표가 두 개 — nrows 인자 `선택 학습(자율 복습)`
# ====================================================================
print(grid.iloc[18:24, [0, 1]])

print(cal["Roll temperature (oC)"].value_counts().sort_index())
print(cal["Target coating weight (GSM)"].value_counts())


# ====================================================================
# 실전 확장 ① 엑셀 파일 읽기 — 현장 데이터의 실제 모습 (25분) `선택 학습(자율 복습)` / 시트가 19장인 워크북 — 행 번호를 하드코딩하지 마라 `선택 학습(자율 복습)`
# ====================================================================
XL2 = "data/wmg/raw/Half-cell (Cathode) Electrochemical Performance.xlsx"
book = pd.ExcelFile(XL2)
print("시트 개수:", len(book.sheet_names))
print(book.sheet_names[:5], "...")
for name in ["Group1", "Group13", "Group14"]:
    sh = pd.read_excel(XL2, sheet_name=name, header=None)
    print(name, sh.shape)

for name in ["Group1", "Group14"]:
    sh = pd.read_excel(XL2, sheet_name=name, header=None)
    print(name, "44행 →", sh.iloc[44, 1], "/ 첫 셀 값", round(sh.iloc[44, 2], 3))


# ====================================================================
# 실전 확장 ① 엑셀 파일 읽기 — 현장 데이터의 실제 모습 (25분) `선택 학습(자율 복습)` / 남이 계산해 준 평균을 믿지 마라 `선택 학습(자율 복습)`
# ====================================================================
g1 = pd.read_excel(XL2, sheet_name="Group1", header=None)
print(g1.iloc[0, 1:5].tolist())
row = g1.iloc[80]
print(row.iloc[0], "|", row.iloc[1])
print("DD001:", row.iloc[2], "/ DD002:", round(row.iloc[3], 2),
      "/ DD056:", round(row.iloc[4], 2))
print("엑셀 Mean:", round(row.iloc[5], 2), "/ 엑셀 SD:", round(row.iloc[6], 2))

print("빈칸 대신 0이 들어간 DD001을 빼고 다시 평균:",
      round((row.iloc[3] + row.iloc[4]) / 2, 2))


# ====================================================================
# 실전 확장 ② 여러 파일을 한 번에 읽기 (20분) `선택 학습(자율 복습)` / glob — 파일 목록을 코드로 얻는다 `선택 학습(자율 복습)`
# ====================================================================
import glob

paths = sorted(glob.glob("data/*/*.csv"))
print("찾은 CSV 파일:", len(paths), "개")
for p in paths[:4]:
    print("  ", p)


# ====================================================================
# 실전 확장 ② 여러 파일을 한 번에 읽기 (20분) `선택 학습(자율 복습)` / 파일 인벤토리 만들기 `선택 학습(자율 복습)`
# ====================================================================
import os

records = []
for p in paths:
    try:
        d = pd.read_csv(p)
    except Exception as e:
        print("읽기 실패:", p, e)
        continue
    records.append({"폴더": os.path.basename(os.path.dirname(p)),
                    "파일": os.path.basename(p),
                    "행": len(d),
                    "열": d.shape[1],
                    "KB": round(os.path.getsize(p) / 1024)})

inv = pd.DataFrame(records)
print(inv.to_string(index=False))
print("합계:", inv["행"].sum(), "행 /", inv["KB"].sum(), "KB")


# ====================================================================
# 실전 확장 ② 여러 파일을 한 번에 읽기 (20분) `선택 학습(자율 복습)` / 이어붙이기 전에 스키마부터 비교하라 `선택 학습(자율 복습)`
# ====================================================================
wmg_paths = sorted(glob.glob("data/wmg/wmg_*.csv"))
tables = {}
for p in wmg_paths:
    tables[os.path.basename(p)] = pd.read_csv(p)

for name, d in tables.items():
    print(f"{name:24s} {d.shape}")

common = set(tables["wmg_cells_54.csv"].columns)
for d in tables.values():
    common = common & set(d.columns)
print("5개 파일 모두에 있는 열:", sorted(common))

big = pd.concat(tables.values(), ignore_index=True)
print("무작정 이어붙인 결과:", big.shape)
print("전체 칸 중 결측 비율: %.1f %%"
      % (big.isna().sum().sum() / big.size * 100))

cells = tables["wmg_cells_54.csv"]
cyc = tables["wmg_cycling_long.csv"]
merged = pd.merge(cyc, cells[["cell_id", "coat_weight_level", "density_level"]],
                  on="cell_id", how="left")
print(merged.shape)
print(merged.groupby("coat_weight_level")["cap_dis_mah"].mean().round(3))


# ====================================================================
# 실전 확장 ② 여러 파일을 한 번에 읽기 (20분) `선택 학습(자율 복습)` / 파일명만으로는 구분되지 않는다 `선택 학습(자율 복습)`
# ====================================================================
print(inv["파일"].value_counts().head(3))
print(inv.loc[inv["파일"] == "data_dictionary.csv", ["폴더", "행", "열"]])


# ====================================================================
# 실전 확장 ③ 실제 데이터를 DB에 넣고 SQL로 묻기 (15분) `선택 학습(자율 복습)` / 두 테이블 적재 `선택 학습(자율 복습)`
# ====================================================================
import sqlite3

cells = pd.read_csv("data/wmg/wmg_cells_54.csv")
cyc = pd.read_csv("data/wmg/wmg_cycling_long.csv")

conn = sqlite3.connect("data/wmg.db")
cells.to_sql("cells", conn, if_exists="replace", index=False)
cyc.to_sql("cycling", conn, if_exists="replace", index=False)
print(pd.read_sql("SELECT COUNT(*) AS 셀수 FROM cells", conn))
print(pd.read_sql("SELECT COUNT(*) AS 사이클행수 FROM cycling", conn))


# ====================================================================
# 실전 확장 ③ 실제 데이터를 DB에 넣고 SQL로 묻기 (15분) `선택 학습(자율 복습)` / GROUP BY 두 단계 — 공정조건이 성능을 어떻게 바꾸는가 `선택 학습(자율 복습)`
# ====================================================================
q = """SELECT coat_weight_level AS 코팅중량,
              density_level     AS 목표밀도,
              COUNT(*)          AS 셀수,
              ROUND(AVG(grav_dis_5c_mahg), 1) AS 평균_5C용량
       FROM cells
       GROUP BY coat_weight_level, density_level
       ORDER BY coat_weight_level, density_level"""
print(pd.read_sql(q, conn))


# ====================================================================
# 실전 확장 ③ 실제 데이터를 DB에 넣고 SQL로 묻기 (15분) `선택 학습(자율 복습)` / 평균이 거짓말을 하는 자리 `선택 학습(자율 복습)`
# ====================================================================
q_all = "SELECT COUNT(*) AS 셀수, ROUND(AVG(grav_dis_10c_mahg), 2) AS 평균_10C용량 FROM cells"
q_ok = q_all + " WHERE grav_dis_10c_mahg >= 1"
print(pd.read_sql(q_all, conn))
print(pd.read_sql(q_ok, conn))

q_bad = """SELECT cell_id, coat_weight_level, density_level,
                  ROUND(grav_dis_10c_mahg, 4) AS 용량_10C
           FROM cells
           WHERE grav_dis_10c_mahg < 1
           ORDER BY cell_id"""
print(pd.read_sql(q_bad, conn))
conn.close()

