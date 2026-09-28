# -*- coding: utf-8 -*-
"""2장 예제 — 2.5 리스트 컴프리헨션과 유용한 내장함수

교재 출처 : manuscript/20_ch2_파이썬문법.md
실행 방법 : example 폴더에서  python "2장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 02_2-2_변수와_자료형.py, 03_2-3_연산자와_제어문.py, 04_2-4_자료구조_4종.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --------------------------------------------------------------------
# [자주 쓰는 내장함수 `필수`]
# --------------------------------------------------------------------
voltages = [3.71, 3.68, 3.65, 3.42, 3.70]
print(len(voltages), sum(voltages), min(voltages), max(voltages))
print(sum(voltages) / len(voltages))   # 평균

# 교재 실행 결과 ------------------------------------------------------
#   5 18.16 3.42 3.71
#   3.632

# --------------------------------------------------------------------
# [자주 쓰는 내장함수 `필수`]
# --------------------------------------------------------------------
lot_ids = ["LOT-A01", "LOT-A02", "LOT-A03"]
voltages = [3.71, 3.42, 3.68]

for i, (lot, v) in enumerate(zip(lot_ids, voltages), start=1):
    print(f"{i}번 {lot}: {v} V")

# 교재 실행 결과 ------------------------------------------------------
#   1번 LOT-A01: 3.71 V
#   2번 LOT-A02: 3.42 V
#   3번 LOT-A03: 3.68 V

# --------------------------------------------------------------------
# [리스트 컴프리헨션 `필수`]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   [3710.0, 3680.0, 3650.0, 3420.0, 3700.0]
#   [3.42]
#   ['LOT-A02', 'LOT-A04']

# --------------------------------------------------------------------
# [실제 공개 데이터 맛보기 — 파일명에서 공정 조건 읽어내기 (12분) `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   7 개
#   R1-600um-top-to-bottom-center
#   569_patch_5.png

# --------------------------------------------------------------------
# [실제 공개 데이터 맛보기 — 파일명에서 공정 조건 읽어내기 (12분) `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
run_txt, gap_txt, position = head.split("-", 2)
print(run_txt, "|", gap_txt, "|", position)
print(int(run_txt[1:]), int(gap_txt[:-2]))

# 교재 실행 결과 ------------------------------------------------------
#   R1 | 600um | top-to-bottom-center
#   1 600

# --------------------------------------------------------------------
# [실제 공개 데이터 맛보기 — 파일명에서 공정 조건 읽어내기 (12분) `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
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

# 교재 실행 결과 ------------------------------------------------------
#   {'run': 1, 'gap_um': 600, 'position': 'top-to-bottom-center', 'frame': 569}
#   {'run': 1, 'gap_um': 700, 'position': 'top-to-bottom-center', 'frame': 597}
#   {'run': 1, 'gap_um': 800, 'position': 'top-to-bottom-center', 'frame': 46}
#   {'run': 1, 'gap_um': 900, 'position': 'top-to-bottom-center', 'frame': 251}
#   {'run': 1, 'gap_um': 1000, 'position': 'top-to-bottom-center', 'frame': 173}
#   {'run': 1, 'gap_um': 1100, 'position': 'top-to-bottom-center-1', 'frame': 220}
#   {'run': 7, 'gap_um': 700, 'position': 'middle', 'frame': 489}

# --------------------------------------------------------------------
# [실제 공개 데이터 맛보기 — 파일명에서 공정 조건 읽어내기 (12분) `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
gaps = sorted(set([r["gap_um"] for r in records]))
print("코팅 갭 수준:", gaps)
print("run 7에서 찍힌 갭:", [r["gap_um"] for r in records if r["run"] == 7])

count = {}
for r in records:
    count[r["gap_um"]] = count.get(r["gap_um"], 0) + 1
print(count)

# 교재 실행 결과 ------------------------------------------------------
#   코팅 갭 수준: [600, 700, 800, 900, 1000, 1100]
#   run 7에서 찍힌 갭: [700]
#   {600: 1, 700: 2, 800: 1, 900: 1, 1000: 1, 1100: 1}

