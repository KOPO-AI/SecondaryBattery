# -*- coding: utf-8 -*-
"""2장 예제 — 2.15 실전 확장 ③ 실제 데이터를 DB에 넣고 SQL로 묻기 (15분) `선택 학습(자율 복습)`

교재 출처 : manuscript/21_ch2_pandas_입출력.md
실행 방법 : example 폴더에서  python "2장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 12_2-12_연습문제.py, 13_2-13_실전_확장_①_엑셀_파일_읽기.py, 14_2-14_실전_확장_②_여러_파일을_한_번에_읽기.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd

# --------------------------------------------------------------------
# [두 테이블 적재 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
import sqlite3

cells = pd.read_csv("data/wmg/wmg_cells_54.csv")
cyc = pd.read_csv("data/wmg/wmg_cycling_long.csv")

conn = sqlite3.connect("data/wmg.db")
cells.to_sql("cells", conn, if_exists="replace", index=False)
cyc.to_sql("cycling", conn, if_exists="replace", index=False)
print(pd.read_sql("SELECT COUNT(*) AS 셀수 FROM cells", conn))
print(pd.read_sql("SELECT COUNT(*) AS 사이클행수 FROM cycling", conn))

# 교재 실행 결과 ------------------------------------------------------
#      셀수
#   0  54
#      사이클행수
#   0   2808

# --------------------------------------------------------------------
# [GROUP BY 두 단계 — 공정조건이 성능을 어떻게 바꾸는가 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
q = """SELECT coat_weight_level AS 코팅중량,
              density_level     AS 목표밀도,
              COUNT(*)          AS 셀수,
              ROUND(AVG(grav_dis_5c_mahg), 1) AS 평균_5C용량
       FROM cells
       GROUP BY coat_weight_level, density_level
       ORDER BY coat_weight_level, density_level"""
print(pd.read_sql(q, conn))

# 교재 실행 결과 ------------------------------------------------------
#     코팅중량 목표밀도  셀수  평균_5C용량
#   0    H    D   9     51.6
#   1    H    M   9     71.7
#   2    H    P   9     29.8
#   3    L    D   9    127.9
#   4    L    M   9    125.9
#   5    L    P   9    118.1

# --------------------------------------------------------------------
# [평균이 거짓말을 하는 자리 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
q_all = "SELECT COUNT(*) AS 셀수, ROUND(AVG(grav_dis_10c_mahg), 2) AS 평균_10C용량 FROM cells"
q_ok = q_all + " WHERE grav_dis_10c_mahg >= 1"
print(pd.read_sql(q_all, conn))
print(pd.read_sql(q_ok, conn))

# 교재 실행 결과 ------------------------------------------------------
#      셀수  평균_10C용량
#   0  54     42.04
#      셀수  평균_10C용량
#   0  47     48.31

# --------------------------------------------------------------------
# [평균이 거짓말을 하는 자리 `선택 학습(자율 복습)`]
# --------------------------------------------------------------------
q_bad = """SELECT cell_id, coat_weight_level, density_level,
                  ROUND(grav_dis_10c_mahg, 4) AS 용량_10C
           FROM cells
           WHERE grav_dis_10c_mahg < 1
           ORDER BY cell_id"""
print(pd.read_sql(q_bad, conn))
conn.close()

# 교재 실행 결과 ------------------------------------------------------
#     cell_id coat_weight_level density_level  용량_10C
#   0   DD027                 H             P  0.0009
#   1   DD028                 H             P  0.0009
#   2   DD029                 H             P  0.0009
#   3   DD030                 H             P  0.0009
#   4   DD031                 H             P  0.0009
#   5   DD051                 H             P  0.0009
#   6   DD059                 H             P  0.0009

