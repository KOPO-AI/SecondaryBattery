# -*- coding: utf-8 -*-
"""2장 예제 — 2.11 SQLite 데이터베이스 연결 — 파일을 넘어 이력 관리로

교재 출처 : manuscript/21_ch2_pandas_입출력.md
실행 방법 : example 폴더에서  python "2장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 08_2-8_모듈과_패키지.py, 09_2-9_예외_처리와_파일_입출력.py, 10_2-10_pandas_기초.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd

# --------------------------------------------------------------------
# [sqlite3로 DB 만들고 pandas로 적재하기 `필수`]
# --------------------------------------------------------------------
import sqlite3

conn = sqlite3.connect("data/process.db")    # 파일이 없으면 새로 만들어진다
df = pd.read_csv("data/battery_process_data.csv")
df.to_sql("process_data", conn, if_exists="replace", index=False)
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM process_data")
print("적재된 행 수:", cur.fetchone()[0])

# 교재 실행 결과 ------------------------------------------------------
#   적재된 행 수: 1000

# --------------------------------------------------------------------
# [SQL 기초 — SELECT, WHERE, GROUP BY `필수`]
# --------------------------------------------------------------------
cur.execute("SELECT Lot_ID, 최종_용량_mAh, 불량_여부 FROM process_data LIMIT 3")
print(cur.fetchall())                                          # SELECT + LIMIT
cur.execute("SELECT Lot_ID, 전극_면저항_mOhmcm2 FROM process_data "
            "WHERE 불량_여부 = 1 LIMIT 3")
print(cur.fetchall())                                          # WHERE
cur.execute("SELECT 불량_여부, COUNT(*), ROUND(AVG(최종_용량_mAh), 2), "
            "ROUND(AVG(전극_면저항_mOhmcm2), 2) FROM process_data GROUP BY 불량_여부")
print(cur.fetchall())                                          # GROUP BY + 집계함수

# 교재 실행 결과 ------------------------------------------------------
#   [('L25000', 3607.4, 0), ('L25001', 3603.1, 0), ('L25002', 3635.7, 0)]
#   [('L25097', 1674.7), ('L25101', 1557.5), ('L25176', 1688.4)]
#   [(0, 981, 3600.51, 1510.91), (1, 19, 3521.17, 1674.32)]

# --------------------------------------------------------------------
# [read_sql — SQL 결과를 DataFrame으로 받기 `필수`]
# --------------------------------------------------------------------
query = """SELECT 불량_여부, COUNT(*) AS lot_수, AVG(최종_용량_mAh) AS 평균_용량
           FROM process_data GROUP BY 불량_여부"""
result = pd.read_sql(query, conn)
print(result.round(2))
conn.close()   # 작업이 끝나면 연결을 닫는다

# 교재 실행 결과 ------------------------------------------------------
#      불량_여부  lot_수    평균_용량
#   0      0    981  3600.51
#   1      1     19  3521.17

