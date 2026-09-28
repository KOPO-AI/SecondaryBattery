# -*- coding: utf-8 -*-
"""
UCI SECOM (반도체 공정 센서) 데이터 -> 교재 실습용 CSV 재생성 스크립트

원본 : https://archive.ics.uci.edu/static/public/179/secom.zip
출처 : McCann, M. & Johnston, A. (2008). SECOM [Data set].
       UCI Machine Learning Repository. https://doi.org/10.24432/C54305
License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

원본 구성 (실측)
  secom.data        : 1567행 x 590열, 공백 1칸 구분, 결측 토큰 'NaN', 헤더 없음
  secom_labels.data : 1567행, '<라벨> "<dd/mm/yyyy HH:MM:SS>"' 형식
  secom.names       : 설명 텍스트 (591 features라고 적혀 있으나 실물은 590열)

산출물 (README.md의 '파일 구성' 표와 1:1 대응)
  secom_merged.csv        1567 x 593  timestamp, label, fail + sensor_000..sensor_589
  secom_clean.csv         1567 x 449  위에서 분산0(116) + 결측>50%(28) 센서 제거
  secom_column_report.csv  590 x 6    센서 컬럼별 품질 요약 + 제거 사유

실행
  python prepare.py                 # 원본이 없으면 UCI에서 자동 다운로드
  python prepare.py --src <폴더>    # 이미 압축 해제한 폴더를 지정

설계 노트
  센서 값은 원본 텍스트 토큰을 그대로 옮긴다(float 재포맷 없음). 따라서
  secom_merged.csv를 read_csv로 되읽은 행렬은 원본 secom.data와 정확히 일치한다.
"""

import argparse
import io
import os
import urllib.request
import zipfile

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
HERE = os.path.dirname(os.path.abspath(__file__))
N_ROWS, N_SENSORS = 1567, 590


def fetch(dest_dir):
    """secom.zip을 내려받아 dest_dir에 풀고 경로를 돌려준다. 이미 있으면 그대로 쓴다."""
    os.makedirs(dest_dir, exist_ok=True)
    need = ["secom.data", "secom_labels.data", "secom.names"]
    if all(os.path.exists(os.path.join(dest_dir, f)) for f in need):
        return dest_dir
    req = urllib.request.Request(URL, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(dest_dir)
    return dest_dir


def build(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1) 센서 값: 원본 토큰 보존을 위해 문자열로 읽는다 -------------------
    sens_txt = pd.read_csv(
        os.path.join(src_dir, "secom.data"),
        sep=" ", header=None, dtype=str, keep_default_na=False,
    )
    assert sens_txt.shape == (N_ROWS, N_SENSORS), sens_txt.shape
    names = [f"sensor_{i:03d}" for i in range(N_SENSORS)]
    sens_txt.columns = names
    # 'NaN' 토큰 -> 빈 칸(CSV 표준 결측). read_csv가 자동으로 NaN으로 읽는다.
    sens_txt = sens_txt.replace("NaN", "")

    # ---- 2) 라벨 + 타임스탬프 ---------------------------------------------
    lab = pd.read_csv(
        os.path.join(src_dir, "secom_labels.data"),
        sep=" ", header=None, names=["label", "ts"], quotechar='"',
    )
    assert len(lab) == N_ROWS, len(lab)
    ts = pd.to_datetime(lab["ts"], format="%d/%m/%Y %H:%M:%S")  # 일/월/년 순서 주의
    assert ts.notna().all(), "timestamp 파싱 실패"
    assert set(lab["label"].unique()) == {-1, 1}

    meta = pd.DataFrame({
        "timestamp": ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "label": lab["label"].astype(int),          # 원본 그대로: -1=Pass, 1=Fail
        "fail": (lab["label"] == 1).astype(int),    # 모델 타깃: 1=Fail, 0=Pass
    })

    merged = pd.concat([meta, sens_txt], axis=1)
    p_merged = os.path.join(out_dir, "secom_merged.csv")
    merged.to_csv(p_merged, index=False, encoding="utf-8")

    # ---- 3) 컬럼 품질 보고서 ----------------------------------------------
    num = sens_txt.apply(pd.to_numeric, errors="raise")   # 전부 수치형이어야 한다
    miss = num.isna().mean()
    var = num.var(skipna=True)
    rep = pd.DataFrame({
        "column": names,
        "missing_rate": miss.values,
        "n_missing": num.isna().sum().values,
        "variance": var.values,
        "n_unique": num.nunique(dropna=True).values,
    })
    # 분산0과 결측>50%은 실측상 서로 겹치지 않는다(교집합 0개).
    rep["drop_reason"] = None
    rep.loc[rep["variance"] == 0, "drop_reason"] = "zero_variance"
    rep.loc[rep["missing_rate"] > 0.5, "drop_reason"] = "missing_gt_50pct"
    p_report = os.path.join(out_dir, "secom_column_report.csv")
    rep.to_csv(p_report, index=False, encoding="utf-8")

    # ---- 4) 정리본: 제거 사유가 있는 센서를 뺀 버전 ------------------------
    keep = rep.loc[rep["drop_reason"].isna(), "column"].tolist()
    clean = pd.concat([meta, sens_txt[keep]], axis=1)
    p_clean = os.path.join(out_dir, "secom_clean.csv")
    clean.to_csv(p_clean, index=False, encoding="utf-8")

    # ---- 5) 검증: 되읽어서 실측값을 다시 찍는다 ---------------------------
    chk = pd.read_csv(p_merged)
    s = chk[[c for c in chk.columns if c.startswith("sensor_")]]
    orig = pd.read_csv(os.path.join(src_dir, "secom.data"), sep=" ", header=None)
    import numpy as np
    print("merged shape            :", chk.shape)
    print("clean shape             :", pd.read_csv(p_clean).shape)
    print("report shape            :", rep.shape)
    print("원본과 값 일치           :", np.array_equal(orig.values, s.values, equal_nan=True))
    print("센서 전부 float64        :", bool((s.dtypes == "float64").all()))
    print("전체 결측률 %            :", round(100 * s.isna().mean().mean(), 4))
    print("컬럼 결측률 최대 %       :", round(100 * s.isna().mean().max(), 4),
          "(", s.isna().mean().idxmax(), ")")
    print("결측률 50% 초과 컬럼 수  :", int((s.isna().mean() > 0.5).sum()))
    print("분산 0 컬럼 수           :", int((s.var(skipna=True) == 0).sum()))
    print("라벨 분포                :", chk["label"].value_counts().to_dict())
    print("timestamp 범위           :", chk["timestamp"].min(), "~", chk["timestamp"].max())
    for p in (p_merged, p_clean, p_report):
        print(f"written: {p}  {os.path.getsize(p):,} bytes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "_raw"),
                    help="secom.data 등이 있는 폴더(없으면 UCI에서 다운로드)")
    ap.add_argument("--out", default=HERE, help="CSV 산출 폴더")
    a = ap.parse_args()
    build(fetch(a.src), a.out)
