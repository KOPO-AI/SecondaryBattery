# -*- coding: utf-8 -*-
"""6장 예제 — 6.22 실전 시계열 ④ 딥러닝 맛보기 — LSTM 오토인코더로 같은 로그를 감시하다 (실습 45분) `선택 학습(자율 복습)`

교재 출처 : manuscript/61_ch6_최적화.md
포함 소절 : 왜 딥러닝인가, 그리고 왜 마지막에 배우는가 / 창(window) 만들기 — 한 행이 아니라 30분을 한 장으로 / 모델 — 누르고(encoder) 펼친다(decoder) / 채점 — 복원 오차와 정답지 대조 / 그래서 딥러닝을 써야 하는가 — 이 데이터가 주는 정직한 답
실행 방법 : example 폴더에서  python "6장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다). tensorflow 필요.
"""

# --------------------------------------------------------------------
# [창(window) 만들기 — 한 행이 아니라 30분을 한 장으로]
# --------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

log = pd.read_csv("data/plc/plc_line_log.csv", parse_dates=["timestamp"], index_col="timestamp")
events = pd.read_csv("data/plc/plc_events.csv")
운전중 = log["EC_C_LINE_SPEED_PV"] > 1
코터SV = [c for c in log.columns if c.startswith("EC_C_") and c.endswith("_SV")]
안정화중 = (log[코터SV].diff().abs().sum(axis=1) > 0).rolling("15min").max().fillna(0) > 0
X = pd.DataFrame({
    "속도_편차":   log["EC_C_LINE_SPEED_PV"] - log["EC_C_LINE_SPEED_SV"],
    "DR1_편차":    log["EC_C_DR1_TEMP_PV"] - log["EC_C_DR1_TEMP_SV"],
    "DR2_편차":    log["EC_C_DR2_TEMP_PV"] - log["EC_C_DR2_TEMP_SV"],
    "DR3_편차":    log["EC_C_DR3_TEMP_PV"] - log["EC_C_DR3_TEMP_SV"],
    "DR4_편차":    log["EC_C_DR4_TEMP_PV"] - log["EC_C_DR4_TEMP_SV"],
    "언와인더_편차": log["EC_C_UNWIN_TEN_PV"] - log["EC_C_UNWIN_TEN_SV"],
    "리와인더_편차": log["EC_C_REWIN_TEN_PV"] - log["EC_C_REWIN_TEN_SV"],
    "상부롤_온도":  log["EC_P_UPPER_ROLL_TEMP_PV"],
    "하부롤_온도":  log["EC_P_LOWER_ROLL_TEMP_PV"],
})[운전중 & ~안정화중]                                   # 6.20절과 동일한 입력
참조 = X.loc[:"2026-09-02 23:59"]
Z = ((X - 참조.mean()) / 참조.std()).astype("float32")   # 참조 기준 표준화

W = 30                                                   # 창 길이(분)
def 창_자르기(Z, W):
    """연속한 W행씩 잘라 (창 수, W, 특징 수) 배열로 — 각 창은 마지막 분의 시각으로 대표한다"""
    arr = Z.to_numpy()
    창 = np.stack([arr[i - W:i] for i in range(W, len(arr) + 1)])
    return 창, Z.index[W - 1:]

창_전체, 시각 = 창_자르기(Z, W)
참조끝 = int((시각 <= pd.Timestamp("2026-09-02 23:59")).sum())
창_참조 = 창_전체[:참조끝]
print("전체 창:", 창_전체.shape, "| 참조 창:", 창_참조.shape)

# 교재 실행 결과 ------------------------------------------------------
#   전체 창: (9756, 30, 9) | 참조 창: (2771, 30, 9)

# --------------------------------------------------------------------
# [모델 — 누르고(encoder) 펼친다(decoder)]
# --------------------------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"                # TensorFlow의 정보성 메시지를 줄인다(경고는 무해하다)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
keras.utils.set_random_seed(42)                          # 재현을 위한 시드(완전히 같지는 않을 수 있다)

n_feat = 창_참조.shape[2]
model = keras.Sequential([
    layers.Input(shape=(W, n_feat)),
    layers.LSTM(32),                                     # encoder: 30분 × 9특징 → 벡터 32개
    layers.RepeatVector(W),                              # 그 벡터를 30번 복사해 시간 축을 되살린다
    layers.LSTM(32, return_sequences=True),              # decoder
    layers.TimeDistributed(layers.Dense(n_feat)),        # 매 분마다 9특징을 복원
])
model.compile(optimizer="adam", loss="mse")
print("학습 파라미터 수:", model.count_params())

hist = model.fit(창_참조, 창_참조, epochs=15, batch_size=64, validation_split=0.1, shuffle=True, verbose=0)
print("학습 손실(MSE) 처음→끝:", round(hist.history["loss"][0], 4), "→", round(hist.history["loss"][-1], 4),
      "| 검증 손실 끝:", round(hist.history["val_loss"][-1], 4))

# 교재 실행 결과 ------------------------------------------------------
#   학습 파라미터 수: 13993
#   학습 손실(MSE) 처음→끝: 0.853 → 0.4423 | 검증 손실 끝: 0.7218

# --------------------------------------------------------------------
# [채점 — 복원 오차와 정답지 대조]
# --------------------------------------------------------------------
복원 = model.predict(창_전체, verbose=0)
오차 = pd.Series(((창_전체 - 복원) ** 2).mean(axis=(1, 2)), index=시각)      # 창 하나의 평균 제곱 오차
임계 = np.percentile(오차.iloc[:참조끝], 99.9)                                # 참조 기간 오차의 99.9 백분위
초과 = 오차 > 임계
연속3 = 초과.rolling(3).sum() >= 3
print(f"임계(참조 99.9%): {임계:.3f} | 참조 기간 초과: {int(초과.iloc[:참조끝].sum())}분 | 전체 초과 {int(초과.sum())}분 → 연속 3분 {int(연속3.sum())}분")

def 사건별_검출(검출, events):
    """6.19.5절과 동일"""
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

표4 = 사건별_검출(연속3, events)
print(표4.to_string(index=False)); print("사건 밖 검출(오탐):", 표4.attrs["오탐(분)"], "분")

특징오차 = pd.DataFrame(((창_전체 - 복원) ** 2)[:, -1, :], index=시각, columns=X.columns)   # 창의 마지막 분, 특징별 오차
주범 = 특징오차.idxmax(axis=1)
for 사건 in ["C1", "C4", "C2", "C3", "P1"]:
    e = events.set_index("사건").loc[사건]
    구간 = 연속3.loc[e["시작"]:e["종료"]]
    print(사건, "→ 주범 태그:", 주범[구간[구간].index].value_counts().head(1).to_dict())

# 교재 실행 결과 ------------------------------------------------------
#   임계(참조 99.9%): 1.145 | 참조 기간 초과: 3분 | 전체 초과 1196분 → 연속 3분 1168분
#   사건                      태그  구간(분)  검출(분)  첫 검출 지연(분)
#   C1        EC_C_DR2_TEMP_PV      5      3         2.0
#   C4       EC_C_UNWIN_TEN_PV     31     29         2.0
#   M1           EC_M_MMF_A_PV     91      0         NaN
#   C2        EC_C_DR2_TEMP_PV      3      0         NaN
#   C3        EC_C_DR3_TEMP_PV    361    165       141.0
#   P1 EC_P_UPPER_ROLL_TEMP_PV    840    802        38.0
#   사건 밖 검출(오탐): 169 분
#   C1 → 주범 태그: {'DR2_편차': 3}
#   C4 → 주범 태그: {'언와인더_편차': 29}
#   C2 → 주범 태그: {}
#   C3 → 주범 태그: {'DR3_편차': 141}
#   P1 → 주범 태그: {'상부롤_온도': 802}

# --------------------------------------------------------------------
# [채점 — 복원 오차와 정답지 대조]
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))
오차.plot(ax=ax, lw=0.6, color="gray", label="복원 오차(30분 창 MSE)")
오차[연속3].plot(ax=ax, style="r.", ms=5, label="검출(연속 3분)")
ax.axhline(임계, color="r", ls=":", label="임계(참조 99.9%)")
for _, e in events[events["이상여부"] == "Y"].iterrows():
    ax.axvspan(pd.Timestamp(e["시작"]), pd.Timestamp(e["종료"]), color="orange", alpha=0.25)
    ax.text(pd.Timestamp(e["시작"]), 오차.max() * 0.98, e["사건"], fontsize=9, va="top")
ax.set_yscale("log"); ax.set_ylabel("복원 오차 (log)"); ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig("fig_6_44_plc_lstm_ae.png", dpi=150); plt.close()

