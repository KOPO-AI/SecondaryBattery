# -*- coding: utf-8 -*-
"""체크포인트 ② — 모델 단계 확인.  실행: python check_model.py  (같은 폴더에 data_io.py, cleaning.py, model.py)"""
import data_io, cleaning, model

df = cleaning.clean_data(data_io.load_csv("../data/battery_process_data.csv"))
X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = model.split_data(df)
print("분할:", X_tr.shape, X_te.shape, "테스트 불량", int(yc_te.sum()))
reg = model.train_regressor(X_tr, yr_tr)
clf = model.train_classifier(X_tr, yc_tr)
print("회귀 RF:", model.evaluate_regressor(reg, X_te, yr_te))
for th in (0.5, 0.3, 0.1):
    print("분류 RF 임계값 %.1f:" % th, model.evaluate_classifier(clf, X_te, yc_te, th))
print("특성 중요도:\n" + model.feature_importance(reg).round(3).to_string())
base = {"믹싱_RPM": 1750, "믹싱_온도": 25.0, "코팅_토출압력": 120.0, "건조로_1구간_온도": 110.0, "프레스_압력": 45.0, "프레스_Gap": 95.0}
print("새 로트(기본값):", model.predict_one(reg, clf, base, 0.3))
print("새 로트(Gap 110, RPM 1900):", model.predict_one(reg, clf, {**base, "프레스_Gap": 110, "믹싱_RPM": 1900}, 0.3))
print("\n[기대값] R² 0.75~0.80, RMSE 15~17 / 임계값을 내릴수록 검출 증가 / 중요도 1위 프레스_Gap")
