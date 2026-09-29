# -*- coding: utf-8 -*-
"""[프롬프트 5~9] Random Forest 회귀·분류, 특성 중요도, 저장/불러오기, 새 로트 예측 — 4장에서 배운 내용."""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

FEATS = ["믹싱_RPM", "믹싱_온도", "코팅_토출압력", "건조로_1구간_온도", "프레스_압력", "프레스_Gap"]
TARGET_REG = "최종_용량_mAh"
TARGET_CLF = "불량_여부"
RANDOM_STATE = 42


def split_data(df):
    """학습 80% / 테스트 20%. 분류 타깃은 stratify 로 불량 비율을 유지한다(교재 4.2·4.8)."""
    X = df[FEATS]
    y_reg = df[TARGET_REG]
    y_clf = df[TARGET_CLF]
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf)
    return X_train, X_test, yr_train, yr_test, yc_train, yc_test


def train_regressor(X_train, y_train):
    """용량(mAh) 예측 RF 회귀 (교재 4.4 기본 설정)."""
    reg = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    reg.fit(X_train, y_train)
    return reg


def train_classifier(X_train, y_train):
    """불량 여부 RF 분류. 불량이 2% 뿐이라 class_weight='balanced' (교재 4.12)."""
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


def evaluate_regressor(reg, X_test, y_test):
    pred = reg.predict(X_test)
    return {"R2": round(r2_score(y_test, pred), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2)}


def evaluate_classifier(clf, X_test, y_test, threshold=0.5):
    """predict 가 아니라 predict_proba + 임계값으로 판정한다(교재 4.13)."""
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {"정확도": round(accuracy_score(y_test, pred), 3),
            "정밀도": round(precision_score(y_test, pred, zero_division=0), 3),
            "재현율": round(recall_score(y_test, pred, zero_division=0), 3),
            "F1": round(f1_score(y_test, pred, zero_division=0), 3),
            "테스트 불량": int(y_test.sum()), "검출": int(pred[y_test.values == 1].sum())}


def feature_importance(model):
    """MDI 특성 중요도를 내림차순 Series 로 (교재 5.9)."""
    return pd.Series(model.feature_importances_, index=FEATS).sort_values(ascending=False)


def save_models(reg, clf, path="battery_rf_models.joblib"):
    joblib.dump({"reg": reg, "clf": clf, "feats": FEATS}, path)
    return path


def load_models(path="battery_rf_models.joblib"):
    bundle = joblib.load(path)
    return bundle["reg"], bundle["clf"]


def predict_one(reg, clf, values, threshold=0.5):
    """values: {열 이름: 값} 6개. 예측 용량, 불량 확률, 판정을 돌려준다(교재 5.13 새 로트 예측 함수)."""
    row = pd.DataFrame([[float(values[f]) for f in FEATS]], columns=FEATS)
    cap = float(reg.predict(row)[0])
    p_def = float(clf.predict_proba(row)[0, 1])
    return {"예측_용량_mAh": round(cap, 1), "불량_확률": round(p_def, 3),
            "판정": "불량 의심" if p_def >= threshold else "정상"}
