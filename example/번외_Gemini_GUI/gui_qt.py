# -*- coding: utf-8 -*-
"""[PyQt5 + Plotly 판, 프롬프트 10~12] 파일 열기 → 정제 → 학습 → 성능 → 새 로트 예측 → Plotly 차트(중요도·산점도).
data_io / cleaning / model 은 tkinter 판과 같은 파일을 그대로 쓴다. 화면만 PyQt5, 차트만 Plotly 로 바뀐다."""
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QLabel, QPlainTextEdit, QDoubleSpinBox,
                             QSlider, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QGridLayout, QGroupBox, QTabWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
import plotly.io as pio

import data_io, cleaning, model

DEFAULTS = {"믹싱_RPM": (1750, 1400, 2100, 10), "믹싱_온도": (25.0, 10, 50, 0.5), "코팅_토출압력": (120.0, 80, 160, 0.5),
            "건조로_1구간_온도": (110.0, 80, 160, 0.5), "프레스_압력": (45.0, 20, 80, 0.5), "프레스_Gap": (95.0, 70, 120, 0.5)}
PRIMARY = "#065A82"


def fig_html(fig):
    """Plotly Figure → QWebEngineView 에 넣을 HTML(plotly.js 는 CDN 에서, 오프라인이면 include_plotlyjs=True)."""
    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), font=dict(family="Malgun Gothic", size=12), template="plotly_white")
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("배터리 로트 품질 예측기 (Random Forest · PyQt5 + Plotly)")
        self.resize(1200, 720)
        self.df_raw = self.df = None
        self.reg = self.clf = None
        self.last_pred = None
        self._build()

    # ---------------- 화면 구성 (프롬프트 10)
    def _build(self):
        root = QWidget(); self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        top = QHBoxLayout(); outer.addLayout(top)
        self.btn_open = QPushButton("① CSV 열기"); self.btn_train = QPushButton("② 정제 + 학습")
        self.btn_save = QPushButton("③ 모델 저장"); self.btn_load = QPushButton("모델 불러오기")
        for b in (self.btn_open, self.btn_train, self.btn_save, self.btn_load):
            top.addWidget(b)
        self.status = QLabel("CSV 파일을 여세요."); top.addWidget(self.status, 1)
        self.btn_open.clicked.connect(lambda: self.on_open()); self.btn_train.clicked.connect(self.on_train)
        self.btn_save.clicked.connect(self.on_save); self.btn_load.clicked.connect(lambda: self.on_load())

        body = QHBoxLayout(); outer.addLayout(body, 1)
        log_box = QGroupBox("진행 로그"); lv = QVBoxLayout(log_box)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        lv.addWidget(self.log); body.addWidget(log_box, 5)

        right = QVBoxLayout(); body.addLayout(right, 6)
        # 입력 폼 (프롬프트 11)
        form = QGroupBox("새 로트 공정 조건 입력"); g = QGridLayout(form)
        self.inputs = {}
        for i, (name, (val, lo, hi, step)) in enumerate(DEFAULTS.items()):
            g.addWidget(QLabel(name), i // 2, (i % 2) * 2)
            sb = QDoubleSpinBox(); sb.setRange(lo, hi); sb.setSingleStep(step); sb.setDecimals(1); sb.setValue(val)
            g.addWidget(sb, i // 2, (i % 2) * 2 + 1); self.inputs[name] = sb
        g.addWidget(QLabel("불량 판정 임계값"), 3, 0)
        self.threshold = QSlider(Qt.Horizontal); self.threshold.setRange(5, 95); self.threshold.setValue(30)
        self.th_label = QLabel("0.30"); self.threshold.valueChanged.connect(lambda v: self.th_label.setText("%.2f" % (v / 100)))
        g.addWidget(self.threshold, 3, 1, 1, 2); g.addWidget(self.th_label, 3, 3)
        self.btn_predict = QPushButton("④ 예측"); self.btn_predict.clicked.connect(self.on_predict); g.addWidget(self.btn_predict, 4, 3)
        self.result = QLabel("예측 결과가 여기에 표시됩니다."); self.result.setStyleSheet("font-weight: bold; color: %s; font-size: 12pt;" % PRIMARY)
        g.addWidget(self.result, 4, 0, 1, 3); right.addWidget(form)
        # Plotly 차트 탭 (프롬프트 12)
        self.tabs = QTabWidget(); self.web_imp = QWebEngineView(); self.web_sc = QWebEngineView()
        self.tabs.addTab(self.web_imp, "특성 중요도"); self.tabs.addTab(self.web_sc, "Gap-용량 산점도 + 예측점")
        right.addWidget(self.tabs, 1)

    def th(self):
        return self.threshold.value() / 100

    def write(self, text):
        self.log.appendPlainText(text)

    # ---------------- 동작
    def on_open(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "CSV 선택", "", "CSV (*.csv)")
        if not path:
            return
        try:
            self.df_raw = data_io.load_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "불러오기 실패", str(e)); return
        self.write("[불러오기] " + path); self.write("  " + data_io.describe_loaded(self.df_raw))
        self.status.setText("불러오기 완료 → ② 정제 + 학습")

    def on_train(self):
        if self.df_raw is None:
            QMessageBox.warning(self, "순서", "먼저 CSV 를 여세요."); return
        self.df = cleaning.clean_data(self.df_raw)
        self.write(cleaning.cleaning_report(self.df_raw, self.df)); self.write(cleaning.eda_summary(self.df))
        X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = model.split_data(self.df)
        self.reg = model.train_regressor(X_tr, yr_tr); self.clf = model.train_classifier(X_tr, yc_tr)
        self.write("[회귀 RF] " + str(model.evaluate_regressor(self.reg, X_te, yr_te)))
        self.write("[분류 RF] " + str(model.evaluate_classifier(self.clf, X_te, yc_te, self.th())))
        self.draw_charts(); self.status.setText("학습 완료 → ④ 예측 / ③ 저장")

    def draw_charts(self):
        imp = model.feature_importance(self.reg)
        fig = go.Figure(go.Bar(x=imp.values[::-1], y=imp.index[::-1], orientation="h", marker_color=PRIMARY,
                               text=[f"{v:.3f}" for v in imp.values[::-1]], textposition="outside"))
        fig.update_layout(title="회귀 RF 특성 중요도 (MDI)", xaxis_title="중요도", xaxis_range=[0, max(imp.values) * 1.2])
        self.web_imp.setHtml(fig_html(fig))
        if self.df is not None:
            d = self.df; ok = d["불량_여부"] == 0
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=d.loc[ok, "프레스_Gap"], y=d.loc[ok, "최종_용량_mAh"], mode="markers", name="정상",
                                      marker=dict(color="#8EA9C1", size=6, opacity=0.6), hovertext=d.loc[ok, "Lot_ID"]))
            fig2.add_trace(go.Scatter(x=d.loc[~ok, "프레스_Gap"], y=d.loc[~ok, "최종_용량_mAh"], mode="markers", name="불량",
                                      marker=dict(color="#C0392B", size=9, symbol="x"), hovertext=d.loc[~ok, "Lot_ID"]))
            if self.last_pred:
                v, r = self.last_pred
                fig2.add_trace(go.Scatter(x=[v["프레스_Gap"]], y=[r["예측_용량_mAh"]], mode="markers+text", name="새 로트 예측",
                                          marker=dict(color="#F2A007", size=16, symbol="star"),
                                          text=[f"{r['예측_용량_mAh']} mAh / p={r['불량_확률']}"], textposition="top center"))
            fig2.update_layout(title="프레스_Gap vs 최종 용량 (정제 후 1,000 Lot)", xaxis_title="프레스_Gap (µm)", yaxis_title="최종_용량_mAh")
            self.web_sc.setHtml(fig_html(fig2))

    def on_predict(self):
        if self.reg is None:
            QMessageBox.warning(self, "순서", "먼저 학습하거나 모델을 불러오세요."); return
        values = {k: sb.value() for k, sb in self.inputs.items()}
        r = model.predict_one(self.reg, self.clf, values, self.th())
        self.last_pred = (values, r)
        self.result.setText("예측 용량 %.1f mAh | 불량 확률 %.3f | %s" % (r["예측_용량_mAh"], r["불량_확률"], r["판정"]))
        self.write("[예측] " + str(values) + " → " + str(r))
        if self.df is not None:
            self.draw_charts(); self.tabs.setCurrentIndex(1)

    def on_save(self):
        if self.reg is None:
            QMessageBox.warning(self, "순서", "저장할 모델이 없습니다."); return
        p = model.save_models(self.reg, self.clf); self.write("[저장] " + p); self.status.setText("저장 완료: " + p)

    def on_load(self, path="battery_rf_models.joblib"):
        try:
            self.reg, self.clf = model.load_models(path)
        except Exception as e:
            QMessageBox.critical(self, "불러오기 실패", str(e)); return
        self.write("[불러오기] " + path); self.draw_charts(); self.status.setText("모델 불러옴 → ④ 예측")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())
