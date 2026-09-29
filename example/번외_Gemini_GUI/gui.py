# -*- coding: utf-8 -*-
"""[프롬프트 10~12] tkinter GUI — 파일 열기 → 정제 → 학습 → 성능 → 새 로트 예측 → 중요도 차트."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"      # 한글 축 라벨 (macOS 는 "AppleGothic")
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import data_io, cleaning, model

DEFAULTS = {"믹싱_RPM": 1750, "믹싱_온도": 25.0, "코팅_토출압력": 120.0,
            "건조로_1구간_온도": 110.0, "프레스_압력": 45.0, "프레스_Gap": 95.0}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("배터리 로트 품질 예측기 (Random Forest)")
        self.geometry("1100x680")
        self.df_raw = self.df = None
        self.reg = self.clf = None
        self._build()

    # ---------------- 화면 구성 (프롬프트 10)
    def _build(self):
        top = ttk.Frame(self, padding=8); top.pack(fill="x")
        ttk.Button(top, text="① CSV 열기", command=self.on_open).pack(side="left")
        ttk.Button(top, text="② 정제 + 학습", command=self.on_train).pack(side="left", padx=6)
        ttk.Button(top, text="③ 모델 저장", command=self.on_save).pack(side="left")
        ttk.Button(top, text="모델 불러오기", command=self.on_load).pack(side="left", padx=6)
        self.status = tk.StringVar(value="CSV 파일을 여세요.")
        ttk.Label(top, textvariable=self.status).pack(side="left", padx=12)

        body = ttk.Frame(self, padding=8); body.pack(fill="both", expand=True)
        # 왼쪽: 로그
        left = ttk.LabelFrame(body, text="진행 로그", padding=6); left.pack(side="left", fill="both", expand=True)
        self.log = tk.Text(left, width=60, font=("Consolas", 10)); self.log.pack(fill="both", expand=True)
        # 오른쪽 위: 입력 폼 (프롬프트 11)
        right = ttk.Frame(body); right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        form = ttk.LabelFrame(right, text="새 로트 공정 조건 입력", padding=6); form.pack(fill="x")
        self.inputs = {}
        for i, (name, val) in enumerate(DEFAULTS.items()):
            ttk.Label(form, text=name).grid(row=i // 2, column=(i % 2) * 2, sticky="e", padx=4, pady=3)
            var = tk.StringVar(value=str(val))
            ttk.Entry(form, textvariable=var, width=12).grid(row=i // 2, column=(i % 2) * 2 + 1, padx=4)
            self.inputs[name] = var
        ttk.Label(form, text="불량 판정 임계값").grid(row=3, column=0, sticky="e", padx=4)
        self.threshold = tk.DoubleVar(value=0.3)      # 0.5 면 불량을 거의 못 잡는다(교재 4.13) → 0.3 에서 시작
        ttk.Scale(form, from_=0.05, to=0.95, variable=self.threshold, orient="horizontal").grid(row=3, column=1, columnspan=2, sticky="ew")
        ttk.Button(form, text="④ 예측", command=self.on_predict).grid(row=3, column=3, padx=4)
        self.result = tk.StringVar(value="예측 결과가 여기에 표시됩니다.")
        ttk.Label(form, textvariable=self.result, font=("Malgun Gothic", 11, "bold"), foreground="#065A82").grid(row=4, column=0, columnspan=4, pady=6)
        # 오른쪽 아래: 중요도 차트 (프롬프트 12)
        chart = ttk.LabelFrame(right, text="특성 중요도 (MDI)", padding=6); chart.pack(fill="both", expand=True, pady=(8, 0))
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def write(self, text):
        self.log.insert("end", text + "\n"); self.log.see("end")

    # ---------------- 동작
    def on_open(self, path=None):
        path = path or filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            self.df_raw = data_io.load_csv(path)
        except Exception as e:
            messagebox.showerror("불러오기 실패", str(e)); return
        self.write("[불러오기] " + path)
        self.write("  " + data_io.describe_loaded(self.df_raw))
        self.status.set("불러오기 완료 → ② 정제 + 학습")

    def on_train(self):
        if self.df_raw is None:
            messagebox.showwarning("순서", "먼저 CSV 를 여세요."); return
        self.df = cleaning.clean_data(self.df_raw)
        self.write(cleaning.cleaning_report(self.df_raw, self.df))
        self.write(cleaning.eda_summary(self.df))
        X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = model.split_data(self.df)
        self.reg = model.train_regressor(X_tr, yr_tr)
        self.clf = model.train_classifier(X_tr, yc_tr)
        self.write("[회귀 RF] " + str(model.evaluate_regressor(self.reg, X_te, yr_te)))
        self.write("[분류 RF] " + str(model.evaluate_classifier(self.clf, X_te, yc_te, self.threshold.get())))
        self.draw_importance()
        self.status.set("학습 완료 → ④ 예측 / ③ 저장")

    def draw_importance(self):
        imp = model.feature_importance(self.reg)
        self.ax.clear()
        self.ax.barh(imp.index[::-1], imp.values[::-1], color="#065A82")
        self.ax.set_title("회귀 RF 특성 중요도")
        self.fig.tight_layout(); self.canvas.draw()

    def on_predict(self):
        if self.reg is None:
            messagebox.showwarning("순서", "먼저 학습하거나 모델을 불러오세요."); return
        try:
            values = {k: float(v.get()) for k, v in self.inputs.items()}
        except ValueError:
            messagebox.showerror("입력 오류", "숫자만 입력하세요."); return
        r = model.predict_one(self.reg, self.clf, values, self.threshold.get())
        self.result.set("예측 용량 %.1f mAh | 불량 확률 %.3f | %s" % (r["예측_용량_mAh"], r["불량_확률"], r["판정"]))
        self.write("[예측] " + str(values) + " → " + str(r))

    def on_save(self):
        if self.reg is None:
            messagebox.showwarning("순서", "저장할 모델이 없습니다."); return
        p = model.save_models(self.reg, self.clf)
        self.write("[저장] " + p); self.status.set("저장 완료: " + p)

    def on_load(self, path="battery_rf_models.joblib"):
        try:
            self.reg, self.clf = model.load_models(path)
        except Exception as e:
            messagebox.showerror("불러오기 실패", str(e)); return
        self.write("[불러오기] " + path); self.draw_importance(); self.status.set("모델 불러옴 → ④ 예측")


if __name__ == "__main__":
    App().mainloop()
