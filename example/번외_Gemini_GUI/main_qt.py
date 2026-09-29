# -*- coding: utf-8 -*-
"""[PyQt5 + Plotly 판, 프롬프트 13] 조립 — 실행: python main_qt.py  (같은 폴더에 data_io.py, cleaning.py, model.py, gui_qt.py)
필요 패키지: pip install pyqt5 PyQtWebEngine plotly"""
import sys
from PyQt5.QtWidgets import QApplication
from gui_qt import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())
