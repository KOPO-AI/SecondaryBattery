# 번외 실습 — Gemini 프롬프트로 조립하는 배터리 품질 예측 GUI (2시간)

3·4장에서 배운 정제 규약(3.7.1)과 Random Forest 회귀·분류(4.4·4.11·4.12)를 **Gemini 에 프롬프트 14개를 차례로 던져** 파이썬 GUI 프로그램 하나로 조립한다.
데이터는 `data/battery_process_data.csv`, 모델은 Random Forest 만 쓴다.

| 파일 | 역할 | 대응 프롬프트 |
|---|---|---|
| `prompts.md` | 수강생용 프롬프트 14개 전문(복사해서 Gemini 에 붙여넣기) | - |
| `data_io.py` | CSV 읽기 + 열 명세 검증 | 2 |
| `cleaning.py` | 정제 4단계(clean_data) + 정제 보고 + EDA 요약 | 3·4 |
| `model.py` | 분할, RF 회귀·분류 학습/평가, 특성 중요도, 저장/불러오기, 새 로트 예측 | 5~9 |
| `gui.py` | tkinter 화면(버튼·로그·입력 폼·임계값 슬라이더·중요도 차트) | 10~12 |
| `main.py` | 실행 진입점 | 13 |
| `gui_qt.py` · `main_qt.py` | **PyQt5 + Plotly 판** 화면(QDoubleSpinBox 폼·QSlider·QWebEngineView 에 Plotly 중요도 막대·Gap-용량 산점도+예측점). data_io/cleaning/model 은 공용 | 10~13 (`prompts_qt.md`) |
| `prompts_qt.md` | PyQt5 + Plotly 판 프롬프트 14개 | - |
| `check_data.py` · `check_model.py` | 체크포인트 ①·② 확인 스크립트 | - |

`data_io.py`~`main.py` 는 **강사용 정답 코드**다. 수강생은 Gemini 가 만들어 준 코드를 같은 파일 이름으로 저장해 조립한다.

실행: 이 폴더에서 `python main.py`(tkinter 판) 또는 `python main_qt.py`(PyQt5 판, `pip install pyqt5 PyQtWebEngine plotly` 필요) → ① CSV 열기(`../data/battery_process_data.csv`) → ② 정제 + 학습 → ④ 예측.

강의 덱: `SecondaryBattary-AI-Simple/번외/2차전지_AI실무_번외_Gemini_GUI실습.pptx`(tkinter 판) · `…_PyQt5+Plotly.pptx`(PyQt5 판)
