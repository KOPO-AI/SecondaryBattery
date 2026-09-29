# Gemini 프롬프트 모음 - 배터리 로트 품질 예측 GUI (14개)

덱 『번외 실습. Gemini 프롬프트로 조립하는 배터리 품질 예측 GUI』와 같은 원문이다. 순서대로 **한 대화 창에서** 붙여넣는다.
각 프롬프트 뒤의 '확인' 이 맞아야 다음으로 간다. 데이터: `data/battery_process_data.csv`, 모델: Random Forest.


## 1부 데이터

### 프롬프트 1. 역할 부여와 프로젝트 요약  (파일: (없음), 약 5분)

```text
너는 파이썬·pandas·scikit-learn 강사이자 시니어 개발자다. 앞으로 내가 요청하는 순서대로 "배터리 로트 품질 예측기" 데스크톱 프로그램을 함께 만든다.
- 데이터: battery_process_data.csv (1,000행). 열: Lot_ID, 믹싱_RPM, 믹싱_온도, 코팅_토출압력, 건조로_1구간_온도, 프레스_압력, 프레스_Gap, 최종_용량_mAh, 전극_면저항_mOhmcm2, 불량_여부(0/1)
- 목표: 공정 조건 6개(믹싱_RPM~프레스_Gap)로 최종_용량_mAh 를 예측하는 회귀 모델과 불량_여부 를 예측하는 분류 모델을 Random Forest 로 만들고, tkinter GUI 에서 쓴다.
- 규칙: 파일은 data_io.py / cleaning.py / model.py / gui.py / main.py 5개로 나눈다. 표준 라이브러리 + pandas, numpy, scikit-learn, matplotlib, joblib 만 쓴다. 코드에는 한국어 주석을 달고, 함수마다 한 줄 docstring 을 쓴다. 내가 요청하지 않은 기능은 넣지 않는다.
이해했으면 "준비됨" 이라고만 답해라.
```

- 기대 결과: Gemini 가 "준비됨" 만 답한다. / 긴 설명·코드를 먼저 내놓으면 "코드는 아직 만들지 마" 로 제지한다.
- 확인: 대화 창 하나를 끝까지 유지한다(맥락 유지). / 열 이름을 한 글자도 바꾸지 않았는지 확인.

### 프롬프트 2. data_io.py - CSV 읽기와 열 검증  (파일: data_io.py, 약 5분)

```text
data_io.py 를 만들어라.
1) EXPECTED_COLUMNS 상수: 위에서 알려 준 열 10개를 순서대로 담은 리스트.
2) load_csv(path): pandas 로 CSV 를 읽되 encoding="utf-8-sig" 를 쓴다. EXPECTED_COLUMNS 중 없는 열이 있으면 ValueError("필수 열이 없습니다: ...") 를 낸다. DataFrame 을 돌려준다.
3) describe_loaded(df): "행 N개, 열 N개, 결측 N칸, 불량 N건" 형식의 문자열 하나를 돌려준다.
파일 전체 코드만 출력해라.
```

- 기대 결과: 함수 2개 + 상수 1개. 파일 40줄 안팎. / `python -c "import data_io; print(data_io.describe_loaded(data_io.load_csv('../data/battery_process_data.csv')))"`
- 확인: 출력: 행 1000개, 열 10개, 결측 5칸, 불량 19건 / 열 이름을 하나 지운 CSV 로 ValueError 가 나는지 확인

### 프롬프트 3. cleaning.py - 교재 3.7.1 정제 규약  (파일: cleaning.py, 약 10분)

```text
cleaning.py 를 만들어라. 교재의 정제 규약을 그대로 옮긴다.
1) PHYSICAL_RULES = {"믹싱_온도": (10, 50), "건조로_1구간_온도": (80, 160), "프레스_압력": (20, 80)}
2) clean_data(df): 원본을 바꾸지 말고 복사본에 다음 4단계를 순서대로 적용해 돌려준다.
   ① 건조로_1구간_온도 의 9999 를 NaN 으로
   ② PHYSICAL_RULES 범위 밖 값을 NaN 으로 (경계값은 정상)
   ③ 수치 열의 NaN 을 그 열의 중앙값으로 대체
   ④ Lot_ID 기준 중복 행 제거 후 인덱스 재설정
3) cleaning_report(before, after): 행 수·결측 칸 수의 전후, 9999 건수, 규칙별 위반 건수를 여러 줄 문자열로 돌려준다.
numpy 는 np, pandas 는 pd 로 import 한다. 파일 전체 코드만 출력해라.
```

- 기대 결과: clean_data 안에 fillna(median) 이 select_dtypes('number') 열에만 적용된다. / 4단계 순서가 바뀌면 결과가 달라진다(9999 를 먼저 NaN 으로 바꿔야 중앙값이 오염되지 않는다).
- 확인: 정제 후: 1000행, 결측 0칸, 9999 5건, 건조로 범위 위반 5건, 나머지 0건 / 교재 3.7.1 과 숫자가 같아야 한다

### 프롬프트 4. cleaning.py - EDA 요약 함수 추가  (파일: cleaning.py, 약 5분)

```text
cleaning.py 에 eda_summary(df, target="최종_용량_mAh") 함수를 추가해라.
- 수치 열의 describe() 에서 mean, std, min, max 만 뽑아 소수 둘째 자리로 반올림한 표
- target 과 나머지 수치 열의 상관계수를 절댓값 기준 내림차순으로 정렬해 상위 3개(소수 셋째 자리)
두 결과를 "[기술통계]", "[최종_용량_mAh 와의 상관 상위 3]" 제목과 함께 한 문자열로 이어서 돌려준다. 추가된 함수만 출력해라.
```

- 기대 결과: 기존 함수는 건드리지 않고 함수 하나만 추가.
- 확인: 상관 1위 프레스_Gap ≈ -0.870, 2위 전극_면저항 ≈ -0.732 (교재 3.11 과 동일)


## 2부 모델

### 프롬프트 5. model.py - 특성·타깃 정의와 분할  (파일: model.py, 약 5분)

```text
model.py 를 시작한다.
1) FEATS = ["믹싱_RPM", "믹싱_온도", "코팅_토출압력", "건조로_1구간_온도", "프레스_압력", "프레스_Gap"], TARGET_REG = "최종_용량_mAh", TARGET_CLF = "불량_여부", RANDOM_STATE = 42
2) split_data(df): X = df[FEATS], 회귀 타깃 y_reg, 분류 타깃 y_clf 를 sklearn train_test_split 로 한 번에 나눈다. test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf. (X_train, X_test, yr_train, yr_test, yc_train, yc_test) 순서로 돌려준다.
여기까지의 model.py 전체 코드만 출력해라.
```

- 기대 결과: train_test_split 에 배열 3개를 넣어 6개를 받는 형태. / stratify 가 빠지면 테스트셋 불량이 0건이 될 수 있다(교재 4.8).
- 확인: X_train 800행, X_test 200행, 테스트 불량 4건

### 프롬프트 6. model.py - Random Forest 회귀  (파일: model.py, 약 10분)

```text
model.py 에 추가해라.
1) train_regressor(X_train, y_train): RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1) 을 학습해 돌려준다.
2) evaluate_regressor(reg, X_test, y_test): {"R2": r2_score 소수 넷째 자리, "RMSE": sqrt(mean_squared_error) 소수 둘째 자리} 를 돌려준다.
추가된 함수와 필요한 import 만 출력해라.
```

- 기대 결과: RMSE 는 mean_squared_error 에 squared=False 대신 np.sqrt 로(버전 호환).
- 확인: R² 0.75~0.79, RMSE 15~17 mAh 범위면 정상(교재 4.4 RF 기본: R² 0.7535, RMSE 15.99; 버전에 따라 소수점이 다르다)

### 프롬프트 7. model.py - Random Forest 분류와 임계값  (파일: model.py, 약 10분)

```text
model.py 에 추가해라.
1) train_classifier(X_train, y_train): RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1) 학습.
2) evaluate_classifier(clf, X_test, y_test, threshold=0.5): predict 를 쓰지 말고 predict_proba[:, 1] 이 threshold 이상이면 1 로 판정한다. {"정확도", "정밀도", "재현율", "F1"(소수 셋째 자리), "테스트 불량"(실제 불량 수), "검출"(그중 잡은 수)} 를 돌려준다. precision/recall/f1 에는 zero_division=0 을 준다.
추가된 함수와 import 만 출력해라.
```

- 기대 결과: threshold 인자가 있어야 GUI 슬라이더와 연결된다.
- 확인: threshold 0.5: 검출 0/4 → 0.3: 2/4 → 0.1: 3/4 처럼 낮출수록 재현율이 오른다(교재 4.13 임계값 이동)

### 프롬프트 8. model.py - 특성 중요도  (파일: model.py, 약 5분)

```text
model.py 에 feature_importance(model) 을 추가해라. model.feature_importances_ 를 FEATS 를 인덱스로 하는 pandas Series 로 만들고 내림차순 정렬해 돌려준다. 함수만 출력해라.
```

- 기대 결과: 3줄짜리 함수.
- 확인: 회귀 RF: 프레스_Gap ≈ 0.79 로 압도적 1위, 코팅_토출압력 2위 (교재 5.9 와 같은 순위)

### 프롬프트 9. model.py - 저장·불러오기·새 로트 예측  (파일: model.py, 약 10분)

```text
model.py 에 추가해라.
1) save_models(reg, clf, path="battery_rf_models.joblib"): {"reg", "clf", "feats"} 딕셔너리를 joblib.dump 로 저장하고 path 를 돌려준다.
2) load_models(path="battery_rf_models.joblib"): (reg, clf) 를 돌려준다.
3) predict_one(reg, clf, values, threshold=0.5): values 는 {열 이름: 값} 딕셔너리(FEATS 6개). FEATS 순서로 1행 DataFrame 을 만들어 {"예측_용량_mAh": 소수 첫째 자리, "불량_확률": 소수 셋째 자리, "판정": 확률이 threshold 이상이면 "불량 의심" 아니면 "정상"} 을 돌려준다.
추가된 함수와 import 만 출력해라.
```

- 기대 결과: predict 에 넘기는 것은 리스트가 아니라 열 이름이 있는 DataFrame 이어야 경고가 없다.
- 확인: 기본값(RPM 1750, 온도 25, 토출압력 120, 건조 110, 압력 45, Gap 95) → 용량 약 3590 mAh, 정상 / Gap 110, RPM 1900 → 용량 약 3510 mAh, 불량 확률 상승


## 3부 GUI

### 프롬프트 10. gui.py - tkinter 뼈대: 버튼·로그·상태 표시  (파일: gui.py, 약 15분)

```text
gui.py 를 만들어라. tkinter 와 ttk 만 쓴다.
1) class App(tk.Tk): 제목 "배터리 로트 품질 예측기 (Random Forest)", 크기 1100x680. 속성 self.df_raw, self.df, self.reg, self.clf 를 None 으로 초기화.
2) 상단 프레임에 버튼 4개: "① CSV 열기", "② 정제 + 학습", "③ 모델 저장", "모델 불러오기". 옆에 상태 문구 Label(StringVar).
3) 본문 왼쪽 LabelFrame "진행 로그" 안에 tk.Text (Consolas 10). write(text) 메서드: 줄을 추가하고 끝으로 스크롤.
4) on_open(): filedialog 로 CSV 를 고르고 data_io.load_csv 로 읽어 self.df_raw 에 넣고 describe_loaded 결과를 로그에 쓴다. 실패하면 messagebox.showerror.
5) on_train(): cleaning.clean_data → cleaning_report 와 eda_summary 를 로그에 쓴다 → model.split_data → train_regressor, train_classifier → 두 evaluate 결과를 로그에 쓴다. df_raw 가 None 이면 경고.
6) on_save(), on_load(): model.save_models / load_models 를 부르고 로그에 남긴다.
7) 파일 끝에 if __name__ == "__main__": App().mainloop()
data_io, cleaning, model 은 같은 폴더의 모듈로 import 한다. 파일 전체 코드만 출력해라.
```

- 기대 결과: 오른쪽 영역은 아직 비어 있어도 된다(다음 프롬프트에서 채운다).
- 확인: python gui.py 로 창이 뜨고 ① → ② 순서로 눌렀을 때 로그에 정제 결과와 R²·F1 이 찍힌다

### 프롬프트 11. gui.py - 입력 폼·임계값 슬라이더·예측 버튼  (파일: gui.py, 약 15분)

```text
gui.py 의 App 에 오른쪽 영역을 추가해라.
1) LabelFrame "새 로트 공정 조건 입력": FEATS 6개 각각 Label + Entry(StringVar) 를 2열 grid 로 배치. 기본값은 {"믹싱_RPM": 1750, "믹싱_온도": 25.0, "코팅_토출압력": 120.0, "건조로_1구간_온도": 110.0, "프레스_압력": 45.0, "프레스_Gap": 95.0}. self.inputs 딕셔너리에 StringVar 를 보관.
2) "불량 판정 임계값" Label + ttk.Scale(0.05~0.95, DoubleVar 기본 0.3).
3) "④ 예측" 버튼 → on_predict(): 입력을 float 로 바꾸고(실패 시 "숫자만 입력하세요" 오류창) model.predict_one 을 임계값과 함께 호출해 "예측 용량 X mAh | 불량 확률 Y | 판정" 을 굵은 파란 Label 에 표시하고 로그에도 쓴다. 모델이 없으면 경고.
4) on_train 의 evaluate_classifier 호출에도 현재 슬라이더 값을 넘겨라.
바뀐 부분을 포함한 gui.py 전체 코드를 출력해라.
```

- 기대 결과: Entry 값은 문자열이므로 float() 변환과 예외 처리가 반드시 있어야 한다.
- 확인: Gap 을 95 → 110 으로 바꾸면 예측 용량이 약 70 mAh 떨어지고 불량 확률이 오른다(교재 4.3 계수 -4.84 mAh/µm 와 방향 일치)

### 프롬프트 12. gui.py - matplotlib 중요도 차트 임베드  (파일: gui.py, 약 10분)

```text
gui.py 에 특성 중요도 차트를 넣어라.
1) 파일 상단에서 matplotlib.use("TkAgg") 를 호출하고 rcParams["font.family"]="Malgun Gothic", rcParams["axes.unicode_minus"]=False 로 한글 축을 깨지지 않게 한다.
2) 오른쪽 아래 LabelFrame "특성 중요도 (MDI)" 에 Figure(figsize=(5,3), dpi=100) 과 FigureCanvasTkAgg 를 넣고 위젯을 pack(fill="both", expand=True).
3) draw_importance(): model.feature_importance(self.reg) 를 가로 막대(barh, 색 "#065A82") 로 그리고 canvas.draw(). 큰 값이 위로 오게 정렬.
4) on_train 과 on_load 의 끝에서 draw_importance() 를 부른다.
바뀐 부분을 포함한 gui.py 전체 코드를 출력해라.
```

- 기대 결과: plt.show() 를 쓰면 안 된다(창이 따로 뜬다). Figure + FigureCanvasTkAgg 조합이어야 한다.
- 확인: ② 학습 후 차트에 프레스_Gap 막대가 가장 길게 나온다


## 조립

### 프롬프트 13. main.py - 조립과 실행  (파일: main.py, 약 5분)

```text
main.py 를 만들어라. gui 모듈에서 App 을 import 해 if __name__ == "__main__": 아래에서 App().mainloop() 를 실행한다. 3줄이면 된다. 코드만 출력해라.
```

- 기대 결과: 5개 파일이 같은 폴더에 있어야 한다.
- 확인: python main.py → ① 열기 → ② 정제+학습 → ④ 예측 → ③ 저장 → 프로그램 종료 후 다시 실행 → 모델 불러오기 → ④ 예측


## 확장

### 프롬프트 14. 확장 과제(선택) - 기능 하나 골라 추가  (파일: (자유), 약 10분)

```text
(하나만 골라 요청)
A. gui.py 에 "CSV 일괄 예측" 버튼을 추가해라. 새 CSV 를 열어 clean_data 로 정제한 뒤 모든 행에 predict_one 과 같은 계산을 벡터 연산으로 적용하고, 원본 열 + 예측_용량_mAh + 불량_확률 + 판정 열을 붙여 "예측결과.csv" 로 저장하고 상위 5행을 로그에 보여 줘라.
B. model.py 에 recommend_threshold(clf, X_test, y_test) 를 추가해라. 임계값 0.05~0.95 를 0.05 간격으로 훑어 F1 이 최대인 값을 돌려주고, GUI 학습 후 슬라이더를 그 값으로 옮겨라.
C. 이 프로그램을 pyinstaller 로 단일 exe 로 만드는 명령과 주의점(한글 경로, matplotlib 백엔드, 데이터 파일 동봉)을 알려 줘라.
```

- 기대 결과: 요청하지 않은 기능이 따라오면 되돌린다("그 부분은 빼고 다시").
- 확인: A: 예측결과.csv 1000행 13열 / B: 추천 임계값이 0.1~0.3 사이 / C: dist 폴더의 exe 가 데이터 없이도 실행되는지


## 디버깅 프롬프트 템플릿

```text
아래 명령을 실행했더니 오류가 났다.
명령: python main.py
오류 전체:
(Traceback 전체 붙여넣기)

관련 파일은 gui.py 다. 원인을 한 줄로 설명하고, 고친 gui.py 전체를 다시 출력해라. 다른 파일은 바꾸지 마라.
```
