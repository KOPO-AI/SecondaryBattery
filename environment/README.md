# 2차전지 교육 실습 환경 설정 (sbai)

Windows 11 환경에서 **Miniconda, sbai (Python 3.12 + ipykernel), VS Code 자동 설치 및 확장 프로그램, 환경 변수(PATH) 등록**까지 원클릭으로 완료해 주는 통합 스크립트입니다.

---

## 📁 파일 구성

| 파일명 | 기능 및 사용법 |
| :--- | :--- |
| **`setup_environment.bat`** | **[통합 원클릭 설치]** 더블 클릭 시 모든 환경 자동 구성 |
| **`run_sbai_env.bat`** | **[가상환경 실행기]** 더블 클릭 시 `(sbai)` 가상환경 터미널 즉시 실행 |
| **`README.md`** | 환경 안내 및 사용 가이드 |

---

## 🚀 `setup_environment.bat` 자동화 단계

1. **Conda 확인 및 설치**:
   - 시스템에 Conda가 없으면 최신 Miniconda 64비트 자동 다운로드 및 무인 설치 진행
2. **Conda 환경 변수(PATH) 자동 등록**:
   - `condabin`, `Scripts`, `Library\bin` 등을 Windows 사용자 환경 변수(`User PATH`)에 영구 등록
3. **`sbai` (Python 3.12 + ipykernel) 가상환경 구축**:
   - Python 3.12 가상환경 생성 및 `ipykernel` 설치
   - Jupyter에 **`Python 3.12 (sbai)`** 커널 자동 등록
   - `conda init` (CMD 및 PowerShell 연동)
4. **Visual Studio Code 확인 및 설치**:
   - 이미 설치되어 있으면: **설치 스킵** 후 다음 단계로 자동 진행
   - 설치되어 있지 않으면: 최신 64비트 VS Code 자동 다운로드 및 무인 백그라운드 설치
5. **VS Code 환경 변수(PATH) 등록 및 Extension 설치**:
   - VS Code 실행 경로를 환경 변수에 영구 등록
   - Python 확장 (`ms-python.python`) 자동 설치
   - Jupyter 확장 (`ms-toolsai.jupyter`) 자동 설치

---

## 💡 사용 방법

### 1. 가상환경 바로 실행
* 폴더 내 [**`run_sbai_env.bat`**](file:///E:/강의/Course/2차전지/environment/run_sbai_env.bat)을 **더블 클릭**합니다.
* 열린 터미널에서 바로 `python --version`, `jupyter`, `code .` 등을 사용하실 수 있습니다.

### 2. VS Code에서 Jupyter Notebook (.ipynb) 커널 선택
1. VS Code에서 실습할 `.ipynb` 파일을 엽니다.
2. 노트북 화면 우측 상단 **[커널 선택 (Select Kernel)]** 클릭
3. **`Python 3.12 (sbai)`** 또는 **`sbai (Python 3.12.x)`** 선택
