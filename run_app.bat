@echo off
echo Portfolio Optimizer 실행중...
echo.

REM Python 환경 확인
python --version > nul 2>&1
if errorlevel 1 (
    echo Python이 설치되어 있지 않습니다.
    echo Python을 설치해주세요: https://www.python.org/downloads/
    pause
    exit
)

REM 필요한 패키지 설치 확인 및 설치
echo 필요한 패키지 확인 중...
pip install -r requirements.txt

echo.
echo 포트폴리오 최적화 앱을 시작합니다...
echo 브라우저가 자동으로 열립니다...
echo.

REM Streamlit 앱 실행
streamlit run streamlit_app.py

pause 