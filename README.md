# Portfolio Optimizer PWA

ETF 포트폴리오 최적화 및 분석을 위한 Progressive Web App입니다.

## 주요 기능

- 다중 ETF 포트폴리오 구성
- 다양한 최적화 전략:
  - Maximum Sharpe Ratio
  - Minimum Volatility
  - Maximum Return
  - Minimum Maximum Drawdown
- 수동 가중치 설정
- 포트폴리오 성과 분석
- Efficient Frontier 분석
- 자동 리밸런싱 시뮬레이션

## 설치 방법

1. 저장소 클론:
```bash
git clone https://github.com/dhslnice77/portfolio-optimizer-pwa.git
cd portfolio-optimizer-pwa
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

3. 앱 실행:
```bash
streamlit run streamlit_app.py
```

## 배포

### 로컬 배포
이 앱은 Progressive Web App으로 설계되어 있어 웹 브라우저에서 설치하여 데스크톱 앱처럼 사용할 수 있습니다.

### Streamlit Cloud 배포
1. GitHub 저장소에 코드를 푸시합니다.
2. [Streamlit Cloud](https://streamlit.io/cloud)에 접속하여 로그인합니다.
3. "New app" 버튼을 클릭하고 GitHub 저장소를 선택합니다.
4. 메인 파일 경로를 `streamlit_app.py`로 설정합니다.
5. "Deploy" 버튼을 클릭하여 배포를 시작합니다.

## 기술 스택

- Python
- Streamlit
- yfinance
- pandas
- numpy
- plotly
- scipy

## API 키 설정

앱을 사용하기 위해서는 Alpha Vantage API 키가 필요합니다. 무료 API 키는 [Alpha Vantage](https://www.alphavantage.co/support/#api-key)에서 발급받을 수 있습니다.
