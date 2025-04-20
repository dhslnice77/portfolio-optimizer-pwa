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

이 앱은 Progressive Web App으로 설계되어 있어 웹 브라우저에서 설치하여 데스크톱 앱처럼 사용할 수 있습니다.

## 기술 스택

- Python
- Streamlit
- yfinance
- pandas
- numpy
- plotly
- scipy
