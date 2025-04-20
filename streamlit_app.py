import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import os
from scipy.optimize import minimize
import itertools

# PWA 설정 추가
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)

# PWA 메타데이터 추가
st.markdown("""
    <link rel="manifest" href="static/manifest.json">
    <meta name="theme-color" content="#000000">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black">
    <meta name="apple-mobile-web-app-title" content="Portfolio Optimizer">
    
    <script>
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', function() {
                navigator.serviceWorker.register('/static/sw.js')
                    .then(function(registration) {
                        console.log('ServiceWorker registration successful');
                    })
                    .catch(function(err) {
                        console.log('ServiceWorker registration failed: ', err);
                    });
            });
        }
    </script>
""", unsafe_allow_html=True)

# 스크립트가 있는 디렉토리로 작업 디렉토리 변경
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def get_portfolio_data(tickers_weights, start_date, end_date):
    """여러 ETF의 데이터를 가져와 결합"""
    portfolio_data = {}
    error_tickers = []
    
    for ticker in tickers_weights.keys():
        try:
            # 티커 심볼을 대문자로 변환
            ticker_upper = ticker.upper()
            
            # 데이터 가져오기 시도 (최대 3번)
            for attempt in range(3):
                try:
                    stock = yf.Ticker(ticker_upper)
                    # period 매개변수를 사용하여 더 안정적인 데이터 가져오기
                    data = stock.history(
                        start=start_date,
                        end=end_date,
                        interval="1d",
                        auto_adjust=True
                    )
                    
                    if not data.empty and len(data) > 0:
                        portfolio_data[ticker_upper] = data['Close']
                        st.success(f"{ticker_upper} 데이터 로드 완료")
                        break
                    else:
                        if attempt == 2:  # 마지막 시도
                            error_tickers.append(ticker)
                            st.warning(f"{ticker_upper} 데이터를 찾을 수 없습니다.")
                except Exception as e:
                    if attempt == 2:  # 마지막 시도
                        error_tickers.append(ticker)
                        st.warning(f"{ticker_upper} 데이터 로드 실패: {str(e)}")
                    continue
                
        except Exception as e:
            error_tickers.append(ticker)
            st.error(f"{ticker_upper} 처리 중 오류 발생: {str(e)}")
            continue
    
    if error_tickers:
        error_msg = f"다음 ETF의 데이터를 가져오는데 실패했습니다: {', '.join(error_tickers)}"
        st.error(error_msg)
        if not portfolio_data:  # 모든 티커가 실패한 경우
            return pd.DataFrame()
    
    if not portfolio_data:
        st.error("선택한 ETF들의 데이터를 가져오는데 실패했습니다. 티커 심볼을 확인해주세요.")
        return pd.DataFrame()
    
    # 데이터 결합 및 누락값 처리
    df = pd.DataFrame(portfolio_data)
    if df.isnull().any().any():
        st.warning("일부 데이터에 누락값이 있어 forward fill로 처리합니다.")
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def rebalance_portfolio(portfolio_data, weights, rebalance_period):
    """포트폴리오 리밸런싱 수행"""
    # 일별 수익률 계산
    daily_returns = portfolio_data.pct_change()
    
    # 리밸런싱 날짜 설정
    if rebalance_period == 'Monthly':
        rebalance_dates = portfolio_data.resample('M').last().index
    elif rebalance_period == 'Quarterly':
        rebalance_dates = portfolio_data.resample('Q').last().index
    elif rebalance_period == 'Semi-Annual':
        rebalance_dates = portfolio_data.resample('6M').last().index
    elif rebalance_period == 'Yearly':
        rebalance_dates = portfolio_data.resample('Y').last().index
    else:  # No rebalancing
        return (daily_returns * pd.Series(weights)).sum(axis=1)

    # 리밸런싱을 적용한 포트폴리오 수익률 계산
    portfolio_returns = pd.Series(0, index=daily_returns.index)
    current_weights = weights.copy()
    
    for i in range(len(portfolio_data.index)):
        date = portfolio_data.index[i]
        # 리밸런싱 날짜인 경우 가중치 재설정
        if date in rebalance_dates:
            current_weights = weights.copy()
        
        # 일별 포트폴리오 수익률 계산
        if i > 0:
            for ticker, weight in current_weights.items():
                portfolio_returns[i] += daily_returns[ticker][i] * weight
            # 가중치 업데이트
            for ticker in current_weights:
                current_weights[ticker] *= (1 + daily_returns[ticker][i])
            # 가중치 정규화
            weight_sum = sum(current_weights.values())
            if weight_sum != 0:
                current_weights = {k: v/weight_sum for k, v in current_weights.items()}
    
    return portfolio_returns

def calculate_metrics(returns):
    """포트폴리오 성과 지표 계산"""
    # 누적 수익률
    cumulative_returns = (1 + returns).cumprod()
    
    # 연간 수익률
    annualized_return = (1 + returns.mean()) ** 252 - 1
    
    # 표준편차
    standard_deviation = returns.std() * np.sqrt(252)
    
    # 최대 낙폭
    drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
    max_drawdown = drawdowns.min()
    
    # 연간 수익률
    annual_returns = {}
    for year in returns.index.year.unique():
        mask = (returns.index.year == year)
        year_returns = returns[mask]
        annual_returns[str(year)] = (1 + year_returns).prod() - 1
    
    return {
        'annualized_return': annualized_return,
        'standard_deviation': standard_deviation,
        'max_drawdown': max_drawdown,
        'annual_returns': annual_returns,
        'cumulative_returns': cumulative_returns
    }

def portfolio_stats(weights, returns):
    """포트폴리오 수익률과 변동성 계산"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_vol  # 무위험 수익률 0% 가정
    return portfolio_return, portfolio_vol, sharpe_ratio

def calculate_max_drawdown(returns, weights):
    """포트폴리오의 최대 낙폭 계산"""
    portfolio_returns = np.sum(returns * weights, axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
    return drawdowns.min()

def optimize_portfolio(returns, optimization_type='Sharpe', constraints=None):
    """포트폴리오 최적화"""
    n_assets = returns.shape[1]
    
    # 기본 제약조건: 비중 합 = 1
    basic_constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, 1) for _ in range(n_assets)]  # 기본 범위: 0 <= 비중 <= 1
    
    if constraints:
        # 개별 자산 비중 제한
        if 'max_weight' in constraints:
            bounds = [(0, min(1, constraints['max_weight'])) for _ in range(n_assets)]
        if 'min_weight' in constraints:
            bounds = [(max(0, constraints['min_weight']), 1) for _ in range(n_assets)]
    
    # 목적 함수 설정
    if optimization_type == 'Sharpe':
        def objective(weights):
            portfolio_return, portfolio_vol, _ = portfolio_stats(weights, returns)
            return -(portfolio_return / portfolio_vol)  # 음수를 붙여 최대화
            
    elif optimization_type == 'Volatility':
        def objective(weights):
            _, portfolio_vol, _ = portfolio_stats(weights, returns)
            return portfolio_vol
            
    elif optimization_type == 'MaxReturn':
        def objective(weights):
            portfolio_return, _, _ = portfolio_stats(weights, returns)
            return -portfolio_return  # 음수를 붙여 최대화
            
    elif optimization_type == 'MinMDD':
        def objective(weights):
            max_drawdown = calculate_max_drawdown(returns, weights)
            return -max_drawdown  # 음수를 붙여 최대화
    
    # 추가 제약조건 설정
    if constraints:
        if 'target_return' in constraints:
            target_return = constraints['target_return']
            basic_constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(returns.mean() * x) * 252 - target_return
            })
            
        if 'max_mdd' in constraints:
            max_mdd = constraints['max_mdd']
            basic_constraints.append({
                'type': 'ineq',
                'fun': lambda x: max_mdd - (-calculate_max_drawdown(returns, x))
            })
    
    # 초기 가중치 = 균등 배분
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # 최적화 실행
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=basic_constraints
    )
    
    return result.x

def generate_random_portfolios(returns, n_portfolios=1000):
    """무작위 포트폴리오 생성 (효율적 투자선 그래프용)"""
    n_assets = returns.shape[1]
    results = []
    
    for _ in range(n_portfolios):
        # 무작위 가중치 생성
        weights = np.random.random(n_assets)
        weights = weights / np.sum(weights)
        
        # 포트폴리오 성과 계산
        portfolio_return, portfolio_vol, sharpe_ratio = portfolio_stats(weights, returns)
        results.append({
            'Return': portfolio_return,
            'Volatility': portfolio_vol,
            'Sharpe': sharpe_ratio,
            'Weights': weights
        })
    
    return pd.DataFrame(results)

def main():
    st.title('Portfolio Analyzer')
    
    # 사이드바에 입력 컨트롤 배치
    with st.sidebar:
        st.header('Portfolio Settings')
        
        # ETF 선택
        st.subheader('ETF Selection')
        num_etfs = st.number_input('Number of ETFs', min_value=2, max_value=10, value=2)
        
        # ETF 티커 입력 도움말 추가
        st.markdown("""
        **ETF 티커 입력 예시:**
        - SPY (S&P 500 ETF)
        - QQQ (나스닥 100 ETF)
        - IWM (Russell 2000 ETF)
        - VTI (Vanguard Total Stock Market ETF)
        - EFA (선진국 주식)
        - EEM (신흥국 주식)
        - AGG (미국 채권)
        - TLT (미국 장기 국채)
        - GLD (금)
        """)
        
        # ETF 티커 입력
        tickers = []
        for i in range(num_etfs):
            ticker = st.text_input(
                f'ETF {i+1} Ticker',
                key=f'ticker_{i}',
                help="ETF 티커 심볼을 입력하세요 (예: SPY, QQQ)"
            ).strip().upper()  # 공백 제거 및 대문자 변환
            if ticker:
                tickers.append(ticker)
        
        # 최적화 옵션
        optimization_option = st.radio(
            'Portfolio Optimization',
            ['Manual Weights', 'Maximum Sharpe Ratio', 'Minimum Volatility', 
             'Maximum Return', 'Minimum Maximum Drawdown']
        )
        
        # 초기 가중치 설정
        weights = {}
        
        # 수동 가중치 입력
        if optimization_option == 'Manual Weights':
            st.write("Enter weights (total should be 100%)")
            total_weight = 0
            for i, ticker in enumerate(tickers):
                weight = st.number_input(f'Weight for {ticker} (%)', 
                                      min_value=0.0, 
                                      max_value=100.0, 
                                      value=100.0/len(tickers) if tickers else 0.0,
                                      step=0.1,
                                      format="%.1f",
                                      key=f'weight_{i}')
                weights[ticker] = weight / 100
                total_weight += weight
            
            if abs(total_weight - 100) > 0.01:  # 0.01% 오차 허용
                st.warning(f'Total weight must be 100%. Current total: {total_weight:.1f}%')
        else:
            # 최적화 옵션 선택 시 균등 가중치로 초기화
            if tickers:
                equal_weight = 1.0 / len(tickers)
                weights = {ticker: equal_weight for ticker in tickers}
        
        # 기간 설정
        st.subheader('Time Period')
        start_date = st.date_input('Start Date', datetime.now() - timedelta(days=365))
        end_date = st.date_input('End Date', datetime.now())
        
        # 리밸런싱 주기
        rebalance_period = st.selectbox(
            'Rebalancing Period',
            ['No Rebalancing', 'Monthly', 'Quarterly', 'Semi-Annual', 'Yearly']
        )
        
        analyze_button = st.button('Analyze Portfolio')
    
    # 메인 영역에 결과 표시
    if analyze_button:
        if len(tickers) < 2:
            st.error('Please select at least 2 ETFs')
        else:
            try:
                # 포트폴리오 데이터 가져오기
                portfolio_data = get_portfolio_data(weights, start_date, end_date)
                
                if portfolio_data.empty:
                    st.error('No data available for the selected ETFs and time period')
                else:
                    # 수익률 계산
                    returns = portfolio_data.pct_change().dropna()
                    
                    # 최적화 옵션에 따른 가중치 계산
                    if optimization_option != 'Manual Weights':
                        opt_type = {
                            'Maximum Sharpe Ratio': 'Sharpe',
                            'Minimum Volatility': 'Volatility',
                            'Maximum Return': 'MaxReturn',
                            'Minimum Maximum Drawdown': 'MinMDD'
                        }[optimization_option]
                        
                        optimized_weights = optimize_portfolio(returns, opt_type)
                        weights = {ticker: weight for ticker, weight in zip(tickers, optimized_weights)}
                    
                    # 포트폴리오 성과 계산
                    portfolio_returns = rebalance_portfolio(
                        portfolio_data,
                        weights,
                        rebalance_period
                    )
                    
                    # 성과 지표 계산
                    metrics = calculate_metrics(portfolio_returns)
                    
                    # Sharpe Ratio 계산 (무위험 수익률 = 0% 가정)
                    sharpe_ratio = metrics['annualized_return'] / metrics['standard_deviation']
                    
                    # 결과 표시
                    st.header('Portfolio Analysis Results')

                    # 가중치 표시
                    st.subheader('Portfolio Weights')
                    weights_df = pd.DataFrame({
                        'ETF': list(weights.keys()),
                        'Weight': [f"{w:.2%}" for w in weights.values()]
                    })
                    st.table(weights_df)
                    
                    # 성과 지표
                    st.subheader('Performance Metrics')
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric('Annual Return', f"{metrics['annualized_return']:.2%}")
                    with col2:
                        st.metric('Volatility', f"{metrics['standard_deviation']:.2%}")
                    with col3:
                        st.metric('Sharpe Ratio', f"{sharpe_ratio:.2f}")
                    with col4:
                        st.metric('Max Drawdown', f"{metrics['max_drawdown']:.2%}")
                    
                    # 추가 성과 지표
                    st.subheader('Risk Metrics')
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        positive_months = (portfolio_returns > 0).sum()
                        total_months = len(portfolio_returns)
                        st.metric('Win Rate', f"{(positive_months/total_months)*100:.1f}%")
                    with col2:
                        best_month = portfolio_returns.max()
                        st.metric('Best Month', f"{best_month:.2%}")
                    with col3:
                        worst_month = portfolio_returns.min()
                        st.metric('Worst Month', f"{worst_month:.2%}")
                    
                    # 수익률 차트
                    st.subheader('Performance Chart')
                    
                    # 누적 수익률 차트
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=metrics['cumulative_returns'].index,
                        y=metrics['cumulative_returns'].values,
                        mode='lines',
                        name='Portfolio Value',
                        fill='tozeroy'
                    ))
                    fig.update_layout(
                        title='Cumulative Returns',
                        xaxis_title='Date',
                        yaxis_title='Growth of $1',
                        hovermode='x unified',
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 연간 수익률
                    annual_returns_df = pd.DataFrame(
                        list(metrics['annual_returns'].items()),
                        columns=['Year', 'Return']
                    )
                    fig = px.bar(
                        annual_returns_df,
                        x='Year',
                        y='Return',
                        title='Annual Returns',
                        text=annual_returns_df['Return'].apply(lambda x: f'{x:.2%}')
                    )
                    fig.update_traces(textposition='outside')
                    fig.update_layout(
                        xaxis_title='Year',
                        yaxis_title='Return',
                        yaxis_tickformat=',.0%'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f'Error analyzing portfolio: {str(e)}')

if __name__ == "__main__":
    main() 