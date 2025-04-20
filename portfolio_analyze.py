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
    for ticker in tickers_weights.keys():
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if not data.empty:
            portfolio_data[ticker] = data['Close']
    
    return pd.DataFrame(portfolio_data)

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

def optimize_portfolio(returns, optimization_type='Sharpe', constraints=None):
    """포트폴리오 최적화
    
    Parameters:
    - returns: 일별 수익률
    - optimization_type: 최적화 방식
    - constraints: 추가 제약조건
        - target_return: 목표 수익률
        - max_mdd: 최대 허용 MDD
        - max_weight: 개별 자산 최대 비중
        - min_weight: 개별 자산 최소 비중
    """
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
    
    def portfolio_metrics(weights):
        """포트폴리오 주요 지표 계산"""
        portfolio_return, portfolio_vol, _ = portfolio_metrics(weights)
        return portfolio_return, portfolio_vol, max_drawdown
    
    # 목적 함수 설정
    if optimization_type == 'Sharpe':
        def objective(weights):
            portfolio_return, portfolio_vol, _ = portfolio_metrics(weights)
            return -(portfolio_return / portfolio_vol)  # 음수를 붙여 최대화
            
    elif optimization_type == 'Volatility':
        def objective(weights):
            _, portfolio_vol, _ = portfolio_metrics(weights)
            return portfolio_vol
            
    elif optimization_type == 'MaxReturn':
        def objective(weights):
            portfolio_return, _, _ = portfolio_metrics(weights)
            return -portfolio_return  # 음수를 붙여 최대화
            
    elif optimization_type == 'MinMDD':
        def objective(weights):
            _, _, max_drawdown = portfolio_metrics(weights)
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
                'fun': lambda x: max_mdd - (-portfolio_metrics(x)[2])  # MDD 제한
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
    
    with st.sidebar:
        st.header('Portfolio Settings')
        
        # ETF 선택
        st.subheader('ETF Selection')
        num_etfs = st.number_input('Number of ETFs', min_value=2, max_value=10, value=2)
        
        # 최적화 옵션
        optimization_option = st.radio(
            'Portfolio Optimization',
            ['Manual Weights', 'Maximum Sharpe Ratio', 'Minimum Volatility', 
             'Maximum Return', 'Minimum Drawdown', 'Custom Constraints']
        )
        
        # 제약조건 설정 (Custom Constraints 선택 시)
        constraints = {}
        if optimization_option == 'Custom Constraints':
            st.subheader('Optimization Constraints')
            
            use_target_return = st.checkbox('Set Target Return')
            if use_target_return:
                target_return = st.slider('Target Annual Return (%)', 
                                        min_value=0, max_value=50, value=10) / 100
                constraints['target_return'] = target_return
            
            use_max_mdd = st.checkbox('Limit Maximum Drawdown')
            if use_max_mdd:
                max_mdd = st.slider('Maximum Allowed Drawdown (%)', 
                                  min_value=5, max_value=50, value=20) / 100
                constraints['max_mdd'] = max_mdd
            
            use_weight_limits = st.checkbox('Set Weight Limits')
            if use_weight_limits:
                col1, col2 = st.columns(2)
                with col1:
                    min_weight = st.number_input('Minimum Weight (%)', 
                                               min_value=0, max_value=50, value=5) / 100
                    constraints['min_weight'] = min_weight
                with col2:
                    max_weight = st.number_input('Maximum Weight (%)', 
                                               min_value=0, max_value=100, value=50) / 100
                    constraints['max_weight'] = max_weight
        
        # 티커 입력
        tickers = {}
        for i in range(num_etfs):
            ticker = st.text_input(f'ETF Ticker {i+1}', 
                                 value=f'SPY' if i==0 else f'QQQ' if i==1 else '')
            if ticker:
                tickers[ticker] = 0
        
        # 나머지 설정들...
        start_date = st.date_input('Start Date', datetime.now() - timedelta(days=365*10))
        end_date = st.date_input('End Date', datetime.now())
        initial_investment = st.number_input('Initial Investment ($)', value=10000)
        rebalance_period = st.selectbox(
            'Rebalancing Period',
            ['None', 'Monthly', 'Quarterly', 'Semi-Annual', 'Yearly']
        )

    if st.sidebar.button('Analyze Portfolio'):
        try:
            # 데이터 가져오기
            portfolio_data = get_portfolio_data(tickers, start_date, end_date)
            
            if portfolio_data.empty:
                st.error("No data available for selected tickers")
                return
            
            # 일별 수익률 계산
            returns = portfolio_data.pct_change().dropna()
            
            # 최적화 수행
            if optimization_option != 'Manual Weights':
                opt_type = {
                    'Maximum Sharpe Ratio': 'Sharpe',
                    'Minimum Volatility': 'Volatility',
                    'Maximum Return': 'MaxReturn',
                    'Minimum Drawdown': 'MinMDD',
                    'Custom Constraints': 'Sharpe'
                }[optimization_option]
                
                optimal_weights = optimize_portfolio(
                    returns,
                    optimization_type=opt_type,
                    constraints=constraints if optimization_option == 'Custom Constraints' else None
                )
                weights = optimal_weights
            
            # 최적화된 포트폴리오 구성 표시를 더 눈에 띄게
            st.header('Optimized Portfolio Results')
            
            # 1. 최적 비중 표시
            st.subheader('Optimal Portfolio Weights')
            composition_df = pd.DataFrame({
                'ETF': list(tickers.keys()),
                'Weight': [f"{w*100:.2f}%" for w in weights],
                'Allocation': [f"${initial_investment * w:,.2f}" for w in weights]
            })
            st.table(composition_df)
            
            # 2. 최적화 포트폴리오 주요 지표
            opt_return, opt_vol, opt_sharpe = portfolio_stats(np.array(weights), returns)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Expected Annual Return",
                    f"{opt_return*100:.2f}%"
                )
            with col2:
                st.metric(
                    "Expected Volatility",
                    f"{opt_vol*100:.2f}%"
                )
            with col3:
                st.metric(
                    "Sharpe Ratio",
                    f"{opt_sharpe:.2f}"
                )
            
            # 3. 효율적 투자선 그래프
            st.subheader('Portfolio Optimization Analysis')
            random_portfolios = generate_random_portfolios(returns)
            
            fig = go.Figure()
            
            # 무작위 포트폴리오 산점도
            fig.add_trace(go.Scatter(
                x=random_portfolios['Volatility'],
                y=random_portfolios['Return'],
                mode='markers',
                name='Random Portfolios',
                marker=dict(
                    size=5,
                    color=random_portfolios['Sharpe'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Sharpe Ratio')
                )
            ))
            
            # 최적 포트폴리오 위치
            fig.add_trace(go.Scatter(
                x=[opt_vol],
                y=[opt_return],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(
                    size=15,
                    symbol='star',
                    color='red'
                )
            ))
            
            fig.update_layout(
                title=f'{optimization_option} Portfolio on Efficient Frontier',
                xaxis_title='Expected Volatility (Standard Deviation)',
                yaxis_title='Expected Annual Return',
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # 4. 백테스트 결과 표시
            st.header('Portfolio Backtest Results')
            st.write("Historical performance analysis using the optimal weights:")

            # 포트폴리오 수익률 계산 (리밸런싱 적용)
            portfolio_returns = rebalance_portfolio(portfolio_data, dict(zip(tickers.keys(), weights)), rebalance_period)
            metrics = calculate_metrics(portfolio_returns)

            # 백테스트 주요 지표
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Historical Return",
                    f"{metrics['annualized_return']*100:.2f}%"
                )
            with col2:
                st.metric(
                    "Historical Volatility",
                    f"{metrics['standard_deviation']*100:.2f}%"
                )
            with col3:
                st.metric(
                    "Maximum Drawdown",
                    f"{metrics['max_drawdown']*100:.2f}%"
                )
            with col4:
                sharpe = metrics['annualized_return'] / metrics['standard_deviation']
                st.metric(
                    "Historical Sharpe",
                    f"{sharpe:.2f}"
                )

            # 포트폴리오 가치 변화
            st.subheader('Portfolio Value Growth')
            portfolio_values = initial_investment * metrics['cumulative_returns']

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=portfolio_values.index,
                    y=portfolio_values,
                    mode='lines',
                    name='Portfolio Value',
                    fill='tozeroy'
                )
            )
            fig.update_layout(
                yaxis_title='Portfolio Value ($)',
                hovermode='x',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # 연간 수익률
            st.subheader('Annual Returns')
            annual_returns_df = pd.DataFrame(
                list(metrics['annual_returns'].items()),
                columns=['Year', 'Return']
            )
            annual_returns_df['Return'] = annual_returns_df['Return'] * 100

            fig = px.bar(
                annual_returns_df,
                x='Year',
                y='Return',
                title='Annual Returns (%)'
            )
            fig.update_traces(marker_color='rgb(0, 123, 255)')
            st.plotly_chart(fig, use_container_width=True)

            # 추가 통계
            st.subheader('Additional Performance Metrics')
            positive_months = (portfolio_returns > 0).sum()
            total_months = len(portfolio_returns.dropna())

            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.metric(
                    "Positive Months",
                    f"{positive_months} / {total_months}"
                )
                st.metric(
                    "Win Rate",
                    f"{(positive_months/total_months)*100:.1f}%"
                )
            with stats_col2:
                st.metric(
                    "Best Year",
                    f"{max(metrics['annual_returns'].values())*100:.2f}%"
                )
                st.metric(
                    "Worst Year",
                    f"{min(metrics['annual_returns'].values())*100:.2f}%"
                )
            with stats_col3:
                st.metric(
                    "Average Annual Return",
                    f"{np.mean(list(metrics['annual_returns'].values()))*100:.2f}%"
                )
                st.metric(
                    "Return Consistency",
                    f"{(len([r for r in metrics['annual_returns'].values() if r > 0]) / len(metrics['annual_returns']))*100:.1f}%"
                )
            
        except Exception as e:
            st.error(f"Error analyzing portfolio: {str(e)}")

if __name__ == "__main__":
    main() 