import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm
import itertools

# Initialize session state
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'show_help' not in st.session_state:
    st.session_state.show_help = False

# Set page config
st.set_page_config(
    page_title="포트폴리오 최적화",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    .help-text {
        font-size: 0.9em;
        color: #666;
        margin-top: 0.5em;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1em;
        border-radius: 0.5em;
        margin-bottom: 1em;
    }
    .error-message {
        color: #ff4b4b;
        font-weight: bold;
    }
    .success-message {
        color: #00cc00;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def show_help():
    """도움말 문서 표시"""
    st.sidebar.title("도움말")
    st.sidebar.markdown("""
    ### ETF 입력
    - 한 줄에 하나의 ETF 티커를 입력하세요.
    - 예시 ETF를 참고하여 입력할 수 있습니다.
    
    ### 최적화 옵션
    - **최대 샤프 비율**: 위험 대비 수익률이 가장 높은 포트폴리오
    - **최소 변동성**: 변동성이 가장 낮은 포트폴리오
    - **목표 수익률**: 지정한 수익률을 달성하는 최소 변동성 포트폴리오
    
    ### 리밸런싱
    - 포트폴리오를 주기적으로 원래 가중치로 조정합니다.
    - 거래 비용을 고려하여 시뮬레이션합니다.
    
    ### 결과 해석
    - **상관관계 행렬**: 자산 간의 관계를 보여줍니다.
    - **낙폭 분석**: 포트폴리오의 최대 손실폭을 보여줍니다.
    - **성과 분석**: 연간/월별 수익률과 통계를 보여줍니다.
    """)

def get_portfolio_data(tickers, start_date, end_date):
    """
    Fetch portfolio data using yfinance
    """
    all_data = pd.DataFrame()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        try:
            status_text.text(f"{ticker} 데이터를 가져오는 중...")
            data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
            data = pd.DataFrame(data).rename(columns={'Adj Close': ticker})
            
            if all_data.empty:
                all_data = data
            else:
                all_data = all_data.join(data)
            
            progress_bar.progress((i + 1) / len(tickers))
            
        except Exception as e:
            st.warning(f"{ticker} 데이터를 가져올 수 없습니다: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if all_data.empty:
        st.error("선택한 ETF들의 데이터를 가져올 수 없습니다.")
        return None
        
    # Handle missing values
    filtered_data = all_data.ffill().bfill()
    
    return filtered_data

def calculate_portfolio_stats(returns, weights):
    """포트폴리오 통계 계산"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_vol
    return portfolio_return, portfolio_vol, sharpe_ratio

def calculate_risk_metrics(returns, weights, market_returns=None):
    """위험 지표 계산"""
    # 포트폴리오 수익률 계산
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # 최대 낙폭 계산
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # VaR 계산 (95% 신뢰구간)
    var_95 = norm.ppf(0.05, portfolio_returns.mean(), portfolio_returns.std())
    var_99 = norm.ppf(0.01, portfolio_returns.mean(), portfolio_returns.std())
    
    # 베타 계산 (시장 수익률이 있는 경우)
    beta = None
    if market_returns is not None:
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        beta = covariance / market_variance
    
    # 승률 계산
    win_rate = (portfolio_returns > 0).mean()
    
    # 최대 상승/하락 계산
    max_gain = portfolio_returns.max()
    max_loss = portfolio_returns.min()
    
    return {
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'var_99': var_99,
        'beta': beta,
        'win_rate': win_rate,
        'max_gain': max_gain,
        'max_loss': max_loss
    }

def calculate_performance_metrics(returns, weights):
    """성과 지표 계산"""
    # 포트폴리오 수익률 계산
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # 연간 수익률 계산
    annual_returns = portfolio_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    
    # 월별 수익률 계산
    monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # 월별 수익률 분포
    monthly_stats = {
        '평균': monthly_returns.mean(),
        '중앙값': monthly_returns.median(),
        '표준편차': monthly_returns.std(),
        '왜도': monthly_returns.skew(),
        '첨도': monthly_returns.kurtosis()
    }
    
    # 승률 계산
    monthly_win_rate = (monthly_returns > 0).mean()
    
    # 최대 연속 상승/하락
    consecutive_returns = portfolio_returns > 0
    max_consecutive_gains = max(len(list(g)) for k, g in itertools.groupby(consecutive_returns) if k)
    max_consecutive_losses = max(len(list(g)) for k, g in itertools.groupby(consecutive_returns) if not k)
    
    return {
        'annual_returns': annual_returns,
        'monthly_returns': monthly_returns,
        'monthly_stats': monthly_stats,
        'monthly_win_rate': monthly_win_rate,
        'max_consecutive_gains': max_consecutive_gains,
        'max_consecutive_losses': max_consecutive_losses
    }

def optimize_portfolio(returns, optimization_type='sharpe', target_return=None):
    """포트폴리오 최적화"""
    n_assets = returns.shape[1]
    
    # 제약조건: 가중치 합 = 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # 가중치 범위: 0 <= w <= 1
    bounds = [(0, 1) for _ in range(n_assets)]
    
    # 목표 수익률 제약조건 추가
    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(returns.mean() * x) * 252 - target_return
        })
    
    # 목적 함수 설정
    if optimization_type == 'sharpe':
        def objective(weights):
            portfolio_return, portfolio_vol, _ = calculate_portfolio_stats(returns, weights)
            return -portfolio_return / portfolio_vol  # 샤프 비율 최대화
    elif optimization_type == 'volatility':
        def objective(weights):
            _, portfolio_vol, _ = calculate_portfolio_stats(returns, weights)
            return portfolio_vol  # 변동성 최소화
    elif optimization_type == 'return':
        def objective(weights):
            portfolio_return, _, _ = calculate_portfolio_stats(returns, weights)
            return -portfolio_return  # 수익률 최대화
    
    # 초기 가중치 = 균등 가중치
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # 최적화 실행
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

def rebalance_portfolio(portfolio_data, target_weights, rebalance_period, transaction_cost=0.001):
    """
    포트폴리오 리밸런싱 시뮬레이션
    """
    # 일별 수익률 계산
    returns = portfolio_data.pct_change()
    
    # 리밸런싱 날짜 설정
    if rebalance_period == '월간':
        rebalance_dates = portfolio_data.resample('M').last().index
    elif rebalance_period == '분기':
        rebalance_dates = portfolio_data.resample('Q').last().index
    elif rebalance_period == '반기':
        rebalance_dates = portfolio_data.resample('6M').last().index
    elif rebalance_period == '연간':
        rebalance_dates = portfolio_data.resample('Y').last().index
    else:  # 리밸런싱 없음
        return (returns * pd.Series(target_weights)).sum(axis=1)
    
    # 초기 포트폴리오 가치
    portfolio_value = 100  # 기준값 100
    current_weights = target_weights.copy()
    
    # 일별 포트폴리오 수익률 저장
    portfolio_returns = pd.Series(0.0, index=returns.index)
    transaction_costs = pd.Series(0.0, index=returns.index)
    
    for i in range(1, len(returns)):
        date = returns.index[i]
        
        # 리밸런싱 날짜인 경우
        if date in rebalance_dates:
            # 리밸런싱 전 가중치 계산
            old_weights = current_weights.copy()
            
            # 리밸런싱 비용 계산
            weight_changes = np.abs(target_weights - old_weights)
            cost = np.sum(weight_changes) * transaction_cost
            transaction_costs[date] = cost
            
            # 리밸런싱 후 수익률 계산 (비용 반영)
            portfolio_returns[date] = (returns.iloc[i] * target_weights).sum() - cost
            current_weights = target_weights.copy()
        else:
            # 리밸런싱이 아닌 날은 현재 가중치로 수익률 계산
            portfolio_returns[date] = (returns.iloc[i] * current_weights).sum()
            
            # 가중치 업데이트 (자연스러운 변화)
            current_weights = current_weights * (1 + returns.iloc[i])
            current_weights = current_weights / current_weights.sum()
    
    return portfolio_returns, transaction_costs

def main():
    st.title("포트폴리오 최적화")
    
    # Help button
    if st.sidebar.button("도움말 보기"):
        st.session_state.show_help = not st.session_state.show_help
    
    if st.session_state.show_help:
        show_help()
    
    # Example ETFs with improved UI
    st.subheader("ETF 입력")
    example_etfs = {
        "미국 주식": "SPY",
        "선진국 주식": "VEA",
        "신흥국 주식": "VWO",
        "미국 채권": "AGG",
        "금": "GLD"
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        etf_input = st.text_area(
            "ETF 티커를 입력하세요 (한 줄에 하나씩):",
            help="분석하려는 ETF 티커를 한 줄에 하나씩 입력하세요.",
            height=100
        )
        
    with col2:
        st.markdown("### 예시 ETF")
        for name, ticker in example_etfs.items():
            st.markdown(f"<div class='metric-card'>{name}: <strong>{ticker}</strong></div>", unsafe_allow_html=True)
    
    # Date range selection with improved UI
    st.subheader("기간 설정")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "시작일",
            datetime.now() - timedelta(days=365*2),
            help="분석을 시작할 날짜를 선택하세요."
        )
    with col2:
        end_date = st.date_input(
            "종료일",
            datetime.now(),
            help="분석을 종료할 날짜를 선택하세요."
        )
        
    if start_date >= end_date:
        st.error("종료일은 시작일보다 이후여야 합니다.")
        return

    # Optimization options with improved UI
    st.subheader("포트폴리오 최적화 옵션")
    optimization_type = st.radio(
        "최적화 유형",
        ["최대 샤프 비율", "최소 변동성", "목표 수익률"],
        horizontal=True,
        help="포트폴리오 최적화 방법을 선택하세요."
    )
    
    target_return = None
    if optimization_type == "목표 수익률":
        target_return = st.number_input(
            "목표 연간 수익률 (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1,
            help="달성하려는 연간 수익률을 입력하세요."
        ) / 100

    # Rebalancing options with improved UI
    st.subheader("리밸런싱 옵션")
    col1, col2 = st.columns(2)
    with col1:
        rebalance_period = st.selectbox(
            "리밸런싱 주기",
            ["리밸런싱 없음", "월간", "분기", "반기", "연간"],
            help="포트폴리오를 리밸런싱할 주기를 선택하세요."
        )
    with col2:
        transaction_cost = st.number_input(
            "거래 비용 (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="리밸런싱 시 발생하는 거래 비용을 입력하세요."
        ) / 100

    # Process ETF input with loading indicator
    if etf_input:
        tickers = [ticker.strip().upper() for ticker in etf_input.split('\n') if ticker.strip()]
        
        if st.button("포트폴리오 분석", help="포트폴리오를 분석합니다."):
            with st.spinner("포트폴리오를 분석하는 중..."):
                portfolio_data = get_portfolio_data(tickers, start_date, end_date)
                
                if portfolio_data is not None:
                    st.session_state.portfolio_data = portfolio_data
                    
                    # Calculate returns
                    returns = portfolio_data.pct_change().dropna()
                    
                    # Get market returns (SPY) for beta calculation
                    market_data = get_portfolio_data(['SPY'], start_date, end_date)
                    market_returns = market_data.pct_change().dropna() if market_data is not None else None
                    
                    # Optimize portfolio
                    opt_type = {
                        "최대 샤프 비율": "sharpe",
                        "최소 변동성": "volatility",
                        "목표 수익률": "return"
                    }[optimization_type]
                    
                    optimal_weights = optimize_portfolio(returns, opt_type, target_return)
                    
                    # Calculate portfolio statistics
                    portfolio_return, portfolio_vol, sharpe_ratio = calculate_portfolio_stats(returns, optimal_weights)
                    
                    # Calculate risk metrics
                    risk_metrics = calculate_risk_metrics(returns, optimal_weights, market_returns)
                    
                    # Calculate performance metrics
                    performance_metrics = calculate_performance_metrics(returns, optimal_weights)
                    
                    # Display results
                    st.success("포트폴리오 분석이 완료되었습니다!")
                    
                    # Display optimal weights
                    st.subheader("최적 포트폴리오 가중치")
                    weights_df = pd.DataFrame({
                        'ETF': tickers,
                        '가중치 (%)': (optimal_weights * 100).round(2)
                    })
                    st.dataframe(weights_df)
                    
                    # Display portfolio statistics
                    st.subheader("포트폴리오 통계")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("연간 수익률", f"{portfolio_return:.2%}")
                    with col2:
                        st.metric("연간 변동성", f"{portfolio_vol:.2%}")
                    with col3:
                        st.metric("샤프 비율", f"{sharpe_ratio:.2f}")
                    
                    # Display risk metrics
                    st.subheader("위험 지표")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("최대 낙폭", f"{risk_metrics['max_drawdown']:.2%}")
                    with col2:
                        st.metric("95% VaR", f"{risk_metrics['var_95']:.2%}")
                    with col3:
                        st.metric("승률", f"{risk_metrics['win_rate']:.2%}")
                    with col4:
                        if risk_metrics['beta'] is not None:
                            st.metric("베타", f"{risk_metrics['beta']:.2f}")
                    
                    # Display performance metrics
                    st.subheader("성과 분석")
                    
                    # Annual returns chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=performance_metrics['annual_returns'].index.year,
                        y=performance_metrics['annual_returns'].values * 100,
                        text=performance_metrics['annual_returns'].apply(lambda x: f"{x:.1%}"),
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title="연간 수익률",
                        xaxis_title="연도",
                        yaxis_title="수익률 (%)",
                        width=800,
                        height=500
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Monthly return distribution
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=performance_metrics['monthly_returns'] * 100,
                        nbinsx=20,
                        name="월별 수익률"
                    ))
                    
                    fig.update_layout(
                        title="월별 수익률 분포",
                        xaxis_title="수익률 (%)",
                        yaxis_title="빈도",
                        width=800,
                        height=500
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Monthly statistics
                    st.subheader("월별 통계")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("평균 수익률", f"{performance_metrics['monthly_stats']['평균']:.2%}")
                    with col2:
                        st.metric("중앙값", f"{performance_metrics['monthly_stats']['중앙값']:.2%}")
                    with col3:
                        st.metric("표준편차", f"{performance_metrics['monthly_stats']['표준편차']:.2%}")
                    with col4:
                        st.metric("월간 승률", f"{performance_metrics['monthly_win_rate']:.2%}")
                    
                    # Consecutive returns
                    st.subheader("연속 수익/손실")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("최대 연속 상승", f"{performance_metrics['max_consecutive_gains']}개월")
                    with col2:
                        st.metric("최대 연속 하락", f"{performance_metrics['max_consecutive_losses']}개월")
                    
                    # Display drawdown chart
                    st.subheader("낙폭 분석")
                    portfolio_returns = (returns * optimal_weights).sum(axis=1)
                    cumulative_returns = (1 + portfolio_returns).cumprod()
                    rolling_max = cumulative_returns.expanding().max()
                    drawdowns = (cumulative_returns - rolling_max) / rolling_max
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=drawdowns.index,
                        y=drawdowns.values * 100,
                        name="낙폭",
                        fill='tozeroy'
                    ))
                    
                    fig.update_layout(
                        title="포트폴리오 낙폭",
                        xaxis_title="날짜",
                        yaxis_title="낙폭 (%)",
                        width=800,
                        height=500
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Simulate rebalancing
                    portfolio_returns, transaction_costs = rebalance_portfolio(
                        portfolio_data,
                        optimal_weights,
                        rebalance_period,
                        transaction_cost
                    )
                    
                    # Calculate cumulative returns
                    cumulative_returns = (1 + portfolio_returns).cumprod()
                    
                    # Display rebalancing results
                    st.subheader("리밸런싱 결과")
                    
                    # Cumulative returns chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns.values * 100,
                        name="포트폴리오 가치",
                        mode='lines'
                    ))
                    
                    fig.update_layout(
                        title="누적 수익률 (기준값=100)",
                        xaxis_title="날짜",
                        yaxis_title="포트폴리오 가치",
                        width=800,
                        height=500
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Transaction costs
                    total_cost = transaction_costs.sum() * 100
                    st.metric("총 거래 비용", f"{total_cost:.2f}%")
                    
                    # Display correlation matrix
                    st.subheader("상관관계 행렬")
                    corr_matrix = returns.corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmin=-1,
                        zmax=1
                    ))
                    
                    fig.update_layout(
                        title="자산 간 상관관계",
                        width=800,
                        height=600
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Display price chart
                    st.subheader("가격 성과")
                    normalized_data = portfolio_data / portfolio_data.iloc[0] * 100
                    
                    fig = go.Figure()
                    for column in normalized_data.columns:
                        fig.add_trace(go.Scatter(
                            x=normalized_data.index,
                            y=normalized_data[column],
                            name=column,
                            mode='lines'
                        ))
                    
                    fig.update_layout(
                        title="정규화된 가격 성과 (기준값=100)",
                        xaxis_title="날짜",
                        yaxis_title="가격",
                        width=800,
                        height=500
                    )
                    
                    st.plotly_chart(fig)

if __name__ == "__main__":
    main() 