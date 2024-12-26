import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List
import ta

class MutualFundAnalyzer:
    def __init__(self):
        pass
        
    def analyze_fund(self, data: pd.DataFrame, info: dict) -> Dict[str, Any]:
        """Analyze mutual fund performance and metrics"""
        result = {}
        
        # Basic Information
        result['fund_info'] = self._get_fund_info(info)
        
        # Performance Metrics
        result['performance'] = self._calculate_performance_metrics(data)
        
        # Risk Metrics
        result['risk'] = self._calculate_risk_metrics(data)
        
        # Returns Analysis
        result['returns'] = self._analyze_returns(data)
        
        # Technical Analysis
        result['technical'] = self._perform_technical_analysis(data)
        
        # Charts
        result['charts'] = self._create_analysis_charts(data)
        
        return result
    
    def _get_fund_info(self, info: dict) -> Dict[str, Any]:
        """Extract and format fund information"""
        return {
            'name': info.get('longName', ''),
            'category': info.get('category', ''),
            'fund_family': info.get('fundFamily', ''),
            'inception_date': info.get('fundInceptionDate', ''),
            'total_assets': info.get('totalAssets', 0),
            'expense_ratio': info.get('annualReportExpenseRatio', 0),
            'investment_strategy': info.get('investmentStrategy', ''),
            'fund_manager': info.get('fundManager', ''),
            'minimum_investment': info.get('minimumInvestment', 0)
        }
    
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics"""
        returns = data['Close'].pct_change()
        
        return {
            'ytd_return': self._calculate_ytd_return(data),
            'one_month_return': self._calculate_period_return(returns, 21),
            'three_month_return': self._calculate_period_return(returns, 63),
            'six_month_return': self._calculate_period_return(returns, 126),
            'one_year_return': self._calculate_period_return(returns, 252),
            'three_year_return': self._calculate_period_return(returns, 756),
            'five_year_return': self._calculate_period_return(returns, 1260),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'alpha': self._calculate_alpha(returns),
            'beta': self._calculate_beta(returns)
        }
    
    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics"""
        returns = data['Close'].pct_change()
        
        return {
            'volatility': returns.std() * np.sqrt(252),  # Annualized volatility
            'max_drawdown': self._calculate_max_drawdown(data['Close']),
            'var_95': self._calculate_var(returns, 0.95),
            'var_99': self._calculate_var(returns, 0.99),
            'downside_deviation': self._calculate_downside_deviation(returns),
            'tracking_error': self._calculate_tracking_error(returns),
            'information_ratio': self._calculate_information_ratio(returns)
        }
    
    def _analyze_returns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze return patterns and distributions"""
        returns = data['Close'].pct_change()
        
        return {
            'best_month': returns.resample('M').sum().max(),
            'worst_month': returns.resample('M').sum().min(),
            'positive_months': (returns.resample('M').sum() > 0).sum() / len(returns.resample('M').sum()),
            'negative_months': (returns.resample('M').sum() < 0).sum() / len(returns.resample('M').sum()),
            'avg_monthly_return': returns.resample('M').sum().mean(),
            'monthly_return_std': returns.resample('M').sum().std(),
            'return_skewness': returns.skew(),
            'return_kurtosis': returns.kurtosis()
        }
    
    def _perform_technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform technical analysis"""
        return {
            'rsi': ta.momentum.rsi(data['Close'])[-1],
            'macd': ta.trend.macd_diff(data['Close'])[-1],
            'bollinger_position': self._calculate_bollinger_position(data),
            'trend_strength': self._calculate_trend_strength(data),
            'momentum': self._calculate_momentum(data)
        }
    
    def _create_analysis_charts(self, data: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create analysis charts"""
        charts = {}
        
        # NAV History Chart
        charts['nav_history'] = self._create_nav_history_chart(data)
        
        # Returns Distribution Chart
        charts['returns_dist'] = self._create_returns_distribution_chart(data)
        
        # Rolling Returns Chart
        charts['rolling_returns'] = self._create_rolling_returns_chart(data)
        
        # Risk-Return Chart
        charts['risk_return'] = self._create_risk_return_chart(data)
        
        return charts
    
    def _calculate_ytd_return(self, data: pd.DataFrame) -> float:
        """Calculate year-to-date return"""
        current_year = pd.Timestamp.now().year
        ytd_data = data[data.index.year == current_year]
        if len(ytd_data) > 0:
            return (ytd_data['Close'][-1] / ytd_data['Close'][0] - 1) * 100
        return 0
    
    def _calculate_period_return(self, returns: pd.Series, periods: int) -> float:
        """Calculate return for a specific period"""
        if len(returns) >= periods:
            return (np.prod(1 + returns[-periods:]) - 1) * 100
        return 0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.03) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252
        if len(excess_returns) > 0:
            return np.sqrt(252) * excess_returns.mean() / returns.std()
        return 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.03) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate/252
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        return 0
    
    def _calculate_alpha(self, returns: pd.Series) -> float:
        """Calculate Jensen's Alpha"""
        # This is a simplified calculation
        return returns.mean() * 252
    
    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate Beta"""
        # This is a simplified calculation
        market_returns = returns  # Should use market index returns
        return returns.cov(market_returns) / market_returns.var()
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = prices.expanding().max()
        drawdowns = prices/rolling_max - 1
        return drawdowns.min() * 100
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk"""
        return returns.quantile(1 - confidence) * 100
    
    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside deviation"""
        negative_returns = returns[returns < 0]
        return negative_returns.std() * np.sqrt(252) * 100
    
    def _calculate_tracking_error(self, returns: pd.Series) -> float:
        """Calculate tracking error"""
        # This is a simplified calculation
        return returns.std() * np.sqrt(252) * 100
    
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate information ratio"""
        # This is a simplified calculation
        return returns.mean() / returns.std() * np.sqrt(252)
    
    def _calculate_bollinger_position(self, data: pd.DataFrame) -> float:
        """Calculate current position within Bollinger Bands"""
        bb = ta.volatility.BollingerBands(data['Close'])
        current_price = data['Close'][-1]
        upper = bb.bollinger_hband()[-1]
        lower = bb.bollinger_lband()[-1]
        middle = bb.bollinger_mavg()[-1]
        
        # Calculate position (-1 to 1, where -1 is at lower band, 1 is at upper band)
        return (current_price - middle) / (upper - middle) if upper != middle else 0
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength"""
        # Using ADX indicator
        adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'])
        return adx.adx()[-1]
    
    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum"""
        # Using ROC (Rate of Change)
        return ta.momentum.roc(data['Close'], window=14)[-1]
    
    def _create_nav_history_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create NAV history chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            name='NAV',
            line=dict(color='#00CC96')
        ))
        
        fig.update_layout(
            title='NAV History',
            xaxis_title='Date',
            yaxis_title='NAV',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def _create_returns_distribution_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create returns distribution chart"""
        returns = data['Close'].pct_change().dropna()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns Distribution',
            marker_color='#00CC96'
        ))
        
        fig.update_layout(
            title='Returns Distribution',
            xaxis_title='Return',
            yaxis_title='Frequency',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def _create_rolling_returns_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create rolling returns chart"""
        returns = data['Close'].pct_change()
        rolling_returns = returns.rolling(window=252).mean() * 252 * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=rolling_returns,
            name='1-Year Rolling Returns',
            line=dict(color='#00CC96')
        ))
        
        fig.update_layout(
            title='1-Year Rolling Returns',
            xaxis_title='Date',
            yaxis_title='Return (%)',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def _create_risk_return_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create risk-return scatter plot"""
        returns = data['Close'].pct_change()
        rolling_returns = returns.rolling(window=252).mean() * 252 * 100
        rolling_vol = returns.rolling(window=252).std() * np.sqrt(252) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=rolling_vol,
            y=rolling_returns,
            mode='markers',
            name='Risk-Return',
            marker=dict(
                size=8,
                color='#00CC96',
                colorscale='Viridis'
            )
        ))
        
        fig.update_layout(
            title='Risk-Return Analysis',
            xaxis_title='Risk (Volatility %)',
            yaxis_title='Return (%)',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig 