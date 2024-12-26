import streamlit as st
from data_fetcher import StockDataFetcher
from stock_analyzer import StockAnalyzer
from ml_predictor import StockPredictor
from ui_components import UIComponents
from utils.stock_lists import get_nifty50_stocks, get_global_stocks, get_default_stocks
import pandas as pd
import plotly.graph_objects as go
import os
from llm_analyzer import LLMAnalyzer
from typing import List, Dict
import yfinance as yf
from datetime import datetime
import numpy as np
import ta

st.set_page_config(page_title="Global Stock Market Analysis", layout="wide")

class StockAnalysisApp:
    def __init__(self):
        self.ui = UIComponents()
        self.data_fetcher = StockDataFetcher()
        self.analyzer = StockAnalyzer()
        self.predictor = StockPredictor()
        self.llm_analyzer = LLMAnalyzer()
        self.ui.apply_custom_styling()
    
    def main(self):
        """Main application logic"""
        try:
            # Create sidebar navigation
            st.sidebar.title("Navigation")
            selected_page = st.sidebar.radio(
                "Select Page",
                ["Company Analysis", "Market Analysis", "Portfolio Analysis", "News Analysis", "AI Predictions"],
                format_func=lambda x: f"{self._get_icon(x)} {x}",
                label_visibility="collapsed"
            )
            
            # Main content area
            st.title("Stock Market Analysis")
            
            # Get stock lists
            market_selection = st.selectbox(
                "Select Market",
                ["Global Stocks", "NIFTY 50", "All Stocks"],
                label_visibility="visible"
            )
            
            # Combine stock lists based on selection
            if market_selection == "Global Stocks":
                stocks = get_global_stocks()
            elif market_selection == "NIFTY 50":
                stocks = get_nifty50_stocks()
            else:  # All Stocks
                stocks = {
                    **get_global_stocks(),
                    **get_nifty50_stocks()
                }
            
            # Create stock selector
            selected_stock = self.ui.create_stock_selector(stocks)
            symbol = stocks[selected_stock]
            
            # Add debug information
            st.write(f"Selected stock: {selected_stock} ({symbol})")
            
            try:
                # Fetch stock data with progress indicator
                with st.spinner('Fetching stock data...'):
                    stock_info = self.data_fetcher.get_stock_info(symbol)
                    historical_data = self.data_fetcher.get_stock_data(symbol, '1y')
                    news_data = self.data_fetcher.get_stock_news(symbol)
                
                # Verify data
                if not stock_info:
                    st.error(f"Could not fetch information for {symbol}")
                    return
                
                if historical_data.empty:
                    st.error(f"No historical data available for {symbol}")
                    return
                
                # Show selected page content
                if selected_page == "Market Analysis":
                    self._show_market_analysis(historical_data, symbol, stock_info)
                elif selected_page == "Portfolio Analysis":
                    self._show_portfolio_analysis()
                elif selected_page == "News Analysis":
                    self._show_news_analysis(news_data)
                elif selected_page == "AI Predictions":
                    self._show_ai_analysis(symbol, stock_info, historical_data, news_data)
                else:  # Default to Company Analysis
                    self._show_company_analysis(historical_data, symbol, stock_info)
                    
            except Exception as e:
                st.error(f"Error processing data for {symbol}: {str(e)}")
                st.write("Debug information:")
                st.write({
                    "stock_info_available": bool(stock_info),
                    "historical_data_available": not historical_data.empty if historical_data is not None else False,
                    "news_data_available": bool(news_data)
                })
                raise e
                
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            raise e
    
    def _get_icon(self, page_name: str) -> str:
        """Get icon for navigation item"""
        icons = {
            "Company Analysis": "📊",
            "Market Analysis": "📈",
            "Portfolio Analysis": "💼",
            "News Analysis": "📰",
            "AI Predictions": "🤖"
        }
        return icons.get(page_name, "")
    
    def _show_company_analysis(self, historical_data: pd.DataFrame, symbol: str, stock_info: dict):
        """Show company analysis with fundamental metrics and technical indicators."""
        try:
            # Add custom CSS for better formatting
            st.markdown("""
            <style>
            .metric-container {
                background-color: rgba(28, 131, 225, 0.1);
                border: 1px solid rgba(28, 131, 225, 0.1);
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
            }
            .chart-container {
                background-color: rgba(17, 17, 17, 0.1);
                border-radius: 10px;
                padding: 20px;
                margin: 15px 0;
            }
            .news-container {
                background-color: rgba(25, 25, 25, 0.2);
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
            }
            .metric-table {
                width: 100%;
                border-collapse: collapse;
            }
            .metric-table th, .metric-table td {
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            .metric-table th {
                background-color: rgba(255, 255, 255, 0.05);
            }
            .news-item {
                padding: 10px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            .news-title {
                font-weight: bold;
                margin-bottom: 5px;
            }
            .news-date {
                font-size: 0.8em;
                color: #888;
            }
            .trade-info {
                background-color: rgba(255, 87, 34, 0.1);
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
            }
            </style>
            """, unsafe_allow_html=True)

            # Row 1: About Company and News
            col1, col2 = st.columns([6, 4])
            
            with col1:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.subheader("About Company")
                company_info = stock_info.get('longBusinessSummary', 'No company description available.')
                st.write(company_info)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='news-container'>", unsafe_allow_html=True)
                st.subheader("Latest News")
                news_data = self._fetch_news(symbol)[:3]  # Get top 3 news
                for news in news_data:
                    st.markdown(f"""
                    <div class='news-item'>
                        <div class='news-title'>{news['title']}</div>
                        <div class='news-date'>{news['published']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Row 2: Trade Information in 3 columns
            st.markdown("<div class='trade-info'>", unsafe_allow_html=True)
            st.subheader("Trade Information")
            
            col1, col2, col3 = st.columns(3)
            
            # Column 1: Trade Information
            with col1:
                trade_data = [
                    ["Traded Volume (Lakhs)", f"{stock_info.get('volume', 0)/100000:.2f}"],
                    ["Traded Value (₹ Cr.)", f"{(stock_info.get('volume', 0) * historical_data['Close'].iloc[-1])/10000000:.2f}"],
                    ["Total Market Cap (₹ Cr.)", f"{stock_info.get('marketCap', 0)/10000000:.2f}"],
                    ["Free Float Market Cap (₹ Cr.)", f"{stock_info.get('marketCap', 0) * 0.5/10000000:.2f}"],
                    ["Impact cost", "0.01"],
                    ["% of Deliverable / Traded Quantity", "51.18 %"],
                    ["Applicable Margin Rate", "12.50"],
                    ["Face Value", "10"]
                ]
                
                st.markdown("""
                <table class='metric-table'>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                """ + "\n".join([f"<tr><td>{row[0]}</td><td>{row[1]}</td></tr>" for row in trade_data]) + """
                </table>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Column 2: Price Information
            with col2:
                price_data = [
                    ["52 Week High", f"₹{historical_data['High'].tail(252).max():,.2f}"],
                    ["52 Week Low", f"₹{historical_data['Low'].tail(252).min():,.2f}"],
                    ["Upper Band", f"₹{historical_data['High'].iloc[-1] * 1.2:,.2f}"],
                    ["Lower Band", f"₹{historical_data['Low'].iloc[-1] * 0.8:,.2f}"],
                    ["Price Band", "No Band"],
                    ["Daily Volatility", f"{historical_data['Close'].pct_change().std() * 100:.2f}%"],
                    ["Annualised Volatility", f"{historical_data['Close'].pct_change().std() * np.sqrt(252) * 100:.2f}%"],
                    ["Tick Size", "0.05"]
                ]
                
                st.markdown("""
                <table class='metric-table'>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                """ + "\n".join([f"<tr><td>{row[0]}</td><td>{row[1]}</td></tr>" for row in price_data]) + """
                </table>
                """, unsafe_allow_html=True)
            
            # Column 3: Securities Information
            with col3:
                securities_data = [
                    ["Status", "Listed"],
                    ["Trading Status", "Active"],
                    ["Date of Listing", stock_info.get('firstTradingDate', 'N/A')],
                    ["Adjusted P/E", f"{stock_info.get('trailingPE', 0):.2f}"],
                    ["Symbol P/E", f"{stock_info.get('trailingPE', 0):.2f}"],
                    ["Index", "NIFTY 50"],
                    ["Basic Industry", stock_info.get('industry', 'N/A')],
                ]
                
                st.markdown("""
                <table class='metric-table'>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                """ + "\n".join([f"<tr><td>{row[0]}</td><td>{row[1]}</td></tr>" for row in securities_data]) + """
                </table>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

            # Row 3: Market Overview and Fundamental Analysis
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.subheader("Current Market Overview")
                
                metrics_data = [
                    ["Current Price", f"${historical_data['Close'].iloc[-1]:.2f}", f"{((historical_data['Close'].iloc[-1] - historical_data['Close'].iloc[-2]) / historical_data['Close'].iloc[-2] * 100):.2f}%"],
                    ["52-Week Range", f"${historical_data['Low'].tail(252).min():.2f} - ${historical_data['High'].tail(252).max():.2f}", ""],
                    ["30-Day Avg Volume", f"{int(historical_data['Volume'].tail(30).mean()):,}", ""],
                    ["P/E Ratio", f"{stock_info.get('trailingPE', stock_info.get('forwardPE', 'N/A')):.2f}", ""],
                    ["EPS", stock_info.get('eps', 'N/A'), ""]
                ]
                
                st.markdown("""
                <table class='metric-table'>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Change</th>
                    </tr>
                """ + "\n".join([f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td></tr>" for row in metrics_data]) + """
                </table>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.subheader("Fundamental Analysis")
                
                fundamental_data = [
                    ["Free Cash Flow", f"${stock_info.get('freeCashflow', 0):,.0f}"],
                    ["Book Value/Share", f"${stock_info.get('bookValue', 0):.2f}"],
                    ["Net Profit Margin", f"{stock_info.get('profitMargins', 0)*100:.1f}%"],
                    ["Debt/Equity", f"{stock_info.get('debtToEquity', 0):.2f}"],
                    ["ROE", f"{stock_info.get('returnOnEquity', 0)*100:.1f}%"],
                    ["Beta", f"{stock_info.get('beta', 1):.2f}"],
                    ["Market Cap", f"${stock_info.get('marketCap', 0):,.0f}"],
                    ["Dividend Yield", f"{stock_info.get('dividendYield', 0)*100:.2f}%"]
                ]
                
                st.markdown("""
                <table class='metric-table'>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                """ + "\n".join([f"<tr><td>{row[0]}</td><td>{row[1]}</td></tr>" for row in fundamental_data]) + """
                </table>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Row 4: Technical Analysis Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.subheader("RSI Analysis")
                fig_rsi = go.Figure()
                rsi = ta.momentum.RSIIndicator(historical_data['Close']).rsi()
                
                fig_rsi.add_trace(go.Scatter(
                    x=historical_data.index[-60:],
                    y=rsi[-60:],
                    name="RSI"
                ))
                
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                
                fig_rsi.update_layout(
                    title="RSI (14)",
                    yaxis_title="RSI",
                    template="plotly_dark",
                    height=300
                )
                
                st.plotly_chart(fig_rsi, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.subheader("MACD Analysis")
                fig_macd = go.Figure()
                macd = ta.trend.MACD(historical_data['Close'])
                
                fig_macd.add_trace(go.Scatter(
                    x=historical_data.index[-60:],
                    y=macd.macd()[-60:],
                    name="MACD"
                ))
                
                fig_macd.add_trace(go.Scatter(
                    x=historical_data.index[-60:],
                    y=macd.macd_signal()[-60:],
                    name="Signal"
                ))
                
                fig_macd.update_layout(
                    title="MACD",
                    yaxis_title="Value",
                    template="plotly_dark",
                    height=300
                )
                
                st.plotly_chart(fig_macd, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Row 5: Price Analysis with Time Filters
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Price Analysis")
            
            # Time period filters
            col1, col2 = st.columns([2, 3])
            with col1:
                time_filter = st.selectbox(
                    "Select Time Period",
                    ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "YTD", "Fiscal Year"],
                    index=2
                )
            
            with col2:
                # Calculate date ranges based on selection
                end_date = historical_data.index[-1]
                if time_filter == "1 Month":
                    start_date = end_date - pd.DateOffset(months=1)
                    default_days = 30
                elif time_filter == "3 Months":
                    start_date = end_date - pd.DateOffset(months=3)
                    default_days = 90
                elif time_filter == "6 Months":
                    start_date = end_date - pd.DateOffset(months=6)
                    default_days = 180
                elif time_filter == "1 Year":
                    start_date = end_date - pd.DateOffset(years=1)
                    default_days = 365
                elif time_filter == "2 Years":
                    start_date = end_date - pd.DateOffset(years=2)
                    default_days = 730
                elif time_filter == "5 Years":
                    start_date = end_date - pd.DateOffset(years=5)
                    default_days = 1825
                elif time_filter == "YTD":
                    start_date = pd.Timestamp(end_date.year, 1, 1)
                    default_days = (end_date - start_date).days
                else:  # Fiscal Year
                    if end_date.month >= 4:
                        start_date = pd.Timestamp(end_date.year, 4, 1)
                    else:
                        start_date = pd.Timestamp(end_date.year - 1, 4, 1)
                    default_days = (end_date - start_date).days
                
                # Fine-tune adjustment slider
                days_to_show = st.slider(
                    "Adjust Time Range (Days)",
                    min_value=30,
                    max_value=len(historical_data),
                    value=min(default_days, len(historical_data)),
                    step=1
                )
            
            # Filter data based on selection
            mask = (historical_data.index >= start_date) & (historical_data.index <= end_date)
            display_data = historical_data[mask].tail(days_to_show)
            
            # Create candlestick chart
            fig = go.Figure()
            
            # Add candlestick
            fig.add_trace(go.Candlestick(
                x=display_data.index,
                open=display_data['Open'],
                high=display_data['High'],
                low=display_data['Low'],
                close=display_data['Close'],
                name="OHLC"
            ))
            
            # Add Bollinger Bands
            bb = ta.volatility.BollingerBands(display_data['Close'])
            fig.add_trace(go.Scatter(
                x=display_data.index,
                y=bb.bollinger_hband(),
                name="BB Upper",
                line=dict(color='gray', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=display_data.index,
                y=bb.bollinger_lband(),
                name="BB Lower",
                line=dict(color='gray', dash='dash'),
                fill='tonexty'
            ))
            
            # Add volume bars
            fig.add_trace(go.Bar(
                x=display_data.index,
                y=display_data['Volume'],
                name="Volume",
                yaxis="y2",
                marker_color='rgba(0,204,150,0.3)'
            ))
            
            fig.update_layout(
                title=f"Price Chart with Bollinger Bands ({time_filter})",
                yaxis_title="Price (₹)",
                yaxis2=dict(
                    title="Volume",
                    overlaying="y",
                    side="right",
                    showgrid=False
                ),
                template="plotly_dark",
                height=600,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add price statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Period High", f"₹{display_data['High'].max():,.2f}")
            with col2:
                st.metric("Period Low", f"₹{display_data['Low'].min():,.2f}")
            with col3:
                st.metric("Period Return", 
                         f"{((display_data['Close'].iloc[-1] / display_data['Close'].iloc[0] - 1) * 100):,.2f}%")
            
            # Calculate technical indicators for the report
            rsi = ta.momentum.RSIIndicator(historical_data['Close']).rsi()
            macd = ta.trend.MACD(historical_data['Close'])
            bb = ta.volatility.BollingerBands(historical_data['Close'])
            
            # Create report data
            report_data = {
                'price_metrics': {
                    'current_price': historical_data['Close'].iloc[-1],
                    'price_change': historical_data['Close'].iloc[-1] - historical_data['Close'].iloc[-2],
                    'volume': historical_data['Volume'].tail(30).mean(),
                    'pe_ratio': stock_info.get('peRatio', 'N/A'),
                    'eps': stock_info.get('eps', 'N/A'),
                    'fifty_two_week_high': historical_data['High'].tail(252).max(),
                    'fifty_two_week_low': historical_data['Low'].tail(252).min(),
                    'avg_volume_30d': historical_data['Volume'].tail(30).mean()
                },
                'fundamental_metrics': {
                    'market_cap': stock_info.get('marketCap', 0),
                    'free_cash_flow': stock_info.get('freeCashflow', 0),
                    'profit_margin': stock_info.get('profitMargins', 0),
                    'debt_equity': stock_info.get('debtToEquity', 0),
                    'roe': stock_info.get('returnOnEquity', 0),
                    'beta': stock_info.get('beta', 1),
                    'book_value': stock_info.get('bookValue', 0),
                    'dividend_yield': stock_info.get('dividendYield', 0)
                },
                'technical_indicators': {
                    'rsi': rsi.iloc[-1],
                    'macd': macd.macd().iloc[-1],
                    'macd_signal': macd.macd_signal().iloc[-1],
                    'bb_upper': bb.bollinger_hband().iloc[-1],
                    'bb_lower': bb.bollinger_lband().iloc[-1],
                    'bb_middle': bb.bollinger_mavg().iloc[-1],
                    'volume_trend': (historical_data['Volume'].iloc[-1] / historical_data['Volume'].tail(20).mean() - 1) * 100
                }
            }
            
            # Generate and display analysis report
            report = self.llm_analyzer.generate_analysis_report(symbol, report_data)
            st.markdown(report, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error in company analysis: {str(e)}")
            raise e
    
    def _show_market_analysis(self, historical_data: pd.DataFrame, symbol: str, stock_info: dict):
        """Display market analysis section"""
        st.markdown("### Market Analysis")
        
        # Sector Performance
        st.markdown("#### Sector Performance")
        sector_data = self.data_fetcher.get_sector_performance()
        if sector_data:
            fig = self.analyzer.plot_sector_performance(sector_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Market Share Analysis
        st.markdown("#### Market Share Analysis")
        market_share_data = self.data_fetcher.get_market_share_data(symbol)
        if market_share_data:
            fig = self.analyzer.plot_market_share(market_share_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Competitor Analysis
            st.markdown("#### Competitor Performance")
            competitor_data = self.data_fetcher.get_competitor_data(symbol)
            if competitor_data:
                fig = self.analyzer.plot_competitor_performance(competitor_data)
                st.plotly_chart(fig, use_container_width=True)
    
    def _show_portfolio_analysis(self):
        """Display portfolio analysis section"""
        st.markdown("### Portfolio Analysis")
        
        # Portfolio settings
        st.markdown("#### Investment Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            investment_amount = st.number_input("Monthly Investment (₹)", min_value=1000, value=10000, step=1000)
        with col2:
            risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
        with col3:
            investment_horizon = st.selectbox("Investment Horizon", ["1 Year", "3 Years", "5 Years", "10 Years"])
        
        # Generate portfolio recommendation
        if st.button("Generate Portfolio"):
            portfolio = self.analyzer.generate_portfolio(investment_amount, risk_tolerance, investment_horizon)
            
            # Display allocation
            st.markdown("#### Recommended Allocation")
            fig = self.analyzer.plot_market_share(portfolio["allocation"])
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            st.markdown("#### Portfolio Metrics")
            metrics = portfolio["metrics"]
            cols = st.columns(len(metrics))
            for col, (metric, value) in zip(cols, metrics.items()):
                col.metric(
                    metric,
                    f"₹{value:,.2f}" if metric in ["Monthly Investment", "Projected Value"] else f"{value:.1f}%"
                )
    
    def _show_ai_analysis(self, symbol: str, stock_info: dict, historical_data: pd.DataFrame, news_data: list):
        """Show AI-powered analysis and predictions with additional insights and charts."""
        try:
            # Add custom CSS for better formatting
            st.markdown("""
            <style>
            .metric-container {
                background-color: rgba(28, 131, 225, 0.1);
                border: 1px solid rgba(28, 131, 225, 0.1);
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
            }
            .chart-container {
                background-color: rgba(17, 17, 17, 0.1);
                border-radius: 10px;
                padding: 20px;
                margin: 15px 0;
            }
            .insight-container {
                background-color: rgba(25, 25, 25, 0.2);
                border-radius: 10px;
                padding: 20px;
                margin: 15px 0;
            }
            .metric-table {
                width: 100%;
                border-collapse: collapse;
            }
            .metric-table th, .metric-table td {
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            .metric-table th {
                background-color: rgba(255, 255, 255, 0.05);
            }
            </style>
            """, unsafe_allow_html=True)

            # Create two main columns for the layout
            left_col, right_col = st.columns([3, 2])

            with left_col:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.subheader("Current Market Overview")
                
                # Create metrics table
                metrics_data = [
                    ["Current Price", f"${historical_data['Close'].iloc[-1]:.2f}", f"{((historical_data['Close'].iloc[-1] - historical_data['Close'].iloc[-2]) / historical_data['Close'].iloc[-2] * 100):.2f}%"],
                    ["52-Week Range", f"${historical_data['Low'].tail(252).min():.2f} - ${historical_data['High'].tail(252).max():.2f}", ""],
                    ["30-Day Avg Volume", f"{int(historical_data['Volume'].tail(30).mean()):,}", ""],
                    ["P/E Ratio", f"{stock_info.get('trailingPE', stock_info.get('forwardPE', 'N/A')):.2f}", ""],
                    ["EPS", stock_info.get('eps', 'N/A'), ""]
                ]
                
                st.markdown("""
                <table class='metric-table'>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Change</th>
                    </tr>
                """ + "\n".join([f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td></tr>" for row in metrics_data]) + """
                </table>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Fundamental Analysis Table
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.subheader("Fundamental Analysis")
                
                fundamental_data = [
                    ["Free Cash Flow", f"${stock_info.get('freeCashflow', 0):,.0f}"],
                    ["Book Value/Share", f"${stock_info.get('bookValue', 0):.2f}"],
                    ["Net Profit Margin", f"{stock_info.get('profitMargins', 0)*100:.1f}%"],
                    ["Debt/Equity", f"{stock_info.get('debtToEquity', 0):.2f}"],
                    ["ROE", f"{stock_info.get('returnOnEquity', 0)*100:.1f}%"],
                    ["Beta", f"{stock_info.get('beta', 1):.2f}"],
                    ["Market Cap", f"${stock_info.get('marketCap', 0):,.0f}"],
                    ["Dividend Yield", f"{stock_info.get('dividendYield', 0)*100:.2f}%"]
                ]
                
                st.markdown("""
                <table class='metric-table'>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                """ + "\n".join([f"<tr><td>{row[0]}</td><td>{row[1]}</td></tr>" for row in fundamental_data]) + """
                </table>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with right_col:
                # Technical Charts Section
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.subheader("Technical Analysis")
                
                # RSI Chart
                fig_rsi = go.Figure()
                rsi = ta.momentum.RSIIndicator(historical_data['Close']).rsi()
                
                fig_rsi.add_trace(go.Scatter(
                    x=historical_data.index[-60:],
                    y=rsi[-60:],
                    name="RSI"
                ))
                
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                
                fig_rsi.update_layout(
                    title="RSI (14)",
                    yaxis_title="RSI",
                    template="plotly_dark",
                    height=250
                )
                
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # MACD Chart
                fig_macd = go.Figure()
                macd = ta.trend.MACD(historical_data['Close'])
                
                fig_macd.add_trace(go.Scatter(
                    x=historical_data.index[-60:],
                    y=macd.macd()[-60:],
                    name="MACD"
                ))
                
                fig_macd.add_trace(go.Scatter(
                    x=historical_data.index[-60:],
                    y=macd.macd_signal()[-60:],
                    name="Signal"
                ))
                
                fig_macd.update_layout(
                    title="MACD",
                    yaxis_title="Value",
                    template="plotly_dark",
                    height=250
                )
                
                st.plotly_chart(fig_macd, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # AI Prediction Section
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("AI Price Predictions")
            
            prediction_days = st.slider(
                label="Number of Days to Predict",
                min_value=1,
                max_value=30,
                value=7,
                help="Select the number of days to predict into the future"
            )
            
            # Generate predictions
            with st.spinner("Generating AI predictions..."):
                predictions = self.predictor.predict(historical_data, symbol, prediction_days)
                
            if predictions is not None:
                # Create figure for predictions
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=historical_data.index[-60:],
                    y=historical_data['Close'][-60:],
                    name='Historical',
                    line=dict(color='#00ff00', width=2)
                ))
                
                # Add predictions
                future_dates = pd.date_range(
                    start=historical_data.index[-1] + pd.Timedelta(days=1),
                    periods=prediction_days,
                    freq='B'
                )
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=predictions,
                    name='Predicted',
                    line=dict(color='#ff9900', width=2, dash='dash')
                ))
                
                # Add confidence interval
                std_dev = historical_data['Close'].std()
                upper_bound = predictions + (std_dev * 1.96)
                lower_bound = predictions - (std_dev * 1.96)
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=upper_bound,
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=lower_bound,
                    name='Lower Bound',
                    line=dict(width=0),
                    fillcolor='rgba(255, 153, 0, 0.2)',
                    fill='tonexty',
                    showlegend=False
                ))
                
                fig.update_layout(
                    title="Stock Price Prediction with Confidence Interval",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    showlegend=True,
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Calculate technical indicators for the report
            rsi = ta.momentum.RSIIndicator(historical_data['Close']).rsi()
            macd = ta.trend.MACD(historical_data['Close'])
            bb = ta.volatility.BollingerBands(historical_data['Close'])
            
            # Create report data
            report_data = {
                'price_metrics': {
                    'current_price': historical_data['Close'].iloc[-1],
                    'price_change': historical_data['Close'].iloc[-1] - historical_data['Close'].iloc[-2],
                    'volume': historical_data['Volume'].tail(30).mean(),
                    'pe_ratio': stock_info.get('peRatio', 'N/A'),
                    'eps': stock_info.get('eps', 'N/A'),
                    'fifty_two_week_high': historical_data['High'].tail(252).max(),
                    'fifty_two_week_low': historical_data['Low'].tail(252).min(),
                    'avg_volume_30d': historical_data['Volume'].tail(30).mean()
                },
                'fundamental_metrics': {
                    'market_cap': stock_info.get('marketCap', 0),
                    'free_cash_flow': stock_info.get('freeCashflow', 0),
                    'profit_margin': stock_info.get('profitMargins', 0),
                    'debt_equity': stock_info.get('debtToEquity', 0),
                    'roe': stock_info.get('returnOnEquity', 0),
                    'beta': stock_info.get('beta', 1),
                    'book_value': stock_info.get('bookValue', 0),
                    'dividend_yield': stock_info.get('dividendYield', 0)
                },
                'technical_indicators': {
                    'rsi': rsi.iloc[-1],
                    'macd': macd.macd().iloc[-1],
                    'macd_signal': macd.macd_signal().iloc[-1],
                    'bb_upper': bb.bollinger_hband().iloc[-1],
                    'bb_lower': bb.bollinger_lband().iloc[-1],
                    'bb_middle': bb.bollinger_mavg().iloc[-1],
                    'volume_trend': (historical_data['Volume'].iloc[-1] / historical_data['Volume'].tail(20).mean() - 1) * 100
                }
            }
            
            # Generate and display analysis report
            report = self.llm_analyzer.generate_analysis_report(symbol, report_data)
            st.markdown(report, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error generating analysis: {str(e)}")
            raise e
    
    def _show_news_analysis(self, news_data):
        """Display news analysis section"""
        if not news_data:
            st.warning("No news data available")
            return
            
        st.markdown("""
        <style>
        .news-card {
            background-color: rgba(49, 51, 63, 0.2);
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .news-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #ffffff;
        }
        .news-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 10px;
        }
        .news-detail-item {
            display: flex;
            flex-direction: column;
        }
        .detail-label {
            font-size: 12px;
            color: #9e9e9e;
            margin-bottom: 3px;
        }
        .detail-value {
            font-size: 14px;
            color: #ffffff;
        }
        .news-summary {
            font-size: 14px;
            line-height: 1.5;
            color: #e0e0e0;
            margin-top: 10px;
        }
        .sentiment-positive { color: #00cc96; }
        .sentiment-negative { color: #ef553b; }
        .sentiment-neutral { color: #ffa15a; }
        </style>
        """, unsafe_allow_html=True)
        
        for news in news_data:
            # Get news details with fallbacks
            title = news.get('title', 'No title available')
            published = news.get('published', 'Date not available')
            source = news.get('source', 'Source not available')
            category = news.get('category', 'General')
            sentiment = news.get('sentiment', 'neutral')
            impact = news.get('impact', 'Medium')
            summary = news.get('summary', 'No summary available')
            
            # Convert sentiment to CSS class
            sentiment_class = f"sentiment-{sentiment.lower()}"
            
            # Create news card HTML
            news_html = f"""
            <div class="news-card">
                <div class="news-title">{title}</div>
                <div class="news-details">
                    <div class="news-detail-item">
                        <div class="detail-label">Published</div>
                        <div class="detail-value">{published}</div>
                    </div>
                    <div class="news-detail-item">
                        <div class="detail-label">Source</div>
                        <div class="detail-value">{source}</div>
                    </div>
                    <div class="news-detail-item">
                        <div class="detail-label">Category</div>
                        <div class="detail-value">{category}</div>
                    </div>
                    <div class="news-detail-item">
                        <div class="detail-label">Impact</div>
                        <div class="detail-value {sentiment_class}">{impact}</div>
                    </div>
                </div>
                <div class="news-summary">{summary}</div>
            </div>
            """
            st.markdown(news_html, unsafe_allow_html=True)

    def _create_technical_charts(self, selected_stock: pd.DataFrame, technical: dict):
        """Create technical analysis charts"""
        st.markdown("### Technical Analysis")
        
        # Create tabs for different technical views
        tech_tabs = st.tabs(["Price & Volume", "Momentum", "Trend"])
        
        with tech_tabs[0]:
            # Price and Volume Chart
            fig = go.Figure()
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=selected_stock.index,
                open=selected_stock['Open'],
                high=selected_stock['High'],
                low=selected_stock['Low'],
                close=selected_stock['Close'],
                name="OHLC"
            ))
            
            # Volume bars
            fig.add_trace(go.Bar(
                x=selected_stock.index,
                y=selected_stock['Volume'],
                name="Volume",
                yaxis="y2",
                marker_color='rgba(0,204,150,0.3)'
            ))
            
            # Add Bollinger Bands if available
            if 'BB_High' in technical and 'BB_Low' in technical:
                fig.add_trace(go.Scatter(
                    x=selected_stock.index,
                    y=technical['BB_High'],
                    name="BB Upper",
                    line=dict(color='rgba(255,255,255,0.5)')
                ))
                
                fig.add_trace(go.Scatter(
                    x=selected_stock.index,
                    y=technical['BB_Low'],
                    name="BB Lower",
                    line=dict(color='rgba(255,255,255,0.5)'),
                    fill='tonexty'
                ))
            
            fig.update_layout(
                yaxis2=dict(
                    title="Volume",
                    overlaying="y",
                    side="right"
                ),
                height=600,
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tech_tabs[1]:
            # Momentum Indicators
            fig = go.Figure()
            
            # RSI
            if 'RSI' in technical:
                fig.add_trace(go.Scatter(
                    x=selected_stock.index,
                    y=technical['RSI'],
                    name="RSI"
                ))
                
                # Add RSI zones
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            
            fig.update_layout(
                height=300,
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)'
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # MACD
            fig = go.Figure()
            
            if all(k in technical for k in ['MACD', 'Signal']):
                fig.add_trace(go.Scatter(
                    x=selected_stock.index,
                    y=technical['MACD'],
                    name="MACD"
                ))
                
                fig.add_trace(go.Scatter(
                    x=selected_stock.index,
                    y=technical['Signal'],
                    name="Signal"
                ))
                
                if 'MACD_Hist' in technical:
                    fig.add_trace(go.Bar(
                        x=selected_stock.index,
                        y=technical['MACD_Hist'],
                        name="Histogram"
                    ))
            
            fig.update_layout(
                height=300,
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)'
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tech_tabs[2]:
            # Moving Averages
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=selected_stock.index,
                y=selected_stock['Close'],
                name="Price"
            ))
            
            if 'SMA_20' in technical:
                fig.add_trace(go.Scatter(
                    x=selected_stock.index,
                    y=technical['SMA_20'],
                    name="SMA 20"
                ))
            
            if 'SMA_50' in technical:
                fig.add_trace(go.Scatter(
                    x=selected_stock.index,
                    y=technical['SMA_50'],
                    name="SMA 50"
                ))
            
            fig.update_layout(
                height=600,
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)'
                )
            )
            st.plotly_chart(fig, use_container_width=True)

    def _fetch_news(self, symbol: str) -> List[Dict]:
        """Fetch news articles for the given symbol."""
        try:
            # Get news from yfinance
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return [{'title': 'No news available', 'published': '', 'summary': 'No recent news found for this stock.'}]
            
            processed_news = []
            for article in news[:3]:  # Only process top 3 articles
                try:
                    # Convert timestamp to datetime
                    published_date = datetime.fromtimestamp(article.get('providerPublishTime', 0))
                    
                    # Get summary and truncate if too long
                    summary = article.get('summary', '')
                    if len(summary) > 200:
                        summary = summary[:200] + '...'
                    
                    news_item = {
                        'title': article.get('title', 'No title available'),
                        'publisher': article.get('publisher', 'Unknown'),
                        'link': article.get('link', '#'),
                        'published': published_date.strftime('%Y-%m-%d %H:%M'),
                        'summary': summary if summary else 'No summary available'
                    }
                    processed_news.append(news_item)
                except Exception as e:
                    print(f"Error processing news article: {str(e)}")
                    continue
            
            return processed_news  # Will return at most 3 items
            
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return [{'title': 'Error fetching news', 'published': '', 'summary': f'Error: {str(e)}'}]

if __name__ == "__main__":
    app = StockAnalysisApp()
    app.main() 