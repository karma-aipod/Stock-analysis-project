import streamlit as st
from src.data_fetcher import StockDataFetcher
from src.stock_analyzer import StockAnalyzer
from src.ml_predictor import StockPredictor
from src.ui_components import UIComponents
from src.utils.stock_lists import get_nifty50_stocks, get_sensex_stocks, get_default_stocks

st.set_page_config(page_title="Global Stock Market Analysis", layout="wide")

class StockAnalysisApp:
    def __init__(self):
        self.ui = UIComponents()
        self.data_fetcher = StockDataFetcher()
        self.analyzer = StockAnalyzer()
        self.predictor = StockPredictor()
        self.ui.apply_custom_styling()
        
        # Add custom CSS for navigation buttons
        st.markdown("""
        <style>
        .nav-button {
            display: block;
            width: 100%;
            padding: 10px 16px;
            margin: 8px 0;
            background-color: rgba(91, 111, 155, 0.5);
            color: white;
            border-radius: 4px;
            text-decoration: none;
            text-align: left;
            cursor: pointer;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: background-color 0.3s;
        }
        .nav-button:hover {
            background-color: rgba(91, 111, 155, 0.8);
        }
        .nav-button.active {
            background-color: rgba(0, 204, 150, 0.5);
            border-color: rgba(0, 204, 150, 0.8);
        }
        .nav-container {
            padding: 10px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .main-content {
            margin-left: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def main(self):
        """Main application logic"""
        st.title("Stock Market Analysis")
        
        # Create two columns - one for navigation and one for content
        nav_col, content_col = st.columns([1, 4])
        
        with nav_col:
            st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
            pages = {
                "Company Analysis": "📊",
                "Market Analysis": "📈",
                "Portfolio Analysis": "💼",
                "AI Predictions": "🤖"
            }
            
            selected_page = None
            for page, icon in pages.items():
                if st.button(f"{icon} {page}", key=f"nav_{page}", help=f"Go to {page}"):
                    selected_page = page
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with content_col:
            st.markdown("<div class='main-content'>", unsafe_allow_html=True)
            
            # Get stock lists
            stocks = get_nifty50_stocks()
            
            # Create stock selector
            selected_stock = self.ui.create_stock_selector(stocks)
            symbol = stocks[selected_stock]
            
            try:
                # Fetch stock data
                stock_info = self.data_fetcher.get_stock_info(symbol)
                
                # Show selected page content
                if selected_page == "Market Analysis":
                    self._show_market_analysis(selected_stock, symbol, stock_info)
                elif selected_page == "Portfolio Analysis":
                    self._show_portfolio_analysis()
                elif selected_page == "AI Predictions":
                    self._show_ai_analysis(symbol)
                else:  # Default to Company Analysis
                    self._show_company_analysis(selected_stock, symbol, stock_info)
                    
            except Exception as e:
                st.error(f"Error loading data for {symbol}: {str(e)}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def _show_company_analysis(self, selected_stock: str, symbol: str, stock_info: dict):
        """Display company analysis section"""
        # Company Overview
        st.markdown("### Company Overview")
        st.markdown(stock_info.get('longBusinessSummary', 'No company description available.'))
        
        # Key Metrics
        self.ui.display_key_metrics(stock_info)
        
        # Stock Chart
        st.markdown("### Stock Performance")
        col1, col2 = st.columns([3, 1])
        with col2:
            period = st.selectbox("Time Period", ['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=3)
            chart_type = st.selectbox("Chart Type", ['Line', 'Candlestick'])
            indicators = st.multiselect("Indicators", ['MA50', 'MA200', 'BB'])
        
        with col1:
            fig = self.analyzer.plot_stock_chart(symbol, period, chart_type, indicators)
            st.plotly_chart(fig, use_container_width=True)
        
        # Financial Performance
        st.markdown("### Financial Performance")
        chart_type = st.selectbox("Chart Type", ["Bar and Line", "Spider", "Area", "Scatter"], key="fin_chart_type")
        fig = self.analyzer.plot_financial_trends(symbol, chart_type)
        st.plotly_chart(fig, use_container_width=True)
        
        # Valuation Comparison
        st.markdown("### Valuation Comparison")
        chart_type = st.selectbox("Chart Type", ["Bar", "Spider"], key="val_chart_type")
        valuation_fig = self.analyzer.plot_valuation_comparison(symbol, chart_type)
        st.plotly_chart(valuation_fig, use_container_width=True)
        
        # Technical Indicators
        st.markdown("### Technical Indicators")
        self.ui.display_technical_indicators(stock_info)
    
    def _show_market_analysis(self, selected_stock: str, symbol: str, stock_info: dict):
        """Display market analysis section"""
        st.markdown("### Market Analysis")
        
        # Sector Performance
        st.markdown("#### Sector Performance")
        sector_fig = self.analyzer.plot_sector_performance(stock_info)
        st.plotly_chart(sector_fig, use_container_width=True)
        
        # Market Share Analysis
        st.markdown("#### Market Share Analysis")
        market_share_data = {
            selected_stock: 35,
            "Competitor 1": 25,
            "Competitor 2": 20,
            "Competitor 3": 15,
            "Others": 5
        }
        market_share_fig = self.analyzer.plot_market_share(market_share_data)
        st.plotly_chart(market_share_fig, use_container_width=True)
        
        # Competitor Analysis
        st.markdown("#### Competitor Performance")
        competitor_data = {
            selected_stock: {"Revenue Growth": 15, "Profit Margin": 25, "Market Share": 35},
            "Competitor 1": {"Revenue Growth": 12, "Profit Margin": 20, "Market Share": 25},
            "Competitor 2": {"Revenue Growth": 10, "Profit Margin": 18, "Market Share": 20},
            "Competitor 3": {"Revenue Growth": 8, "Profit Margin": 15, "Market Share": 15}
        }
        competitor_fig = self.analyzer.plot_competitor_performance(competitor_data)
        st.plotly_chart(competitor_fig, use_container_width=True)
    
    def _show_portfolio_analysis(self):
        """Display portfolio analysis section"""
        st.markdown("### Portfolio Analysis")
        
        # Investment Settings
        st.markdown("#### Investment Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            investment_amount = st.number_input("Monthly Investment (₹)", min_value=1000, value=10000, step=1000)
        with col2:
            risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
        with col3:
            investment_horizon = st.selectbox("Investment Horizon", ["1 Year", "3 Years", "5 Years", "10 Years"])
        
        # Generate portfolio recommendation
        portfolio = self.analyzer.generate_portfolio(investment_amount, risk_tolerance, investment_horizon)
        
        # Display allocation
        st.markdown("#### Recommended Allocation")
        col1, col2 = st.columns([2, 1])
        with col1:
            chart_type = st.selectbox("Chart Type", ["Pie", "Treemap", "Bar", "Spider"])
            fig = self.analyzer.plot_custom_portfolio(portfolio["allocation"], chart_type)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Portfolio Metrics")
            metrics = portfolio["metrics"]
            st.markdown(f"""
            - Expected Annual Return: {metrics['Expected Annual Return']:.1f}%
            - Risk Level: {metrics['Risk Level']:.1f}%
            - Investment Horizon: {metrics['Investment Horizon']} years
            - Monthly Investment: ₹{metrics['Monthly Investment']:,.0f}
            - Projected Value: ₹{metrics['Projected Value']:,.0f}
            """)
    
    def _show_ai_analysis(self, symbol: str):
        """Display AI analysis section"""
        st.markdown("### AI Analysis")
        
        # Price Prediction
        st.markdown("#### Price Prediction")
        prediction_days = st.slider("Prediction Days", 7, 30, 14)
        
        with st.spinner("Generating predictions..."):
            predictions = self.predictor.predict_future_prices(symbol, prediction_days)
            if predictions is not None:
                st.plotly_chart(predictions['chart'], use_container_width=True)
                
                # Model Insights
                st.markdown("#### Model Insights")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Trend", predictions['trend'])
                    st.metric("Confidence Score", f"{predictions['confidence']:.1f}%")
                with col2:
                    st.metric("Price Range", f"₹{predictions['price_range'][0]:,.2f} - ₹{predictions['price_range'][1]:,.2f}")
                    st.metric("Volatility Score", f"{predictions['volatility']:.1f}%")
            else:
                st.error("Unable to generate predictions. Please try again later.")

if __name__ == "__main__":
    app = StockAnalysisApp()
    app.main() 