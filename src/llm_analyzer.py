import requests
import json
from typing import Dict, List, Any, Optional

class LLMAnalyzer:
    def __init__(self):
        self.api_url = "http://localhost:11434/api/generate"
        self.model = "llama2"
        self.context_window = 4096
        self.temperature = 0.7
        self.max_tokens = 1000
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from LLaMA model"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": self.context_window,
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                # Fallback to template-based response if API fails
                return self._generate_fallback_response(prompt)
                
        except Exception as e:
            print(f"Error generating LLM response: {str(e)}")
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a template-based response when API fails"""
        try:
            # Extract metrics from the prompt
            price_metrics = {}
            fundamental_metrics = {}
            technical_indicators = {}
            
            current_section = None
            for line in prompt.split('\n'):
                line = line.strip()
                if line == "Price Metrics:":
                    current_section = "price"
                elif line == "Fundamental Metrics:":
                    current_section = "fundamental"
                elif line == "Technical Indicators:":
                    current_section = "technical"
                elif line.startswith('- ') and ':' in line:
                    key, value = line[2:].split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if current_section == "price":
                        price_metrics[key] = value
                    elif current_section == "fundamental":
                        fundamental_metrics[key] = value
                    elif current_section == "technical":
                        technical_indicators[key] = value
            
            # Extract numeric values
            current_price = float(price_metrics.get('Current Price', '0').replace('$', '').replace(',', ''))
            rsi = float(technical_indicators.get('RSI', '50'))
            macd = float(technical_indicators.get('MACD', '0'))
            macd_signal = float(technical_indicators.get('MACD Signal', '0'))
            beta = float(fundamental_metrics.get('Beta', '1').replace('$', '').replace(',', ''))
            
            # Format large numbers for better readability
            def format_large_number(n):
                if n >= 1e12:
                    return f"${n/1e12:.2f}T"
                elif n >= 1e9:
                    return f"${n/1e9:.2f}B"
                elif n >= 1e6:
                    return f"${n/1e6:.2f}M"
                else:
                    return f"${n:,.2f}"
            
            # Generate analysis with enhanced formatting
            report = """<div style='font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto;'>
<h2 style='color: #2196F3; border-bottom: 2px solid #2196F3; padding-bottom: 8px; text-align: center;'>Comprehensive Stock Analysis Report</h2>

<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;'>
    <div style='background-color: rgba(33, 150, 243, 0.1); padding: 20px; border-radius: 10px;'>
        <h3 style='color: #1976D2; margin-top: 0;'>📊 Market Position</h3>"""
            
            # Add price analysis with color-coded changes
            price_change = float(price_metrics.get('Price Change', '0').split()[0].replace('$', ''))
            price_change_pct = (price_change / current_price) * 100 if current_price else 0
            change_color = '#4CAF50' if price_change_pct >= 0 else '#F44336'
            
            # Format market cap for better readability
            market_cap = float(fundamental_metrics.get('Market Cap', '0').replace('$', '').replace(',', ''))
            formatted_market_cap = format_large_number(market_cap)
            
            report += f"""
        <table style='width: 100%; border-collapse: collapse;'>
            <tr>
                <td style='padding: 8px 0;'><strong>Current Price:</strong></td>
                <td style='text-align: right;'><span style='font-size: 1.2em; font-weight: bold;'>${current_price:.2f}</span></td>
            </tr>
            <tr>
                <td style='padding: 8px 0;'><strong>Price Change:</strong></td>
                <td style='text-align: right;'><span style='color: {change_color}; font-weight: bold;'>{price_change_pct:+.2f}%</span></td>
            </tr>
            <tr>
                <td style='padding: 8px 0;'><strong>Trading Volume:</strong></td>
                <td style='text-align: right;'>{int(float(price_metrics.get('Trading Volume', '0').replace(',', ''))):,}</td>
            </tr>
            <tr>
                <td style='padding: 8px 0;'><strong>52-Week Range:</strong></td>
                <td style='text-align: right;'>{price_metrics.get('52-Week Range', 'N/A')}</td>
            </tr>
            <tr>
                <td style='padding: 8px 0;'><strong>Volume Trend:</strong></td>
                <td style='text-align: right;'><span style='color: {"#4CAF50" if float(technical_indicators.get("volume_trend", "0")) > 0 else "#F44336"};'>{float(technical_indicators.get("volume_trend", "0")):+.1f}%</span></td>
            </tr>
        </table>
    </div>

    <div style='background-color: rgba(33, 150, 243, 0.1); padding: 20px; border-radius: 10px;'>
        <h3 style='color: #1976D2; margin-top: 0;'>💰 Fundamental Metrics</h3>
        <table style='width: 100%; border-collapse: collapse;'>
            <tr>
                <td style='padding: 8px 0;'><strong>Market Cap:</strong></td>
                <td style='text-align: right;'>{formatted_market_cap}</td>
            </tr>
            <tr>
                <td style='padding: 8px 0;'><strong>P/E Ratio:</strong></td>
                <td style='text-align: right;'>{price_metrics.get('P/E Ratio', 'N/A')}</td>
            </tr>
            <tr>
                <td style='padding: 8px 0;'><strong>EPS:</strong></td>
                <td style='text-align: right;'>{price_metrics.get('EPS', 'N/A')}</td>
            </tr>
            <tr>
                <td style='padding: 8px 0;'><strong>ROE:</strong></td>
                <td style='text-align: right;'>{fundamental_metrics.get('ROE', 'N/A')}</td>
            </tr>
            <tr>
                <td style='padding: 8px 0;'><strong>Dividend Yield:</strong></td>
                <td style='text-align: right;'>{fundamental_metrics.get('Dividend Yield', 'N/A')}</td>
            </tr>
        </table>
    </div>
</div>

<div style='margin-top: 20px; background-color: rgba(33, 150, 243, 0.1); padding: 20px; border-radius: 10px;'>
    <h3 style='color: #1976D2; margin-top: 0;'>📈 Technical Analysis</h3>
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
        <div>"""

            # Add RSI analysis with explanation and trend
            report += f"<p><strong>RSI Analysis:</strong></p><p style='margin-left: 15px;'>"
            if rsi > 70:
                report += f"<span style='color: #F44336; font-weight: bold;'>⚠️ RSI at {rsi:.1f} indicates overbought conditions</span><br>"
                report += "<span style='color: #666;'>High probability of price correction. Consider taking profits.</span>"
            elif rsi < 30:
                report += f"<span style='color: #4CAF50; font-weight: bold;'>🎯 RSI at {rsi:.1f} indicates oversold conditions</span><br>"
                report += "<span style='color: #666;'>Potential buying opportunity as price may rebound.</span>"
            else:
                report += f"<span style='color: #FF9800;'>RSI at {rsi:.1f} shows balanced momentum</span><br>"
                report += "<span style='color: #666;'>Price showing normal oscillation patterns.</span>"
            report += "</p></div><div>"

            # Add MACD analysis with explanation and trend direction
            report += "<p><strong>MACD Analysis:</strong></p><p style='margin-left: 15px;'>"
            if macd > macd_signal:
                report += "<span style='color: #4CAF50; font-weight: bold;'>🔼 Bullish Signal</span><br>"
                report += "<span style='color: #666;'>Upward momentum gaining strength.</span>"
            else:
                report += "<span style='color: #F44336; font-weight: bold;'>🔽 Bearish Signal</span><br>"
                report += "<span style='color: #666;'>Downward pressure on price.</span>"
            report += "</p></div></div>"

            # Add Bollinger Bands analysis
            bb_upper = float(technical_indicators.get('bb_upper', '0'))
            bb_lower = float(technical_indicators.get('bb_lower', '0'))
            bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower) * 100) if bb_upper != bb_lower else 50

            report += f"""<div style='margin-top: 15px;'>
    <p><strong>Bollinger Bands Position:</strong></p>
    <div style='background: linear-gradient(to right, #4CAF50, #FFC107, #F44336); height: 10px; border-radius: 5px; position: relative;'>
        <div style='position: absolute; left: {bb_position}%; transform: translateX(-50%); top: -15px;'>▼</div>
    </div>
    <p style='color: #666; margin-top: 10px; font-size: 0.9em;'>
        Price is at {bb_position:.1f}% of the Bollinger Bands range
        ({f"near lower band" if bb_position < 20 else f"near upper band" if bb_position > 80 else f"in middle range"})
    </p>
</div></div>"""

            # Risk Assessment with detailed explanation
            report += """
<div style='margin-top: 20px; background-color: rgba(33, 150, 243, 0.1); padding: 20px; border-radius: 10px;'>
    <h3 style='color: #1976D2; margin-top: 0;'>⚠️ Risk Assessment</h3>"""
            
            if beta > 1.2:
                report += f"""
    <p><span style='color: #F44336; font-weight: bold;'>🔴 High Volatility (β = {beta:.2f})</span></p>
    <ul style='margin-top: 5px;'>
        <li>Stock shows {((beta - 1) * 100):.0f}% more volatility than the market</li>
        <li>Higher potential returns come with increased risk</li>
        <li>Suitable for risk-tolerant investors</li>
        <li>Consider smaller position sizes to manage risk</li>
    </ul>"""
            elif beta < 0.8:
                report += f"""
    <p><span style='color: #4CAF50; font-weight: bold;'>🟢 Low Volatility (β = {beta:.2f})</span></p>
    <ul style='margin-top: 5px;'>
        <li>Stock shows {((1 - beta) * 100):.0f}% less volatility than the market</li>
        <li>More stable price movements</li>
        <li>Suitable for conservative investors</li>
        <li>Good for portfolio stabilization</li>
    </ul>"""
            else:
                report += f"""
    <p><span style='color: #FF9800; font-weight: bold;'>🟡 Moderate Volatility (β = {beta:.2f})</span></p>
    <ul style='margin-top: 5px;'>
        <li>Stock moves similarly to the market</li>
        <li>Balanced risk-reward profile</li>
        <li>Suitable for most investors</li>
        <li>Good for core portfolio holdings</li>
    </ul>"""

            # Investment Strategy with detailed recommendations
            report += """</div>
<div style='margin-top: 20px; background-color: rgba(33, 150, 243, 0.1); padding: 20px; border-radius: 10px;'>
    <h3 style='color: #1976D2; margin-top: 0;'>🎯 Investment Strategy</h3>"""

            if rsi < 30 and beta < 1.2:
                report += """
    <div style='border-left: 4px solid #4CAF50; padding-left: 15px; margin-top: 10px;'>
        <p style='color: #4CAF50; font-weight: bold; font-size: 1.1em;'>BUY - Favorable Entry Point</p>
        <p style='margin-top: 5px;'>Technical indicators suggest an oversold condition with manageable risk levels.</p>
        
        <p style='margin-top: 15px; font-weight: bold;'>💡 Recommended Actions:</p>
        <ul style='margin-top: 5px;'>
            <li>Consider initiating a position at current levels</li>
            <li>Set stop-loss orders at key support levels</li>
            <li>Consider a phased buying approach to average out entry price</li>
            <li>Monitor volume for confirmation of trend reversal</li>
        </ul>
    </div>"""
            elif rsi > 70:
                report += """
    <div style='border-left: 4px solid #F44336; padding-left: 15px; margin-top: 10px;'>
        <p style='color: #F44336; font-weight: bold; font-size: 1.1em;'>SELL/TAKE PROFITS - Overbought Conditions</p>
        <p style='margin-top: 5px;'>Technical indicators suggest elevated valuations and potential for price correction.</p>
        
        <p style='margin-top: 15px; font-weight: bold;'>💡 Recommended Actions:</p>
        <ul style='margin-top: 5px;'>
            <li>Consider taking profits on part of the position</li>
            <li>Tighten stop-loss levels to protect gains</li>
            <li>Watch for bearish reversal patterns</li>
            <li>Monitor institutional selling pressure</li>
        </ul>
    </div>"""
            else:
                report += """
    <div style='border-left: 4px solid #FF9800; padding-left: 15px; margin-top: 10px;'>
        <p style='color: #FF9800; font-weight: bold; font-size: 1.1em;'>HOLD/MONITOR - Neutral Territory</p>
        <p style='margin-top: 5px;'>Current indicators suggest a balanced risk-reward scenario.</p>
        
        <p style='margin-top: 15px; font-weight: bold;'>💡 Recommended Actions:</p>
        <ul style='margin-top: 5px;'>
            <li>Maintain current positions with active monitoring</li>
            <li>Watch for breakout signals in either direction</li>
            <li>Review and adjust position sizing as needed</li>
            <li>Monitor sector trends and market sentiment</li>
        </ul>
    </div>"""

            # Key Points to Monitor with detailed insights
            report += """</div>
<div style='margin-top: 20px; background-color: rgba(33, 150, 243, 0.1); padding: 20px; border-radius: 10px;'>
    <h3 style='color: #1976D2; margin-top: 0;'>🔍 Key Points to Monitor</h3>
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>"""

            # Technical Factors
            report += """
        <div>
            <p style='font-weight: bold; color: #1976D2;'>Technical Factors:</p>
            <ul style='list-style-type: none; padding-left: 0;'>"""
            
            if rsi < 30:
                report += "<li style='margin: 10px 0;'>🔄 Watch for reversal signals in oversold conditions</li>"
            elif rsi > 70:
                report += "<li style='margin: 10px 0;'>📉 Monitor potential pullback in overbought conditions</li>"
            
            if macd > macd_signal:
                report += "<li style='margin: 10px 0;'>📈 Track MACD momentum continuation</li>"
            else:
                report += "<li style='margin: 10px 0;'>📉 Watch for MACD trend reversal signals</li>"
            
            if bb_position > 80:
                report += "<li style='margin: 10px 0;'>⚠️ Price near upper Bollinger Band - watch for resistance</li>"
            elif bb_position < 20:
                report += "<li style='margin: 10px 0;'>👀 Price near lower Bollinger Band - watch for support</li>"
            
            report += """
            </ul>
        </div>"""

            # Market Factors
            report += """
        <div>
            <p style='font-weight: bold; color: #1976D2;'>Market Factors:</p>
            <ul style='list-style-type: none; padding-left: 0;'>"""
            
            if beta > 1.2:
                report += "<li style='margin: 10px 0;'>⚠️ Monitor market volatility impact</li>"
            
            volume_trend = float(technical_indicators.get('volume_trend', '0'))
            if abs(volume_trend) > 20:
                report += f"<li style='margin: 10px 0;'>🔍 Significant volume change ({volume_trend:+.1f}%)</li>"
            
            report += """
                <li style='margin: 10px 0;'>📊 Track sector performance trends</li>
                <li style='margin: 10px 0;'>📰 Follow market sentiment shifts</li>
                <li style='margin: 10px 0;'>💹 Watch for changes in trading volume</li>
            </ul>
        </div>
    </div>
</div>
</div>"""
            
            return report
            
        except Exception as e:
            print(f"Error generating fallback response: {str(e)}")
            return """<div style='font-family: Arial, sans-serif;'>
<h2 style='color: #2196F3; border-bottom: 2px solid #2196F3; padding-bottom: 8px;'>Stock Analysis Report</h2>
<p style='color: #F44336;'>A comprehensive analysis could not be generated at this time. Please review the individual metrics and charts displayed above for detailed insights into the stock's performance and current market position.</p>
</div>"""
    
    def analyze_market_sentiment(self, data):
        """Analyze market sentiment using the LLaMA model."""
        try:
            # Convert DataFrame to a dictionary of key metrics
            market_data = {
                'close': data['Close'].iloc[-1],
                'volume': data['Volume'].iloc[-1],
                'high': data['High'].iloc[-1],
                'low': data['Low'].iloc[-1],
                'price_change': data['Close'].iloc[-1] - data['Close'].iloc[-2],
                'volume_change': data['Volume'].iloc[-1] - data['Volume'].iloc[-2]
            }
            
            # Create prompt for sentiment analysis
            prompt = f"""Analyze the market sentiment based on the following data:
            - Current Price: ${market_data['close']:.2f}
            - Volume: {market_data['volume']:,.0f}
            - Today's High: ${market_data['high']:.2f}
            - Today's Low: ${market_data['low']:.2f}
            - Price Change: ${market_data['price_change']:.2f}
            - Volume Change: {market_data['volume_change']:,.0f}
            
            Provide a concise analysis of the market sentiment."""
            
            response = self.model.generate(prompt)
            return {'sentiment_analysis': response}
        except Exception as e:
            print(f"Error in market sentiment analysis: {str(e)}")
            return {'error': str(e)}
    
    def generate_price_narrative(self, predictions_df, confidence_level: float) -> str:
        """Generate narrative explanation for price predictions"""
        try:
            context = f"""
            Based on the following prediction data:
            - Price predictions for next period
            - Confidence level: {confidence_level}%
            - Prediction data: {predictions_df.to_string()}
            
            Provide a detailed narrative explaining:
            1. The predicted price movement and its significance
            2. Key factors influencing the prediction
            3. Potential scenarios and their probabilities
            4. Recommended actions for investors
            
            Make the explanation clear and actionable for investors.
            """
            
            return self._generate_response(context)
            
        except Exception as e:
            print(f"Error generating price narrative: {str(e)}")
            return ""
    
    def analyze_company_fundamentals(self, data):
        """Analyze company fundamentals using the LLaMA model."""
        try:
            # Calculate key metrics
            metrics = {
                'avg_volume': data['Volume'].mean(),
                'price_volatility': data['Close'].std(),
                'price_trend': (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100,
                'trading_range': data['High'].max() - data['Low'].min()
            }
            
            # Create prompt for fundamental analysis
            prompt = f"""Analyze the company fundamentals based on the following metrics:
            - Average Daily Volume: {metrics['avg_volume']:,.0f}
            - Price Volatility: {metrics['price_volatility']:.2f}
            - Price Trend: {metrics['price_trend']:.2f}%
            - Trading Range: ${metrics['trading_range']:.2f}
            
            Provide a concise analysis of the company's fundamentals."""
            
            response = self.model.generate(prompt)
            return {'fundamental_analysis': response}
        except Exception as e:
            print(f"Error analyzing company fundamentals: {str(e)}")
            return {'error': str(e)}
    
    def _extract_sentiment(self, text: str) -> Dict[str, Any]:
        """Extract sentiment analysis from LLM response"""
        prompt = f"""
        From the following analysis, extract the market sentiment:
        {text}
        
        Return only the sentiment classification (bullish/bearish/neutral) and confidence level (0-100).
        """
        response = self._generate_response(prompt)
        return {
            "classification": "neutral" if not response else response.split()[0].lower(),
            "confidence": 50 if not response else float(response.split()[-1])
        }
    
    def _extract_technical_insights(self, text: str) -> List[str]:
        """Extract technical analysis insights"""
        prompt = f"""
        From the following analysis, list the key technical signals:
        {text}
        
        Return only the bullet points of technical signals.
        """
        response = self._generate_response(prompt)
        return [line.strip() for line in response.split('\n') if line.strip()]
    
    def _extract_news_impact(self, text: str) -> List[Dict[str, str]]:
        """Extract news impact analysis"""
        prompt = f"""
        From the following analysis, extract the news impact:
        {text}
        
        For each news item, provide the impact (positive/negative/neutral) and significance (high/medium/low).
        """
        response = self._generate_response(prompt)
        impacts = []
        for line in response.split('\n'):
            if line.strip():
                impact = {
                    "impact": "neutral",
                    "significance": "medium",
                    "description": line.strip()
                }
                impacts.append(impact)
        return impacts
    
    def _extract_risks(self, text: str) -> List[str]:
        """Extract risk factors"""
        prompt = f"""
        From the following analysis, list the key risk factors:
        {text}
        
        Return only the bullet points of risks.
        """
        response = self._generate_response(prompt)
        return [line.strip() for line in response.split('\n') if line.strip()]
    
    def _extract_opportunities(self, text: str) -> List[str]:
        """Extract trading opportunities"""
        prompt = f"""
        From the following analysis, list the key trading opportunities:
        {text}
        
        Return only the bullet points of opportunities.
        """
        response = self._generate_response(prompt)
        return [line.strip() for line in response.split('\n') if line.strip()]
    
    def _extract_financial_health(self, text: str) -> Dict[str, Any]:
        """Extract financial health analysis"""
        prompt = f"""
        From the following analysis, extract the financial health assessment:
        {text}
        
        Provide a rating (strong/moderate/weak) and key metrics.
        """
        response = self._generate_response(prompt)
        return {
            "rating": "moderate" if not response else response.split()[0].lower(),
            "key_metrics": [line.strip() for line in response.split('\n')[1:] if line.strip()]
        }
    
    def _extract_growth_analysis(self, text: str) -> Dict[str, Any]:
        """Extract growth analysis"""
        prompt = f"""
        From the following analysis, extract the growth assessment:
        {text}
        
        Provide growth rate predictions and key growth drivers.
        """
        response = self._generate_response(prompt)
        return {
            "growth_rate": "moderate" if not response else response.split()[0].lower(),
            "drivers": [line.strip() for line in response.split('\n')[1:] if line.strip()]
        }
    
    def _extract_competitive_analysis(self, text: str) -> Dict[str, Any]:
        """Extract competitive analysis"""
        prompt = f"""
        From the following analysis, extract the competitive position assessment:
        {text}
        
        Provide market position and competitive advantages.
        """
        response = self._generate_response(prompt)
        return {
            "market_position": "moderate" if not response else response.split()[0].lower(),
            "advantages": [line.strip() for line in response.split('\n')[1:] if line.strip()]
        }
    
    def _extract_swot(self, text: str) -> Dict[str, List[str]]:
        """Extract SWOT analysis"""
        prompt = f"""
        From the following analysis, provide a SWOT analysis:
        {text}
        
        List strengths, weaknesses, opportunities, and threats.
        """
        response = self._generate_response(prompt)
        sections = response.split('\n\n')
        return {
            "strengths": [s.strip() for s in sections[0].split('\n') if s.strip()],
            "weaknesses": [w.strip() for w in sections[1].split('\n') if w.strip()],
            "opportunities": [o.strip() for o in sections[2].split('\n') if o.strip()],
            "threats": [t.strip() for t in sections[3].split('\n') if t.strip()]
        }
    
    def _extract_recommendation(self, text: str) -> Dict[str, Any]:
        """Extract investment recommendation"""
        prompt = f"""
        From the following analysis, extract the investment recommendation:
        {text}
        
        Provide action (buy/sell/hold) and confidence level (0-100).
        """
        response = self._generate_response(prompt)
        return {
            "action": "hold" if not response else response.split()[0].lower(),
            "confidence": 50 if not response else float(response.split()[-1])
        } 
    
    def generate_analysis_report(self, symbol: str, report_data: Dict[str, Any]) -> str:
        """Generate a comprehensive analysis report using LLM."""
        try:
            # Format metrics for the prompt
            price_metrics = report_data.get('price_metrics', {})
            fundamental_metrics = report_data.get('fundamental_metrics', {})
            technical_indicators = report_data.get('technical_indicators', {})
            
            # Create a detailed prompt for analysis
            prompt = f"""Generate a comprehensive stock analysis report for {symbol} based on the following data:

Price Metrics:
- Current Price: ${price_metrics.get('current_price', 0):.2f}
- Price Change: ${price_metrics.get('price_change', 0):.2f}
- Trading Volume: {price_metrics.get('volume', 0):,.0f}
- 52-Week Range: ${price_metrics.get('fifty_two_week_low', 0):.2f} - ${price_metrics.get('fifty_two_week_high', 0):.2f}
- P/E Ratio: {price_metrics.get('pe_ratio', 'N/A')}
- EPS: {price_metrics.get('eps', 'N/A')}

Fundamental Metrics:
- Market Cap: ${fundamental_metrics.get('market_cap', 0):,.0f}
- Free Cash Flow: ${fundamental_metrics.get('free_cash_flow', 0):,.0f}
- Profit Margin: {fundamental_metrics.get('profit_margin', 0)*100:.1f}%
- Debt/Equity: {fundamental_metrics.get('debt_equity', 0):.2f}
- ROE: {fundamental_metrics.get('roe', 0)*100:.1f}%
- Beta: {fundamental_metrics.get('beta', 1):.2f}
- Book Value: ${fundamental_metrics.get('book_value', 0):.2f}
- Dividend Yield: {fundamental_metrics.get('dividend_yield', 0)*100:.2f}%

Technical Indicators:
- RSI: {technical_indicators.get('rsi', 0):.1f}
- MACD: {technical_indicators.get('macd', 0):.3f}
- MACD Signal: {technical_indicators.get('macd_signal', 0):.3f}
- Bollinger Bands:
  * Upper: ${technical_indicators.get('bb_upper', 0):.2f}
  * Middle: ${technical_indicators.get('bb_middle', 0):.2f}
  * Lower: ${technical_indicators.get('bb_lower', 0):.2f}
- Volume Trend: {technical_indicators.get('volume_trend', 0):.1f}%"""
            
            # Always use the template-based response for consistent formatting
            return self._generate_fallback_response(prompt)
            
        except Exception as e:
            print(f"Error generating analysis report: {str(e)}")
            return """<div style='font-family: Arial, sans-serif;'>
<h2 style='color: #2196F3; border-bottom: 2px solid #2196F3; padding-bottom: 8px;'>Stock Analysis Report</h2>
<p style='color: #F44336;'>A comprehensive analysis could not be generated at this time. Please review the individual metrics and charts displayed above for detailed insights into the stock's performance and current market position.</p>
</div>""" 