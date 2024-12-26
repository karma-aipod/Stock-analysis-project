import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_stock_chart(stock_data, chart_type="stock_price"):
    """Display interactive stock charts"""
    df = stock_data["historical_data"]
    
    if chart_type == "stock_price":
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            )
        )
        
        # Add volume bar chart
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                yaxis='y2'
            )
        )
        
        fig.update_layout(
            title='Stock Price & Volume',
            yaxis_title='Price',
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right'
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True) 