import streamlit as st
import yfinance as yf

def display_stock_news(symbol):
    """Display recent news for the stock"""
    ticker = yf.Ticker(symbol)
    news = ticker.news
    
    st.subheader("Recent News")
    
    for article in news[:5]:  # Display last 5 news items
        with st.container():
            st.write(f"**{article['title']}**")
            st.write(f"Source: {article['source']}")
            st.write(f"Published: {article['providerPublishTime']}")
            st.write(article['link'])
            st.markdown("---") 