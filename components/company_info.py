import streamlit as st

def display_company_info(stock_data):
    """Display company header information"""
    info = stock_data["info"]
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Display company logo if available
        if "logo_url" in info:
            st.image(info["logo_url"], width=100)
    
    with col2:
        st.subheader(f"{info['longName']} ({stock_data['symbol']})")
        st.write(f"Sector: {info.get('sector', 'N/A')}")
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
        with metrics_col2:
            st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,.0f}")
        with metrics_col3:
            st.metric("P/E Ratio", info.get('trailingPE', 'N/A')) 