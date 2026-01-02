import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Quant-Audit: Sigma Scanner", layout="centered")
st.title("🛡️ Quant Auditor: Statistical Extension")

# --- INPUT SECTION ---
ticker = st.text_input("Enter Ticker Symbol", "MU").upper()

# --- EXECUTION BUTTON ---
if st.button("Run Senior Audit"):
    try:
        with st.spinner(f'Auditing {ticker} market data...'):
            # 1. ROBUST DATA FETCHING (Fixes "Holiday Bug")
            # Fetch 3 months to ensure we have at least 20 trading days
            data = yf.download(ticker, period="3mo", interval="1d", progress=False)
            
            # Handle YFinance MultiIndex bug (Common in v0.2+)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Slice strictly the last 20 closed sessions
            prices = data['Close'].tail(20)
            
            if len(prices) < 20:
                st.error(f"Insufficient data. Only found {len(prices)} trading days.")
                st.stop()

            # Get Live Price (Real-time)
            stock_info = yf.Ticker(ticker).fast_info
            current_price = float(stock_info['lastPrice'])

        # 2. SENIOR MATH ENGINE
        mu = prices.mean()
        sigma = prices.std()
        
        # Z-Score: (Price - Mean) / Volatility
        z_score = (current_price - mu) / sigma
        
        # PROBABILITY FIX (Two-Tailed Test)
        # We want the probability of being THIS far away (up or down)
        # p_value represents the "rarity" of the event
        p_value = 2 * (1 - norm.cdf(abs(z_score))) 
        rarity_percent = p_value * 100

        # 3. DISPLAY METRICS
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}")
        col2.metric("20-Day Mean", f"${mu:.2f}")
        col3.metric("Sigma (Z-Score)", f"{z_score:.2f}σ", 
                    delta_color="inverse" if abs(z_score) > 2 else "normal")

        # 4. DYNAMIC VERDICT GENERATOR
        if abs(z_score) >= 2.0:
            st.error(f"🚨 EXTREME EVENT: Price is {z_score:.2f} deviations from the mean. "
                     f"The theoretical probability of this is {rarity_percent:.2f}%. "
                     "Statistical mean reversion is highly likely.")
        elif abs(z_score) >= 1.0:
            st.warning(f"⚠️ EXTENDED: Price is trending. {rarity_percent:.2f}% probability zone.")
        else:
            st.success(f"✅ NORMAL: Price is within expected volatility limits.")

        # 5. VISUALIZATION (Normal Curve)
        x = np.linspace(-4, 4, 1000)
        y = norm.pdf(x, 0, 1)
        
        fig = go.Figure()
        
        # The Bell Curve
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Normal Distribution', line=dict(color='gray')))
        
        # The Live Price Line
        fig.add_vline(x=z_score, line_width=3, line_dash="solid", line_color="red" if abs(z_score) > 2 else "green")
        fig.add_annotation(x=z_score, y=0.1, text=f"YOU ARE HERE\n({z_score:.2f}σ)", showarrow=True, arrowhead=1)

        # Highlight the "Tail" (The danger zone)
        # If Positive Z (Rally) -> Shade Right Tail
        if z_score > 0:
            fig.add_trace(go.Scatter(x=x[x >= z_score], y=y[x >= z_score], fill='tozeroy', fillcolor='rgba(255,0,0,0.5)', name='Tail Risk'))
        # If Negative Z (Crash) -> Shade Left Tail
        else:
            fig.add_trace(go.Scatter(x=x[x <= z_score], y=y[x <= z_score], fill='tozeroy', fillcolor='rgba(255,0,0,0.5)', name='Tail Risk'))

        fig.update_layout(title=f"Visualizing the Outlier: {ticker}", template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"System Error: {e}")