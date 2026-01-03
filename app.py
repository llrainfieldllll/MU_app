import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Quant-Audit: Sigma Scanner", layout="wide")
st.title("🛡️ Quant Auditor: Statistical Extension")

ticker = st.text_input("Enter Ticker Symbol", "MU").upper()

if st.button("Run Senior Audit"):
    try:
        with st.spinner(f'Auditing {ticker}...'):
            # Data Fetching
            data = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if data.empty: st.stop()
            
            # MultiIndex Fix 
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            prices = data['Close'].tail(34) # Extra for RSI buffer
            analysis_prices = prices.tail(20)

            # Live Price 
            try:
                current_price = float(yf.Ticker(ticker).fast_info['lastPrice'])
            except:
                current_price = float(analysis_prices.iloc[-1])

            # Stats Engine 
            mu, sigma = analysis_prices.mean(), analysis_prices.std()
            z_score = (current_price - mu) / sigma
            p_val = 2 * (1 - norm.cdf(abs(z_score)))

            # RSI Engine
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            current_rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1]

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${current_price:.2f}")
            c2.metric("Z-Score", f"{z_score:.2f}σ")
            c3.metric("RSI", f"{current_rsi:.1f}")
            c4.metric("Prob.", f"{p_val*100:.1f}%")

            # Visualization 
            x = np.linspace(-4, 4, 1000)
            y = norm.pdf(x, 0, 1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='gray')))
            fig.add_vline(x=z_score, line_width=3, line_color="red" if abs(z_score) > 2 else "green")
            
            

            if z_score > 0:
                fig.add_trace(go.Scatter(x=x[x >= z_score], y=y[x >= z_score], fill='tozeroy', fillcolor='rgba(255,0,0,0.5)'))
            else:
                fig.add_trace(go.Scatter(x=x[x <= z_score], y=y[x <= z_score], fill='tozeroy', fillcolor='rgba(255,0,0,0.5)'))

            fig.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")