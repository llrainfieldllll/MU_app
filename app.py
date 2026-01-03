import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from tenacity import retry, stop_after_attempt, wait_fixed
import logging

# --- CONFIGURATION ---
st.set_page_config(page_title="Quant-Audit: Pro", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- INDUSTRIAL STANDARD DATA FETCHING ---
@st.cache_data(ttl=3600, show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_market_data(ticker: str) -> pd.DataFrame:
    """
    Fetches data using yfinance's internal advanced handling.
    Requires 'curl-cffi' installed in requirements.txt to work on Cloud IPs.
    """
    try:
        # REMOVED: Manual session creation (Option 1).
        # NEW STRATEGY: Let yfinance detect 'curl-cffi' and handle the handshake.
        
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="3mo", interval="1d")
        
        # Validation Checks
        if data.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()

        # Handle Timezone Awareness
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        if len(data) < 34: 
            logger.warning(f"Insufficient data points: {len(data)}")
            return pd.DataFrame()

        return data

    except Exception as e:
        logger.error(f"API Error: {e}")
        raise e

# --- METRICS ENGINE (UNCHANGED) ---
def calculate_metrics(df: pd.DataFrame):
    prices = df['Close']
    analysis_slice = prices.tail(20)
    current_price = float(analysis_slice.iloc[-1])

    mu = analysis_slice.mean()
    sigma = analysis_slice.std()
    
    if sigma == 0:
        z_score = 0
    else:
        z_score = (current_price - mu) / sigma
        
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    
    last_gain = gain.iloc[-1]
    last_loss = loss.iloc[-1]

    if last_loss == 0:
        rsi = 100
    else:
        rs = last_gain / last_loss
        rsi = 100 - (100 / (1 + rs))

    return {
        "price": current_price,
        "mu": mu,
        "z_score": z_score,
        "p_value": p_value,
        "rsi": rsi
    }

# --- MAIN UI ---
def main():
    st.title("🛡️ Quant Auditor: Cloud Edition")
    
    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker Symbol", "MU").upper()
        run_btn = st.button("Run Audit", type="primary")

    if run_btn:
        with st.spinner(f"Securely fetching data for {ticker}..."):
            try:
                # 1. Fetch
                data = fetch_market_data(ticker)
                
                if data.empty:
                    st.error(f"❌ Failed to retrieve data for {ticker}. The API may be blocking cloud traffic.")
                    st.stop()

                # 2. Calculate
                m = calculate_metrics(data)

                # 3. Display
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Price", f"${m['price']:.2f}")
                k2.metric("20-Day Mean", f"${m['mu']:.2f}")
                k3.metric("Z-Score", f"{m['z_score']:.2f}σ", delta_color="inverse" if abs(m['z_score']) > 2 else "normal")
                k4.metric("RSI", f"{m['rsi']:.1f}")

                # 4. Verdict
                st.divider()
                if abs(m['z_score']) >= 2.0:
                    st.error(f"🚨 EXTREME DEVIATION: {m['z_score']:.2f}σ")
                elif m['rsi'] > 70:
                    st.warning("⚠️ OVERBOUGHT SIGNAL")
                elif m['rsi'] < 30:
                    st.warning("⚠️ OVERSOLD SIGNAL")
                else:
                    st.success("✅ NORMAL VOLATILITY")

                # 5. Visuals
                x = np.linspace(-4, 4, 1000)
                y = norm.pdf(x, 0, 1)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#666')))
                
                z = m['z_score']
                fig.add_vline(x=z, line_width=3, line_color="#FF4B4B" if abs(z) > 2 else "#00FF00")
                
                fill_x = x[x >= z] if z > 0 else x[x <= z]
                fill_y = y[x >= z] if z > 0 else y[x <= z]
                fig.add_trace(go.Scatter(x=fill_x, y=fill_y, fill='tozeroy', fillcolor='rgba(255, 75, 75, 0.4)'))

                fig.update_layout(template="plotly_dark", showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"System Failure: {e}")

if __name__ == "__main__":
    main()