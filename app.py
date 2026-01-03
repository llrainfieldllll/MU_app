import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
from tenacity import retry, stop_after_attempt, wait_fixed
import logging

# --- CONFIGURATION ---
st.set_page_config(page_title="Quant-Audit: Domain Expert", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ROBUST DATA ENGINE ---
@st.cache_data(ttl=900, show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_market_data(ticker: str) -> pd.DataFrame:
    try:
        # Fetch 6mo to ensure stability. 
        # Using internal yfinance spoofing via curl_cffi
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="6mo", interval="1d")
        
        if data.empty: return pd.DataFrame()
        if data.index.tz is not None: data.index = data.index.tz_localize(None)
        if len(data) < 34: return pd.DataFrame() # Min requirement for RSI + SMA

        return data
    except Exception as e:
        logger.error(f"Data Fetch Error: {e}")
        raise e

# --- DOMAIN KNOWLEDGE ENGINE ---
def calculate_metrics(df: pd.DataFrame):
    prices = df['Close']
    volumes = df['Volume']
    
    # 1. PRICE CONTEXT
    current_price = float(prices.iloc[-1])
    current_volume = float(volumes.iloc[-1])
    
    # 2. STATISTICAL ENGINE (Student's t-Distribution)
    # We use t-dist instead of Normal because stock returns have "Fat Tails"
    analysis_slice = prices.tail(20)
    mu = analysis_slice.mean()    
    sigma = analysis_slice.std()
    
    if sigma == 0:
        z_score = 0
    else:
        z_score = (current_price - mu) / sigma
        
    # Degrees of Freedom (df). 
    # df=5 is the financial standard for daily stock returns (modeling kurtosis).
    df_kurtosis = 5 
    
    # One-Sided P-Value (Fat Tail Adjusted)
    if z_score > 0:
        p_value = 1 - t.cdf(z_score, df=df_kurtosis) 
    else:
        p_value = t.cdf(z_score, df=df_kurtosis)
        
    # 3. VOLUME DYNAMICS (The "Truth" Serum)
    # Compare current volume to the 20-day average volume
    vol_slice = volumes.tail(20)
    avg_volume = vol_slice.mean()
    
    if avg_volume == 0:
        vol_ratio = 1.0 # Safety for low-float/halted stocks
    else:
        vol_ratio = current_volume / avg_volume

    # 4. RSI (Simple)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    last_loss = loss.iloc[-1]
    
    if last_loss == 0:
        rsi = 100
    else:
        rs = gain.iloc[-1] / last_loss
        rsi = 100 - (100 / (1 + rs))

    return {
        "price": current_price,
        "mu": mu,
        "z_score": z_score,
        "p_value": p_value,
        "price_adjustment": mu - current_price,
        "rsi": rsi,
        "vol_ratio": vol_ratio,
        "vol_avg": avg_volume
    }

# --- UI RENDERER ---
def main():
    st.title("🛡️ Quant Auditor: Domain Expert Edition")
    st.markdown("#### t-Distribution & Volume Dynamics Scanner")
    
    with st.sidebar:
        st.header("Search")
        ticker = st.text_input("Ticker Symbol", "MU").upper()
        run_btn = st.button("Run Analysis", type="primary")

    if run_btn:
        with st.spinner(f"Applying domain logic to {ticker}..."):
            try:
                data = fetch_market_data(ticker)
                if data.empty:
                    st.error("❌ Data Fetch Failed. API blocked or invalid ticker.")
                    st.stop()

                m = calculate_metrics(data)

                # --- 1. KEY METRICS ---
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Live Price", f"${m['price']:.2f}")
                k2.metric("20-Day SMA", f"${m['mu']:.2f}", delta=f"{m['price_adjustment']:.2f}", delta_color="inverse")
                k3.metric("Z-Score (Fat Tail)", f"{m['z_score']:.2f}σ", delta_color="off")
                k4.metric("Volume Strength", f"{m['vol_ratio']:.1f}x", 
                          help="Current Volume vs 20-Day Avg. >1.5x indicates high conviction.",
                          delta="High Conviction" if m['vol_ratio'] > 1.2 else "Low Conviction")

                st.divider()

                # --- 2. DOMAIN KNOWLEDGE VISUALIZATION ---
                # Generate t-Distribution Curve (Fatter tails than Normal)
                x = np.linspace(-4, 4, 1000)
                y = t.pdf(x, df=5) # df=5 for fat tails
                
                fig = go.Figure()

                # Curve
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='t-Distribution (df=5)', line=dict(color='black', width=2)))
                
                # Z-Score Line
                z = m['z_score']
                color = "#FF4B4B" if abs(z) >= 2 else "#222222"
                fig.add_vline(x=z, line_width=2, line_dash="dash", line_color=color)

                # Tail Shading
                if z > 0:
                    fill_x = x[x >= z]
                    fill_y = y[x >= z]
                else:
                    fill_x = x[x <= z]
                    fill_y = y[x <= z]
                
                fig.add_trace(go.Scatter(x=fill_x, y=fill_y, fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.4)', line=dict(width=0), name="Tail Risk"))

                # Annotations
                fig.add_annotation(x=z, y=max(y)*0.4, text=f"Price: {z:.2f}σ", showarrow=True, arrowhead=2, font=dict(color=color))

                fig.update_layout(
                    title=f"<b>{ticker} Statistical Profile (Fat-Tail Adjusted)</b>",
                    xaxis_title="Standard Deviations (σ)",
                    yaxis_title="Probability Density",
                    template="plotly_white",
                    height=500,
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- 3. THE VERDICT (DOMAIN KNOWLEDGE INTEGRATION) ---
                st.subheader("👨‍💻 Quant Verdict")
                
                # Logic Engine
                z_abs = abs(z)
                vol_high = m['vol_ratio'] > 1.2
                vol_low = m['vol_ratio'] < 0.8
                
                if z_abs < 1.0:
                    status = "✅ **NORMAL NOISE:** Market is undefined. No statistical edge."
                    action = "WAIT"
                elif z_abs >= 2.0:
                    # The "Crisis" or "Climax" Zone
                    if vol_high:
                        status = "⚠️ **BREAKOUT / CRASH (High Vol):** Price is extending with volume support. Do NOT fade blindly. The 'Fat Tail' event is active."
                        action = "MOMENTUM FOLLOW"
                    elif vol_low:
                        status = "🚨 **EXHAUSTION (Low Vol):** Price is extending but volume is dying. Reversion probability is MAXIMUM."
                        action = "MEAN REVERSION"
                    else:
                        status = "⚠️ **STATISTICAL EXTREME:** Price is highly deviated."
                        action = "WATCH"
                else:
                    status = "ℹ️ **TRENDING:** Price is moving within standard volatility."
                    action = "HOLD"

                st.info(f"""
                **1. Statistical Reality (t-Stat):** There is a **{m['p_value']*100:.2f}% probability** of this move. (Adjusted for market fat-tails).
                
                **2. Volume Confirmation:** Volume is **{m['vol_ratio']:.1f}x** the average. {'This confirms the move.' if vol_high else 'This suggests lack of participation.'}
                
                **3. Strategy Signal:** {status}
                """)

            except Exception as e:
                st.error(f"System Error: {str(e)}")

if __name__ == "__main__":
    main()
