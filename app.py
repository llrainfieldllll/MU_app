import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
from tenacity import retry, stop_after_attempt, wait_fixed

# --- CONSTANTS ---
MIN_DATA_POINTS = 30  # Crash prevention for new IPOs
PAGE_TITLE = "Quant Scanner: Educational"

# --- CONFIGURATION ---
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# --- HELPER: OBSERVATION ENGINE ---
def get_technical_observation(z_score, vol_ratio, rsi):
    z_abs = abs(z_score)
    
    # 1. NORMAL NOISE (0.0 - 1.0)
    if z_abs < 1.0:
        return {
            "status": "✅ MARKET NOISE (NEUTRAL)",
            "color": "blue",
            "observation": "Price is within 1 standard deviation of the mean. No statistical anomaly detected."
        }
    
    # 2. TRENDING (1.0 - 2.0)
    elif z_abs >= 1.0 and z_abs < 2.0:
        return {
            "status": "ℹ️ TRENDING",
            "color": "green",
            "observation": "Price is trending. Statistical probability of continuation is currently higher than reversion."
        }
    
    # 3. EXTREME (> 2.0)
    else: 
        # A. HIGH VOLUME (Breakout)
        if vol_ratio > 1.2:
            return {
                "status": "⚠️ HIGH VOLUME ANOMALY",
                "color": "orange",
                "observation": "Price is >2σ with high volume (>1.2x). Historically, this pattern often precedes a continued trend (Breakout) rather than an immediate reversion."
            }
        
        # B. LOW VOLUME (Exhaustion)
        elif vol_ratio < 0.8:
            return {
                "status": "🚨 STATISTICAL EXHAUSTION",
                "color": "red",
                "observation": "Price is >2σ on low volume. This divergence suggests weakening participation. Mean reversion probability is statistically elevated."
            }
            
        # C. RSI EXTREME
        elif rsi > 80 or rsi < 20:
            return {
                "status": "⚠️ OSCILLATOR EXTREME",
                "color": "red",
                "observation": f"RSI is at {rsi:.0f}. Combined with Z-Score >2.0, this indicates a statistically overextended condition."
            }
            
        else:
            return {
                "status": "⚠️ STATISTICAL OUTLIER",
                "color": "yellow",
                "observation": "Price is currently a statistical outlier. Volatility is elevated."
            }

# --- DATA ENGINE ---
@st.cache_data(ttl=900, show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_market_data(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        # Fetch 6mo to ensure plenty of data for moving averages
        data = ticker_obj.history(period="6mo", interval="1d")
        
        # 1. Empty Check
        if data.empty: 
            return pd.DataFrame()
        
        # 2. Timezone Cleanup (Critical for index math)
        if data.index.tz is not None: 
            data.index = data.index.tz_localize(None)
        
        # 3. NaNs Cleanup
        data = data.dropna()

        # 4. Insufficient History Check (Prevents IPO Crash)
        if len(data) < MIN_DATA_POINTS:
            return pd.DataFrame()
        
        return data
    except Exception:
        return pd.DataFrame()

def calculate_metrics(df):
    try:
        prices = df['Close']
        volumes = df['Volume']
        
        current_price = prices.iloc[-1]
        current_volume = volumes.iloc[-1]
        
        # --- Z-SCORE ENGINE ---
        analysis_slice = prices.tail(20)
        mu = analysis_slice.mean()
        sigma = analysis_slice.std()
        
        # Safety: Prevent division by zero if stock hasn't moved
        if sigma == 0:
            z_score = 0
            p_value = 0.5
        else:
            z_score = (current_price - mu) / sigma
            # P-Value (Student's t, df=5 for fat tails)
            if z_score > 0:
                p_value = 1 - t.cdf(z_score, df=5)
            else:
                p_value = t.cdf(z_score, df=5)
        
        # --- VOLUME ENGINE ---
        # Use Median to avoid skew from one massive volume day
        vol_avg = volumes.tail(20).median()
        # Safety: Prevent division by zero
        vol_ratio = (current_volume / vol_avg) if vol_avg > 0 else 1.0

        # --- RSI ENGINE ---
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        
        # Handle cases where history exists but rolling window is NaN (start of array)
        if len(loss) > 0 and pd.notna(loss.iloc[-1]):
            last_loss = loss.iloc[-1]
            last_gain = gain.iloc[-1]
            
            if last_loss == 0:
                rsi = 100
            else:
                rs = last_gain / last_loss
                rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50 # Default neutral if calculation fails

        return {
            "price": current_price,
            "mu": mu,
            "z": z_score,
            "p": p_value,
            "vol": vol_ratio,
            "rsi": rsi,
            "valid": True
        }
    except Exception as e:
        # Fallback for unexpected math errors
        return {"valid": False, "error": str(e)}

# --- UI RENDERER ---
def main():
    st.title("🛡️ Quant Scanner: Statistical Analyzer")
    
    # LEGAL DISCLAIMER
    st.error("""
    **LEGAL DISCLAIMER:** This tool is for **Educational Purposes Only**. 
    The "Observations" below are generated by a statistical algorithm and do NOT constitute financial advice. 
    Data is sourced from third parties and may be inaccurate. 
    """)
    
    with st.sidebar:
        st.header("Configuration")
        ticker = st.text_input("Ticker Symbol", "MU").upper()
        run_btn = st.button("Run Analysis", type="primary")
        st.caption("Data provided by Yahoo Finance (Unofficial).")

    if run_btn:
        with st.spinner(f"Calculating statistical profile for {ticker}..."):
            data = fetch_market_data(ticker)
            
            # Error Handling: Data Fetch
            if data.empty:
                st.warning(f"⚠️ Unable to fetch sufficient data for '{ticker}'. The ticker may be new, invalid, or delisted.")
                st.stop()

            m = calculate_metrics(data)

            # Error Handling: Math Calculation
            if not m.get("valid", False):
                st.error("⚠️ Error calculating metrics. The data provided by the exchange may be malformed.")
                st.stop()

            # Logic
            obs = get_technical_observation(m['z'], m['vol'], m['rsi'])

            # --- DASHBOARD ---
            st.subheader(obs['status'])
            st.info(f"**Technical Observation:** {obs['observation']}")

            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${m['price']:.2f}")
            c2.metric("20-Day SMA", f"${m['mu']:.2f}")
            c3.metric("Z-Score", f"{m['z']:.2f}σ", help="Measures deviation from the mean.")
            c4.metric("Volume", f"{m['vol']:.1f}x", delta="High" if m['vol']>1.2 else "Normal")

            st.divider()

            # --- VISUALIZATION ---
            x = np.linspace(-4, 4, 1000)
            y = t.pdf(x, df=5)
            
            fig = go.Figure()
            # 1. Curve
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#333', width=2), name='Probability Dist'))
            
            # 2. Line
            z = m['z']
            line_col = "#FF4B4B" if abs(z) >= 2 else "#2ECC71"
            fig.add_vline(x=z, line_width=2, line_dash="dash", line_color=line_col)
            
            # 3. Shading
            if z > 0:
                fill_x = x[x >= z]
                fill_y = y[x >= z]
            else:
                fill_x = x[x <= z]
                fill_y = y[x <= z]
            
            fig.add_trace(go.Scatter(x=fill_x, y=fill_y, fill='tozeroy', fillcolor='rgba(255,0,0,0.2)', line=dict(width=0), name='Tail Region'))

            # 4. Marker
            fig.add_annotation(
                x=z, y=0.35, 
                text=f"CURRENT<br>{z:.2f}σ", 
                showarrow=True, arrowhead=2, 
                font=dict(color=line_col, size=12)
            )

            fig.update_layout(
                template="plotly_white", 
                title=f"Statistical Distribution ({ticker})", 
                height=450, 
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
