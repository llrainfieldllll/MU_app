import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
from tenacity import retry, stop_after_attempt, wait_fixed
import socket
import re

# --- 0. CONFIGURATION & SECURITY ---
st.set_page_config(page_title="Quant Scanner v2.1 | Hardened", layout="wide", page_icon="🛡️")
socket.setdefaulttimeout(10) # Increased timeout for slow connections

# --- 1. CSS (Dr. Vol's Dark Mode) ---
st.markdown("""
<style>
    .matrix-table { width: 100%; border-collapse: collapse; font-family: 'Roboto Mono', monospace; font-size: 13px; }
    .matrix-table th { background-color: #0E1117; color: #FAFAFA; border-bottom: 2px solid #333; padding: 8px; text-align: left; }
    .matrix-table td { padding: 8px; border-bottom: 1px solid #262730; color: #E0E0E0; }
    
    /* Status Indicators */
    .signal-sleep { background-color: #262730; color: #666 !important; opacity: 0.6; }
    .signal-breakout { background-color: #1E3A23; border-left: 4px solid #00FF00; color: #FFF; font-weight: bold; }
    .signal-exhaustion { background-color: #3A1E1E; border-left: 4px solid #FF0000; color: #FFF; font-weight: bold; }
    .signal-trend { background-color: #1C2E4A; border-left: 4px solid #2196F3; color: #FFF; }
    .signal-anomaly { background-color: #3D3D00; border-left: 4px solid #FFFF00; color: #FFF; }
    .faded { opacity: 0.3; }
</style>
""", unsafe_allow_html=True)

# --- 2. HELPER: INPUT VALIDATION ---
def validate_ticker(ticker_input):
    """Sanitizes and validates ticker input (Alphanumeric + . - only)."""
    clean_ticker = ticker_input.upper().strip()
    if not clean_ticker: return None
    # Regex: Allow letters, numbers, dots (BRK.B), and dashes
    if not re.match(r"^[\w\-\.]+$", clean_ticker):
        return None
    return clean_ticker

# --- 3. HELPER: ADX CALCULATION (Isolated) ---
def calculate_adx_series(df, period=14):
    """
    Calculates ADX using Wilder's Smoothing.
    Returns the final ADX value (scalar) or None if calculation fails.
    """
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range Calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Wilder's Smoothing (alpha = 1/period)
        # We use 'adjust=False' to mimic Wilder's RMA
        alpha = 1 / period
        ma_tr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
        ma_plus = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
        ma_minus = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
        
        # SAFETY: Handle Division by Zero
        ma_tr = ma_tr.replace(0, np.nan) # Avoid div/0
        
        plus_di = 100 * (ma_plus / ma_tr)
        minus_di = 100 * (ma_minus / ma_tr)
        
        # DX Calculation
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, 1.0) # SAFETY: Prevent crash on flat markets
        
        dx = 100 * abs(plus_di - minus_di) / di_sum
        
        # Final ADX Smoothing
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        
        return adx.iloc[-1]
    except Exception:
        return 0.0

# --- 4. DATA ENGINE (Hardened) ---
@st.cache_data(ttl=300, show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_data(ticker):
    try:
        # Fetch slightly more data to ensure ADX warms up
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, threads=False)
        
        if df.empty: return pd.DataFrame()
        
        # NUCLEAR FIX: Flatten MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Normalize Columns
        df.columns = [c.capitalize() for c in df.columns]
        
        # Data Integrity Check
        required = ['High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            if 'Adj close' in df.columns: 
                df['Close'] = df['Adj close']
            else:
                return pd.DataFrame() # Fail if critical columns missing

        if df.index.tz is not None: 
            df.index = df.index.tz_localize(None)
            
        return df.ffill().bfill()
    except Exception:
        return pd.DataFrame()

# --- 5. MATH ENGINE ---
def calculate_metrics(df):
    try:
        # Require 30 days minimum for stable statistical significance
        if len(df) < 30: return {"error": "Insufficient Data (Need 30+ candles)"}
        
        # A. Z-Score (20-Day Rolling)
        closes = df['Close']
        window = 20
        
        current_price = closes.iloc[-1]
        mu = closes.rolling(window).mean().iloc[-1]
        sigma = closes.rolling(window).std().iloc[-1]
        
        # Safety: Sigma 0 check
        z_score = (current_price - mu) / sigma if sigma > 0.0001 else 0.0
        
        # B. P-Value (Student's t, df=5)
        if z_score > 0: p_val = (1 - t.cdf(z_score, df=5)) * 2
        else: p_val = t.cdf(z_score, df=5) * 2
        
        # C. Volume Ratio (Robust Median)
        curr_vol = df['Volume'].iloc[-1]
        med_vol = df['Volume'].rolling(window).median().iloc[-1]
        vol_ratio = (curr_vol / med_vol) if med_vol > 0 else 1.0
        
        # D. ADX (Trend Strength)
        adx_val = calculate_adx_series(df)
        
        return {
            "price": current_price, "z": z_score, "p": p_val, 
            "vol": vol_ratio, "adx": adx_val, "mu": mu, "valid": True
        }
    except Exception as e:
        return {"error": f"Calc Error: {str(e)}"}

# --- 6. DECISION LOGIC ---
def get_verdict(m):
    z = m['z']
    vol = m['vol']
    adx = m['adx']
    abs_z = abs(z)
    
    # Priority 1: Extreme Anomalies (Panic overrides chop)
    if abs_z >= 3.0:
        return "anomaly", "⚠️ EXTREME ANOMALY"
    
    # Priority 2: The "Anti-Chop" Filter
    if adx < 20:
        return "sleep", "😴 SLEEP (Chop)"
        
    # Priority 3: Trading Signals
    if abs_z > 2.0:
        if vol > 1.2:
            direction = "UP" if z > 0 else "DOWN"
            return "breakout", f"🚀 BREAKOUT ({direction})"
        elif vol < 0.8:
            return "exhaustion", "🛑 EXHAUSTION (Reversal)"
        else:
            return "anomaly", "⚠️ VOLATILITY ALERT"
            
    if 1.0 <= abs_z <= 2.0 and adx > 25:
        return "trend", "🌊 RIDE TREND"
        
    return "sleep", "😴 WAIT / NOISE"

# --- 7. UI RENDERER ---
def main():
    st.title("🛡️ Quant Scanner v2.1")
    st.caption("Hardened Edition: Robust Math • ADX Filter • Safety Checks")
    
    with st.sidebar:
        st.header("Config")
        raw_ticker = st.text_input("Ticker Symbol", "MU")
        
        if st.button("Run Analysis", type="primary"):
            ticker = validate_ticker(raw_ticker)
            
            if not ticker:
                st.error("Invalid Ticker. Use Alphanumeric only.")
            else:
                run_analysis(ticker)

def run_analysis(ticker):
    with st.spinner(f"Scanning {ticker}..."):
        data = fetch_data(ticker)
        
        if data.empty:
            st.error(f"Failed to fetch data for '{ticker}'. Check symbol or try again later.")
            return
            
        m = calculate_metrics(data)
        
        if m.get("error"):
            st.warning(m["error"])
            return
            
        # Get Verdict
        state, label = get_verdict(m)
        
        # --- Top Banner ---
        if state == "breakout": st.success(f"SIGNAL: {label}")
        elif state == "exhaustion": st.error(f"SIGNAL: {label}")
        elif state == "anomaly": st.warning(f"ALERT: {label}")
        elif state == "sleep": st.info(f"STATUS: {label} - No edge detected.")
        else: st.info(f"STATUS: {label}")
        
        # --- Metrics ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Z-Score (20d)", f"{m['z']:.2f}σ", delta="Extreme" if abs(m['z'])>2 else "Normal", delta_color="inverse")
        c2.metric("ADX (Trend)", f"{m['adx']:.1f}", delta="Trending" if m['adx']>25 else "Choppy", delta_color="normal" if m['adx']>25 else "off")
        c3.metric("Volume Ratio", f"{m['vol']:.1f}x", help="vs 20-Day Median")
        c4.metric("Rarity (P-Val)", f"{m['p']*100:.1f}%")
        
        st.divider()
        
        # --- Matrix ---
        rows = [
            {"id": "sleep", "cond": "ADX < 20", "z": "Any", "vol": "Any", "verdict": "😴 SLEEP (Chop)"},
            {"id": "breakout", "cond": "ADX > 20", "z": "> 2.0 σ", "vol": "> 1.2x", "verdict": "🚀 BREAKOUT"},
            {"id": "exhaustion", "cond": "ADX > 20", "z": "> 2.0 σ", "vol": "< 0.8x", "verdict": "🛑 EXHAUSTION"},
            {"id": "trend", "cond": "ADX > 25", "z": "1.0 - 2.0 σ", "vol": "Normal", "verdict": "🌊 RIDE TREND"},
            {"id": "anomaly", "cond": "Any", "z": "> 3.0 σ", "vol": "Any", "verdict": "⚠️ ANOMALY"}
        ]
        
        html = ['<table class="matrix-table"><tr><th>Condition</th><th>Z-Score</th><th>Volume</th><th>Verdict</th></tr>']
        for row in rows:
            # Highlight logic
            is_active = (row['id'] == state)
            css_class = f"signal-{row['id']}" if is_active else "faded"
            html.append(f'<tr class="{css_class}"><td>{row["cond"]}</td><td>{row["z"]}</td><td>{row["vol"]}</td><td>{row["verdict"]}</td></tr>')
        html.append("</table>")
        st.markdown("".join(html), unsafe_allow_html=True)
        
        # --- Chart ---
        st.divider()
        x = np.linspace(-4, 4, 1000)
        y = t.pdf(x, df=5)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='#555'), name='Dist'))
        
        # Dynamic Color for Line
        line_color = '#00FF00' # Green default
        if abs(m['z']) >= 2: line_color = '#FF4B4B' # Red extreme
        
        fig.add_vline(x=m['z'], line=dict(color=line_color, width=2, dash='dash'))
        fig.add_annotation(x=m['z'], y=0.05, text=f"PRICE<br>{m['z']:.2f}σ", font=dict(color=line_color), showarrow=True, arrowhead=2)
        
        fig.update_layout(template="plotly_dark", height=300, margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
