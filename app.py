import streamlit as st

# --- 1. CRITICAL STARTUP (Must be Line 1) ---
st.set_page_config(page_title="Quant Scanner v3.4", layout="wide", page_icon="🛡️")

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
from tenacity import retry, stop_after_attempt, wait_fixed
import socket
import re

# --- SAFETY SETTINGS ---
socket.setdefaulttimeout(15)

# --- CSS STYLING ---
st.markdown("""
<style>
    .matrix-table { width: 100%; border-collapse: collapse; font-size: 14px; margin-bottom: 20px; }
    .matrix-table th { background-color: #262730; color: white; padding: 10px; text-align: left; border-bottom: 2px solid #444; }
    .matrix-table td { padding: 10px; border-bottom: 1px solid #ddd; color: #333; }
    .highlight-blue { background-color: #e3f2fd; border-left: 5px solid #2196f3; font-weight: bold; }
    .highlight-green { background-color: #e8f5e9; border-left: 5px solid #4caf50; font-weight: bold; }
    .highlight-orange { background-color: #fff3e0; border-left: 5px solid #ff9800; font-weight: bold; }
    .highlight-red { background-color: #ffebee; border-left: 5px solid #f44336; font-weight: bold; }
    .highlight-grey { background-color: #f0f2f6; border-left: 5px solid #999; color: #666; }
    .faded { color: #999 !important; opacity: 0.5; }
</style>
""", unsafe_allow_html=True)

# --- HELPER: INPUT VALIDATION (Security Fix) ---
def validate_ticker(ticker_input):
    """Sanitizes input to prevent garbage injection."""
    if not ticker_input: return None
    clean = ticker_input.upper().strip()
    # Allow letters, numbers, dots (BRK.B), and dashes
    if not re.match(r"^[\w\-\.]+$", clean):
        return None
    return clean

# --- MATH ENGINE ---
def calculate_adx_safe(df, period=14):
    try:
        high, low, close = df['High'], df['Low'], df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up = high - high.shift(1)
        down = low.shift(1) - low
        pos_dm = np.where((up > down) & (up > 0), up, 0.0)
        neg_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        alpha = 1 / period
        # Use min_periods=1 to catch early data, fill NaNs to prevent crashes
        tr_smooth = tr.ewm(alpha=alpha, min_periods=1, adjust=False).mean().replace(0, np.nan).ffill()
        pos_smooth = pd.Series(pos_dm).ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        neg_smooth = pd.Series(neg_dm).ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        
        pos_di = 100 * (pos_smooth / tr_smooth)
        neg_di = 100 * (neg_smooth / tr_smooth)
        
        denom = (pos_di + neg_di).replace(0, np.nan).ffill()
        dx = 100 * abs(pos_di - neg_di) / denom
        
        adx = dx.ewm(alpha=alpha, min_periods=1, adjust=False).mean().iloc[-1]
        
        return 0.0 if np.isnan(adx) else adx
    except:
        return 0.0

def calculate_metrics(df):
    try:
        if len(df) < 50: return None
        
        closes = df['Close']
        window = 20
        curr = closes.iloc[-1]
        mu = closes.rolling(window).mean().iloc[-1]
        sigma = closes.rolling(window).std().iloc[-1]
        z = (curr - mu) / sigma if sigma > 0 else 0
        
        p = (1 - t.cdf(abs(z), df=5)) * 2
        
        # FIX: Handle Zero Volume (Halted stocks)
        curr_vol = df['Volume'].iloc[-1]
        med_vol = df['Volume'].rolling(window).median().iloc[-1]
        if med_vol == 0: med_vol = 1 # Prevent div/0
        vol = curr_vol / med_vol
        
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        adx = calculate_adx_safe(df)
        
        return {"price": curr, "z": z, "p": p, "vol": vol, "rsi": rsi, "adx": adx, "mu": mu}
    except:
        return None

# --- DATA ENGINE ---
@st.cache_data(ttl=300)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, threads=False)
        if df.empty: return pd.DataFrame()
        
        # Nuclear Flattening
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.capitalize() for c in df.columns]
        
        # Normalize Close
        if 'Close' not in df.columns and 'Adj close' in df.columns:
            df['Close'] = df['Adj close']
            
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        
        # FIX: Drop NaN rows that often appear at start of yfinance data
        return df.dropna().ffill().bfill()
    except: return pd.DataFrame()

# --- MAIN UI ---
def main():
    st.title("🛡️ Quant Scanner v3.4")
    
    with st.sidebar:
        raw_ticker = st.text_input("Ticker Symbol", "MU")
        if st.button("Run Analysis", type="primary"):
            # Security Check
            valid_ticker = validate_ticker(raw_ticker)
            if valid_ticker:
                st.session_state.run = True
                st.session_state.ticker = valid_ticker
            else:
                st.error("Invalid Ticker. Alphanumeric only.")

    if st.session_state.get('run'):
        target = st.session_state.get('ticker')
        with st.spinner(f"Scanning {target}..."):
            df = fetch_data(target)
            if df.empty: st.error("Data Fetch Error"); return
            m = calculate_metrics(df)
            if not m: st.error("Insufficient Data History"); return
            
            # --- LOGIC PRIORITY (The "Fix") ---
            z_abs = abs(m['z'])
            vol = m['vol']
            adx = m['adx']
            
            state = "sleep"
            if z_abs >= 3.0: state = "anomaly"
            elif z_abs >= 2.0 and vol > 1.2: state = "breakout"
            elif z_abs >= 2.0 and vol < 0.8: state = "exhaustion"
            elif z_abs >= 2.0: state = "anomaly"
            elif adx < 20: state = "sleep"
            elif 1.0 <= z_abs < 2.0 and adx > 25: state = "trend"
            
            # --- DISPLAY METRICS ---
            if z_abs > 2.0:
                st.error(f"🚨 FAT TAIL EVENT: {m['z']:.2f}σ")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Price", f"${m['price']:.2f}")
            c2.metric("Z-Score", f"{m['z']:.2f}σ", delta="Extreme" if z_abs>2 else "Normal", delta_color="inverse")
            c3.metric("Volume", f"{m['vol']:.1f}x")
            
            adx_label = "N/A" if m['adx'] == 0 else f"{m['adx']:.0f}"
            c4.metric("ADX", adx_label, delta="Trending" if m['adx']>25 else "Choppy", delta_color="normal" if m['adx']>25 else "off")
            c5.metric("RSI", f"{m['rsi']:.0f}")
            
            st.divider()
            
            # --- MATRIX ---
            rows = [
                {"id": "breakout", "cond": "High Momentum", "z": "> 2.0 σ", "vol": "> 1.2x", "adx": "Any", "verdict": "🚀 BREAKOUT"},
                {"id": "exhaustion", "cond": "Exhaustion", "z": "> 2.0 σ", "vol": "< 0.8x", "adx": "Any", "verdict": "🛑 REVERSAL"},
                {"id": "anomaly", "cond": "Stat Outlier", "z": "> 2.0 σ", "vol": "Normal", "adx": "Any", "verdict": "⚠️ ANOMALY"},
                {"id": "trend", "cond": "Trending", "z": "1.0 - 2.0 σ", "vol": "Normal", "adx": "> 25", "verdict": "🌊 RIDE TREND"},
                {"id": "sleep", "cond": "Normal / Chop", "z": "Any", "vol": "Any", "adx": "< 20", "verdict": "😴 SLEEP"},
            ]
            
            html = ['<table class="matrix-table"><tr><th>Condition</th><th>Z-Score</th><th>Volume</th><th>ADX</th><th>Verdict</th></tr>']
            for row in rows:
                theme = "faded"
                if row['id'] == state:
                    if state == "breakout": theme = "highlight-orange"
                    elif state == "exhaustion": theme = "highlight-red"
                    elif state == "trend": theme = "highlight-green"
                    elif state == "anomaly": theme = "highlight-orange"
                    elif state == "sleep": theme = "highlight-grey"
                html.append(f'<tr class="{theme}"><td>{row["cond"]}</td><td>{row["z"]}</td><td>{row["vol"]}</td><td>{row["adx"]}</td><td>{row["verdict"]}</td></tr>')
            html.append('</table>')
            st.markdown("".join(html), unsafe_allow_html=True)
            
            st.divider()
            
            # --- CONCLUSION (Restored) ---
            gap = m['price'] - m['mu']
            direction = "above" if m['z'] > 0 else "below"
            
            st.markdown("### 📝 Statistical Observations")
            obs = f"""
            * **Rarity:** There is only a **{m['p']*100:.2f}% probability** of price being this far {direction} the average.
            * **Mean Reversion:** The 20-Day SMA is **${m['mu']:.2f}**. Price is **${abs(gap):.2f}** {direction} this mean.
            * **Momentum:** RSI is **{m['rsi']:.1f}**. (Values >70 are Overbought, <30 are Oversold).
            """
            if z_abs > 2.0: st.warning(obs)
            else: st.info(obs)
            
            st.divider()
            
            # --- CHART ---
            x = np.linspace(-4, 4, 1000)
            y = t.pdf(x, df=5)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='#333')))
            color = "#FF4B4B" if z_abs >= 2 else "#2ECC71"
            fig.add_vline(x=m['z'], line=dict(color=color, width=2, dash='dash'))
            fig.add_annotation(x=m['z'], y=0.3, text=f"YOU<br>{m['z']:.2f}σ", font=dict(color=color))
            fig.update_layout(template="plotly_white", height=300, margin=dict(t=20, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
