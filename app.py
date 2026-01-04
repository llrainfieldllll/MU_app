import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
from tenacity import retry, stop_after_attempt, wait_fixed
import socket

# --- 1. CONFIGURATION (Dr. Vol's Clean UI) ---
st.set_page_config(page_title="Quant Scanner: Statistical Analyzer", layout="wide", page_icon="🛡️")
socket.setdefaulttimeout(10)

# Custom CSS to match the ORIGINAL look you liked
st.markdown("""
<style>
    .matrix-table { width: 100%; border-collapse: collapse; font-size: 14px; margin-bottom: 20px; }
    .matrix-table th { background-color: #262730; color: white; padding: 10px; text-align: left; border-bottom: 2px solid #444; }
    .matrix-table td { padding: 10px; border-bottom: 1px solid #ddd; color: #333; }
    
    /* Highlight Classes (Restored v1.0 Colors) */
    .highlight-blue { background-color: #e3f2fd; border-left: 5px solid #2196f3; font-weight: bold; }
    .highlight-green { background-color: #e8f5e9; border-left: 5px solid #4caf50; font-weight: bold; }
    .highlight-orange { background-color: #fff3e0; border-left: 5px solid #ff9800; font-weight: bold; }
    .highlight-red { background-color: #ffebee; border-left: 5px solid #f44336; font-weight: bold; }
    .highlight-grey { background-color: #f0f2f6; border-left: 5px solid #999; color: #666; }
    
    /* Faded Rows */
    .faded { color: #999 !important; opacity: 0.5; }
</style>
""", unsafe_allow_html=True)

# --- 2. MATH ENGINE (Hardened & Simplified) ---
def calculate_adx_safe(df, period=14):
    """Calculates ADX with a 'nan' failsafe."""
    try:
        high, low, close = df['High'], df['Low'], df['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up = high - high.shift(1)
        down = low.shift(1) - low
        pos_dm = np.where((up > down) & (up > 0), up, 0.0)
        neg_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        # Smoothing (Wilder's)
        alpha = 1 / period
        tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
        pos_smooth = pd.Series(pos_dm).ewm(alpha=alpha, adjust=False).mean()
        neg_smooth = pd.Series(neg_dm).ewm(alpha=alpha, adjust=False).mean()
        
        # Avoid Div/0
        tr_smooth = tr_smooth.replace(0, 1)
        
        pos_di = 100 * (pos_smooth / tr_smooth)
        neg_di = 100 * (neg_smooth / tr_smooth)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di).replace(0, 1)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        
        val = adx.iloc[-1]
        # FAILSAFE: If NaN, return 0 (Sleep Mode)
        return 0.0 if np.isnan(val) else val
    except:
        return 0.0

def calculate_metrics(df):
    try:
        if len(df) < 50: return None # Need enough data for ADX
        
        # 1. Z-Score (20d)
        closes = df['Close']
        window = 20
        curr_price = closes.iloc[-1]
        mu = closes.rolling(window).mean().iloc[-1]
        sigma = closes.rolling(window).std().iloc[-1]
        z_score = (curr_price - mu) / sigma if sigma > 0 else 0
        
        # 2. P-Value (Fat Tail)
        p_val = (1 - t.cdf(abs(z_score), df=5)) * 2
        
        # 3. Volume Ratio (Median)
        curr_vol = df['Volume'].iloc[-1]
        med_vol = df['Volume'].rolling(window).median().iloc[-1]
        vol_ratio = (curr_vol / med_vol) if med_vol > 0 else 1.0
        
        # 4. RSI (Context)
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # 5. ADX (Trend Filter)
        adx = calculate_adx_safe(df)
        
        return {
            "price": curr_price, "z": z_score, "p": p_val, 
            "vol": vol_ratio, "rsi": rsi, "adx": adx, "mu": mu
        }
    except:
        return None

# --- 3. DATA ENGINE (The Nuclear Fix) ---
@st.cache_data(ttl=300)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_data(ticker):
    try:
        # FETCH 1 YEAR to ensure ADX math doesn't return NaN
        df = yf.download(ticker, period="1y", interval="1d", progress=False, threads=False)
        if df.empty: return pd.DataFrame()
        
        # Nuclear Flattening
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.capitalize() for c in df.columns]
        
        # Fallback Check
        if 'Close' not in df.columns and 'Adj close' in df.columns:
            df['Close'] = df['Adj close']
            
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df.ffill().bfill()
    except: return pd.DataFrame()

# --- 4. MATRIX RENDERER (Restored v1.0 Style) ---
def render_matrix(m):
    z_abs = abs(m['z'])
    adx = m['adx']
    vol = m['vol']
    
    # Logic Tree
    state = "sleep" # Default
    if z_abs >= 3.0: state = "anomaly"
    elif adx < 20: state = "sleep"
    elif z_abs >= 2.0 and vol > 1.2: state = "breakout"
    elif z_abs >= 2.0 and vol < 0.8: state = "exhaustion"
    elif z_abs >= 2.0: state = "anomaly"
    elif 1.0 <= z_abs < 2.0 and adx > 25: state = "trend"
    
    # Definitions
    rows = [
        {"id": "sleep", "cond": "Normal / Chop", "z": "Any", "vol": "Any", "adx": "< 20", "verdict": "😴 SLEEP (No Edge)"},
        {"id": "trend", "cond": "Trending", "z": "1.0 - 2.0 σ", "vol": "Normal", "adx": "> 25", "verdict": "🌊 RIDE TREND"},
        {"id": "breakout", "cond": "High Momentum", "z": "> 2.0 σ", "vol": "> 1.2x", "adx": "> 20", "verdict": "🚀 BREAKOUT"},
        {"id": "exhaustion", "cond": "Exhaustion", "z": "> 2.0 σ", "vol": "< 0.8x", "adx": "> 20", "verdict": "🛑 REVERSAL"},
        {"id": "anomaly", "cond": "Stat Outlier", "z": "> 2.0 σ", "vol": "Normal", "adx": "Any", "verdict": "⚠️ ANOMALY"}
    ]
    
    # Build HTML
    html = ['<table class="matrix-table"><tr><th>Market Condition</th><th>Z-Score</th><th>Volume</th><th>ADX (Trend)</th><th>Verdict</th></tr>']
    
    for row in rows:
        if row['id'] == state:
            # Assign color based on state
            if state == "breakout": theme = "highlight-orange"
            elif state == "exhaustion": theme = "highlight-red"
            elif state == "trend": theme = "highlight-green"
            elif state == "anomaly": theme = "highlight-orange"
            else: theme = "highlight-grey"
            
            html.append(f'<tr class="{theme}"><td>👉 {row["cond"]}</td><td>{row["z"]}</td><td>{row["vol"]}</td><td>{row["adx"]}</td><td>{row["verdict"]}</td></tr>')
        else:
            html.append(f'<tr class="faded"><td>{row["cond"]}</td><td>{row["z"]}</td><td>{row["vol"]}</td><td>{row["adx"]}</td><td>{row["verdict"]}</td></tr>')
            
    html.append('</table>')
    st.markdown("".join(html), unsafe_allow_html=True)
    return state

# --- 5. MAIN UI (Restored Layout) ---
def main():
    st.title("🛡️ Quant Scanner: Statistical Analyzer")
    
    # Sidebar for Input ONLY (Like v1.0)
    with st.sidebar:
        ticker = st.text_input("Ticker Symbol", "AMD").upper().strip()
        if st.button("Run Analysis", type="primary"):
            st.session_state.run = True

    # Main Execution
    if st.session_state.get('run'):
        with st.spinner(f"Scanning {ticker}..."):
            df = fetch_data(ticker)
            if df.empty: st.error("Data Fetch Failed"); return
            
            m = calculate_metrics(df)
            if not m: st.error("Insufficient Data for ADX Calc"); return
            
            # 1. ALERT BANNER
            if abs(m['z']) > 2.0:
                st.error(f"🚨 FAT TAIL EVENT: Price is {m['z']:.2f}σ from mean. Rarity: {m['p']*100:.1f}%")
            
            # 2. METRICS ROW (Clean, no weird columns)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Price", f"${m['price']:.2f}")
            c2.metric("Z-Score", f"{m['z']:.2f}σ", delta="Extreme" if abs(m['z'])>2 else "Normal", delta_color="inverse")
            c3.metric("Volume", f"{m['vol']:.1f}x")
            
            # Fix ADX Display: Show "Chop" if low, "Trend" if high
            adx_label = "Choppy" if m['adx'] < 20 else "Trending"
            c4.metric("ADX", f"{m['adx']:.0f}", delta=adx_label, delta_color="normal" if m['adx']>25 else "off")
            
            c5.metric("RSI", f"{m['rsi']:.0f}")

            st.divider()

            # 3. DECISION MATRIX
            st.subheader("📊 Decision Matrix")
            state = render_matrix(m)

            st.divider()

            # 4. CHART (Visual)
            x = np.linspace(-4, 4, 1000)
            y = t.pdf(x, df=5)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='#333'), name='Dist'))
            
            color = "#FF4B4B" if abs(m['z']) >= 2 else "#2ECC71"
            fig.add_vline(x=m['z'], line=dict(color=color, width=2, dash='dash'))
            fig.add_annotation(x=m['z'], y=0.3, text=f"CURRENT<br>{m['z']:.2f}σ", showarrow=True, arrowhead=2, font=dict(color=color))
            
            fig.update_layout(template="plotly_white", height=350, margin=dict(t=20, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
