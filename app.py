import streamlit as st

# --- 1. CONFIGURATION (MUST BE LINE 1) ---
st.set_page_config(page_title="Quant Scanner v3.5", layout="wide", page_icon="🛡️")

# --- 2. SETUP & IMPORTS ---
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from scipy.stats import t
    from tenacity import retry, stop_after_attempt, wait_fixed
    import re
except ImportError as e:
    st.error(f"CRITICAL: Missing dependency. {e}")
    st.stop()

# --- 3. CUSTOM CSS ---
st.markdown("""
<style>
    .matrix-table { width: 100%; border-collapse: collapse; font-family: 'Roboto Mono', monospace; font-size: 13px; }
    .matrix-table th { background-color: #0E1117; color: #FAFAFA; border-bottom: 2px solid #333; padding: 8px; text-align: left; }
    .matrix-table td { padding: 8px; border-bottom: 1px solid #262730; color: #E0E0E0; }
    
    .signal-sleep { background-color: #262730; color: #666 !important; opacity: 0.6; }
    .signal-breakout { background-color: #1E3A23; border-left: 4px solid #00FF00; color: #FFF; font-weight: bold; }
    .signal-exhaustion { background-color: #3A1E1E; border-left: 4px solid #FF0000; color: #FFF; font-weight: bold; }
    .signal-trend { background-color: #1C2E4A; border-left: 4px solid #2196F3; color: #FFF; }
    .signal-anomaly { background-color: #3D3D00; border-left: 4px solid #FFFF00; color: #FFF; }
    .faded { opacity: 0.3; }
</style>
""", unsafe_allow_html=True)

# --- 4. MATH ENGINE (Optimized) ---
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
        
        # Z-Score
        mu = closes.rolling(window).mean().iloc[-1]
        sigma = closes.rolling(window).std().iloc[-1]
        z = (curr - mu) / sigma if sigma > 0 else 0
        p = (1 - t.cdf(abs(z), df=5)) * 2
        
        # Volume
        med_vol = df['Volume'].rolling(window).median().iloc[-1]
        if med_vol == 0: med_vol = 1
        vol = df['Volume'].iloc[-1] / med_vol
        
        # RSI
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # ADX
        adx = calculate_adx_safe(df)
        
        return {"price": curr, "z": z, "p": p, "vol": vol, "rsi": rsi, "adx": adx, "mu": mu}
    except:
        return None

# --- 5. DATA ENGINE ---
@st.cache_data(ttl=300, show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_data(ticker):
    try:
        # Fetch 1y to ensure ADX stability
        df = yf.download(ticker, period="1y", interval="1d", progress=False, threads=False)
        if df.empty: return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.capitalize() for c in df.columns]
        
        if 'Close' not in df.columns and 'Adj close' in df.columns:
            df['Close'] = df['Adj close']
            
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df.dropna().ffill().bfill()
    except: return pd.DataFrame()

# --- 6. MAIN UI ---
def main():
    st.title("🛡️ Quant Scanner v3.5")
    
    with st.sidebar:
        raw_ticker = st.text_input("Ticker Symbol", "MU")
        if st.button("Run Analysis", type="primary"):
            # Input Sanitization
            if raw_ticker and re.match(r"^[\w\-\.]+$", raw_ticker.strip()):
                st.session_state.run = True
                st.session_state.ticker = raw_ticker.upper().strip()
            else:
                st.error("Invalid Ticker")

    if st.session_state.get('run'):
        target = st.session_state.get('ticker')
        with st.spinner(f"Scanning {target}..."):
            df = fetch_data(target)
            if df.empty: st.error("Data Fetch Error"); return
            m = calculate_metrics(df)
            if not m: st.error("Insufficient Data"); return
            
            # --- LOGIC PRIORITY ---
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
            
            # --- DISPLAY ---
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
                    if state == "breakout": theme = "signal-breakout"
                    elif state == "exhaustion": theme = "signal-exhaustion"
                    elif state == "trend": theme = "signal-trend"
                    elif state == "anomaly": theme = "signal-anomaly"
                    elif state == "sleep": theme = "signal-sleep"
                html.append(f'<tr class="{theme}"><td>{row["cond"]}</td><td>{row["z"]}</td><td>{row["vol"]}</td><td>{row["adx"]}</td><td>{row["verdict"]}</td></tr>')
            html.append('</table>')
            st.markdown("".join(html), unsafe_allow_html=True)
            
            st.divider()
            
            # --- CONCLUSION ---
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
            fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='#555')))
            color = "#FF4B4B" if z_abs >= 2 else "#2ECC71"
            fig.add_vline(x=m['z'], line=dict(color=color, width=2, dash='dash'))
            fig.add_annotation(x=m['z'], y=0.3, text=f"YOU<br>{m['z']:.2f}σ", font=dict(color=color))
            fig.update_layout(template="plotly_dark", height=300, margin=dict(t=20, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
