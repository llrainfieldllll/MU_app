import streamlit as st

# --- 1. CONFIGURATION (Line 1) ---
st.set_page_config(page_title="Quant Scanner v4.0", layout="wide", page_icon="🛡️")

# --- 2. IMPORTS ---
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from scipy.stats import t
    from tenacity import retry, stop_after_attempt, wait_fixed
    import re
    from datetime import datetime
except ImportError as e:
    st.error(f"CRITICAL ERROR: Missing Library. {e}")
    st.stop()

# --- 3. HIGH-CONTRAST CSS ---
st.markdown("""
<style>
    /* Table Styling */
    .matrix-table { width: 100%; border-collapse: collapse; font-family: 'Roboto Mono', monospace; font-size: 14px; margin-bottom: 20px; }
    .matrix-table th { background-color: #000000; color: #FFFFFF; border-bottom: 3px solid #444; padding: 12px; text-align: left; cursor: help; }
    .matrix-table td { padding: 12px; border-bottom: 1px solid #ddd; color: #000; font-weight: 500; }
    
    /* Signal Rows */
    .signal-breakout { background-color: #e8f5e9; border-left: 6px solid #2e7d32; color: #1b5e20; } 
    .signal-exhaustion { background-color: #ffebee; border-left: 6px solid #c62828; color: #b71c1c; } 
    .signal-anomaly { background-color: #fff8e1; border-left: 6px solid #fbc02d; color: #f57f17; } 
    .signal-trend { background-color: #e3f2fd; border-left: 6px solid #1565c0; color: #0d47a1; } 
    .signal-sleep { background-color: #f5f5f5; border-left: 6px solid #9e9e9e; color: #616161; }
    
    .plain-row { background-color: #ffffff; color: #999; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { color: #000 !important; font-weight: 700 !important; }
    div[data-testid="stMetricLabel"] { color: #444 !important; font-weight: 600 !important; }
    
    /* Date Caption */
    .date-caption { font-size: 12px; color: #666; font-style: italic; margin-top: -15px; }
</style>
""", unsafe_allow_html=True)

# --- 4. MATH ENGINE ---
def calculate_adx_safe(df, period=14):
    try:
        df = df.copy().dropna()
        if len(df) < period * 2: return 0.0

        high, low, close = df['High'], df['Low'], df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up = high - high.shift(1)
        down = low.shift(1) - low
        pos_dm = np.where((up > down) & (up > 0), up, 0.0)
        neg_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        pos_dm = pd.Series(pos_dm, index=df.index)
        neg_dm = pd.Series(neg_dm, index=df.index)
        
        tr = tr.iloc[1:]
        pos_dm = pos_dm.iloc[1:]
        neg_dm = neg_dm.iloc[1:]
        
        alpha = 1 / period
        
        tr_s = tr.ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        pos_s = pos_dm.ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        neg_s = neg_dm.ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        
        pos_di = 100 * (pos_s / tr_s)
        neg_di = 100 * (neg_s / tr_s)
        
        denom = pos_di + neg_di
        denom = denom.replace(0, np.nan).ffill()
        
        dx = 100 * abs(pos_di - neg_di) / denom
        adx = dx.ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        
        final_val = adx.iloc[-1]
        if np.isnan(final_val): return 0.0
        return final_val
        
    except Exception:
        return 0.0

def calculate_metrics(df):
    try:
        df = df.dropna()
        if len(df) < 50: return None
        
        closes = df['Close']
        window = 20
        curr = closes.iloc[-1]
        last_date = df.index[-1]
        
        mu = closes.rolling(window).mean().iloc[-1]
        sigma = closes.rolling(window).std().iloc[-1]
        z = (curr - mu) / sigma if sigma > 0 else 0
        p = (1 - t.cdf(abs(z), df=5)) * 2
        
        med_vol = df['Volume'].rolling(window).median().iloc[-1]
        vol_ratio = (df['Volume'].iloc[-1] / med_vol) if med_vol > 0 else 1.0
        
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        adx = calculate_adx_safe(df)
        
        return {
            "price": curr, "z": z, "p": p, "vol": vol_ratio, 
            "rsi": rsi, "adx": adx, "mu": mu, "date": last_date
        }
    except:
        return None

# --- 5. DATA ENGINE ---
@st.cache_data(ttl=300, show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, threads=False)
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
    st.title("🛡️ Quant Scanner v4.0")
    
    with st.sidebar:
        raw_ticker = st.text_input("Ticker Symbol", "MU")
        if st.button("Run Analysis", type="primary"):
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
            
            # --- LOGIC ---
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
            
            # --- METRICS & DATE LABEL ---
            if z_abs > 2.0:
                st.error(f"🚨 FAT TAIL EVENT: {m['z']:.2f}σ")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            
            # TOOLTIPS ADDED HERE (The 'help' parameter)
            c1.metric("Price", f"${m['price']:.2f}", help="Current Market Price. Data may be 15min delayed.")
            c1.caption(f"📅 Data: {m['date'].strftime('%Y-%m-%d')}")
            
            c2.metric("Z-Score", f"{m['z']:.2f}σ", 
                delta="Extreme" if z_abs>2 else "Normal", delta_color="inverse",
                help="DISTANC FROM AVERAGE.\n• 0.0 - 1.0: Noise (Ignore)\n• > 2.0: Anomaly/Breakout (Pay Attention)\n• > 3.0: Extreme Event")
            
            c3.metric("Volume", f"{m['vol']:.1f}x", 
                help="ACTIVITY LEVEL.\n• 1.0x: Average activity.\n• > 1.2x: High Buying/Selling pressure (Conviction).\n• < 0.8x: Low interest (Apathy).")
            
            adx_label = f"{m['adx']:.0f}"
            c4.metric("ADX", adx_label, 
                delta="Trending" if m['adx']>25 else "Choppy", delta_color="normal" if m['adx']>25 else "off",
                help="TREND STRENGTH (Lagging).\n• < 20: Sleep/Chop (No trend).\n• > 25: Strong Trend.\n• NOTE: We IGNORE this for sudden Breakouts.")
            
            c5.metric("RSI", f"{m['rsi']:.0f}",
                help="MOMENTUM.\n• > 70: Overbought (Expensive).\n• < 30: Oversold (Cheap).\n• 50: Neutral.")
            
            st.divider()
            
            # --- MATRIX WITH HOVER TOOLTIPS ---
            # Added 'title' attributes to headers for hover explanations
            rows = [
                {"id": "breakout", "cond": "High Momentum", "z": "> 2.0 σ", "vol": "> 1.2x", "adx": "--", "verdict": "🚀 BREAKOUT"},
                {"id": "exhaustion", "cond": "Exhaustion", "z": "> 2.0 σ", "vol": "< 0.8x", "adx": "--", "verdict": "🛑 REVERSAL"},
                {"id": "anomaly", "cond": "Stat Outlier", "z": "> 2.0 σ", "vol": "0.8x - 1.2x", "adx": "--", "verdict": "⚠️ ANOMALY"},
                {"id": "trend", "cond": "Trending", "z": "1.0 - 2.0 σ", "vol": "0.8x - 1.2x", "adx": "> 25", "verdict": "🌊 RIDE TREND"},
                {"id": "sleep", "cond": "Normal / Chop", "z": "Any", "vol": "--", "adx": "< 20", "verdict": "😴 SLEEP"},
            ]
            
            html = [
                '<table class="matrix-table">',
                '<tr>',
                '<th title="The specific market scenario we are looking for.">Condition ⓘ</th>',
                '<th title="Z-Score: Distance from 20-Day Average. >2.0 is an anomaly.">Z-Score ⓘ</th>',
                '<th title="Volume Ratio: Current Vol vs Average. >1.2x is conviction.">Volume ⓘ</th>',
                '<th title="ADX: Trend Strength. We ignore it for Breakouts.">ADX ⓘ</th>',
                '<th title="Dr. Vol\'s Recommendation based on logic.">Verdict ⓘ</th>',
                '</tr>'
            ]
            
            for row in rows:
                if row['id'] == state:
                    css_class = f"signal-{row['id']}"
                    verdict = f"✅ {row['verdict']}"
                else:
                    css_class = "plain-row"
                    verdict = row['verdict']
                html.append(f'<tr class="{css_class}"><td>{row["cond"]}</td><td>{row["z"]}</td><td>{row["vol"]}</td><td>{row["adx"]}</td><td>{verdict}</td></tr>')
            html.append('</table>')
            st.markdown("".join(html), unsafe_allow_html=True)
            
            st.divider()
            
            # --- CONCLUSION ---
            gap = m['price'] - m['mu']
            direction = "above" if m['z'] > 0 else "below"
            st.markdown("### 📝 Statistical Observations")
            obs_text = f"""
            * **Rarity:** There is only a **{m['p']*100:.2f}% probability** of price being this far {direction} the average.
            * **Mean Reversion:** The 20-Day SMA is **${m['mu']:.2f}**. Price is **${abs(gap):.2f}** {direction} this mean.
            * **Momentum:** RSI is **{m['rsi']:.1f}**. (Values >70 are Overbought, <30 are Oversold).
            """
            if z_abs > 2.0: st.warning(obs_text)
            else: st.info(obs_text)
            
            st.divider()
            
            # --- CHART ---
            x = np.linspace(-4, 4, 1000)
            y = t.pdf(x, df=5)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='#000000', width=2)))
            color = "#D50000" if z_abs >= 2 else "#00C853"
            fig.add_vline(x=m['z'], line=dict(color=color, width=3, dash='dash'))
            fig.add_annotation(x=m['z'], y=0.35, text=f"<b>YOU</b><br>{m['z']:.2f}σ", font=dict(color=color, size=14))
            fig.update_layout(template="plotly_white", height=350, margin=dict(t=20, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
