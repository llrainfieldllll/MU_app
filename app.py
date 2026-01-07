import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
from tenacity import retry, stop_after_attempt, wait_fixed
from curl_cffi import requests as crequests # The "Nuclear" Browser Spoofer
import re
from datetime import datetime
import yfinance as yf # Imported as requested (keeps the dependency valid)

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Quant Scanner v4.3", layout="wide", page_icon="🛡️")

# --- 2. HIGH-CONTRAST CSS ---
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
    
    /* Shockwave Alert */
    .shock-alert { padding: 15px; border-radius: 8px; font-weight: bold; margin-bottom: 20px; text-align: center; border: 2px solid; }
</style>
""", unsafe_allow_html=True)

# --- 3. THE "NUCLEAR" DATA ENGINE (Yahoo Finance Direct) ---
@st.cache_data(ttl=300, show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_data_nuclear(ticker):
    """
    Fetches full OHLCV data directly from Yahoo Finance API.
    Uses browser spoofing (Chrome 110) to bypass the 'Insufficient Data' block.
    """
    try:
        # 1. Yahoo Raw JSON Endpoint
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"
        
        # 2. Impersonate Chrome 110 (Crucial for bypass)
        r = crequests.get(url, impersonate="chrome110", timeout=10)
        
        # 3. Handle Block/Error
        if r.status_code != 200: return pd.DataFrame()

        # 4. Parse Complex JSON
        data = r.json()
        result = data['chart']['result'][0]
        indicators = result['indicators']['quote'][0]
        
        # Extract columns safely
        timestamps = result['timestamp']
        opens = indicators.get('open', [])
        highs = indicators.get('high', [])
        lows = indicators.get('low', [])
        closes = indicators.get('close', [])
        volumes = indicators.get('volume', [])
        
        # 5. Build DataFrame
        df = pd.DataFrame({
            'Open': opens, 'High': highs, 'Low': lows, 
            'Close': closes, 'Volume': volumes, 'Timestamp': timestamps
        })
        
        # 6. Clean and Index
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
        # FIX: Normalize timezone to prevent issues
        df['Date'] = df['Date'].dt.tz_localize(None) 
        df.set_index('Date', inplace=True)
        df = df.dropna()
        
        return df
        
    except Exception as e:
        return pd.DataFrame()

# --- 4. MATH ENGINE ---
def calculate_adx_safe(df, period=14):
    try:
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
        
        alpha = 1 / period
        tr_s = tr.ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        pos_s = pos_dm.ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        neg_s = neg_dm.ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        
        pos_di = 100 * (pos_s / tr_s)
        neg_di = 100 * (neg_s / tr_s)
        
        denom = pos_di + neg_di
        # FIX: Avoid Division by Zero
        denom = denom.replace(0, np.nan).ffill()
        
        dx = 100 * abs(pos_di - neg_di) / denom
        adx = dx.ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        
        return adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0.0
    except: return 0.0

def check_shockwave(closes):
    """Checks for +/- 10% move in last 5 trading days."""
    try:
        if len(closes) < 6: return None
        curr = closes.iloc[-1]
        past = closes.iloc[-6] # 5 trading days ago (approx 1 week)
        pct = (curr - past) / past
        
        if pct >= 0.10: return ("🚀 ROCKET", pct, "#d4edda", "#155724") # Green
        if pct <= -0.10: return ("🩸 CRASHING", pct, "#f8d7da", "#721c24") # Red
        return None
    except: return None

def calculate_metrics(df):
    try:
        if len(df) < 200: return None
        closes = df['Close']
        window = 20
        curr = closes.iloc[-1]
        last_date = df.index[-1]
        
        # Z-Score
        mu = closes.rolling(window).mean().iloc[-1]
        sigma = closes.rolling(window).std().iloc[-1]
        z = (curr - mu) / sigma if sigma > 0 else 0
        p = (1 - t.cdf(abs(z), df=5)) * 2
        
        # Volume Ratio
        med_vol = df['Volume'].rolling(window).median().iloc[-1]
        vol_ratio = (df['Volume'].iloc[-1] / med_vol) if med_vol > 0 else 1.0
        
        # RSI
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # ADX
        adx = calculate_adx_safe(df)
        
        # Regime (SMA)
        sma50 = closes.rolling(50).mean().iloc[-1]
        sma200 = closes.rolling(200).mean().iloc[-1]
        
        regime = "NEUTRAL"
        if curr > sma200: regime = "BULL"
        elif curr < sma200 and curr > sma50: regime = "RECOVERY"
        else: regime = "BEAR"
        
        return {
            "price": curr, "z": z, "p": p, "vol": vol_ratio, 
            "rsi": rsi, "adx": adx, "mu": mu, "date": last_date,
            "regime": regime, "sma200": sma200
        }
    except: return None

# --- 5. MAIN UI ---
def main():
    st.title("🛡️ Quant Scanner v4.3")
    
    with st.sidebar:
        raw_ticker = st.text_input("Ticker Symbol", "ASTS")
        if st.button("Run Analysis", type="primary"):
            if raw_ticker and re.match(r"^[\w\-\.]+$", raw_ticker.strip()):
                st.session_state.run = True
                st.session_state.ticker = raw_ticker.upper().strip()
            else:
                st.error("Invalid Ticker")

    if st.session_state.get('run'):
        target = st.session_state.get('ticker')
        
        with st.spinner(f"Establishing Secure Connection to {target}..."):
            # Use Nuclear Fetcher (Yahoo Direct)
            df = fetch_data_nuclear(target)
            
            # Validation
            if df.empty:
                st.error("🛑 **Connection Blocked:** Yahoo refused the data connection. Try again in 5 mins.")
                return
            
            m = calculate_metrics(df)
            if not m: 
                st.error(f"⚠️ **Insufficient History:** {target} has less than 200 days of data.")
                return
            
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
            
            # --- ALERT: SHOCKWAVE (The New Flag) ---
            shock = check_shockwave(df['Close'])
            if shock:
                label, pct, bg_col, txt_col = shock
                st.markdown(f"""
                <div class="shock-alert" style="background-color: {bg_col}; color: {txt_col}; border-color: {txt_col};">
                    {label}: {pct:+.1%} Move in 7 Days
                </div>
                """, unsafe_allow_html=True)

            # --- CONTEXT BADGE ---
            regime = m['regime']
            if regime == "BULL":
                st.success(f"🟢 **MARKET CONTEXT: BULL REGIME** (Price > 200 SMA). Safe for Breakouts.")
            elif regime == "RECOVERY":
                st.warning(f"🟡 **MARKET CONTEXT: RECOVERY** (Price > 50 SMA). Tread carefully.")
            else:
                st.error(f"🔴 **MARKET CONTEXT: BEAR REGIME** (Price < 200 SMA). Favor Reversals/Shorts.")

            # --- METRICS ---
            c1, c2, c3, c4, c5 = st.columns(5)
            
            c1.metric("Price", f"${m['price']:.2f}")
            c1.caption(f"📅 {m['date'].strftime('%Y-%m-%d')}")
            
            c2.metric("Z-Score", f"{m['z']:.2f}σ", 
                delta="Extreme" if z_abs>2 else "Normal", delta_color="inverse",
                help="DISTANCE FROM AVERAGE.\n• > 2.0: Anomaly/Breakout")
            
            c3.metric("Volume", f"{m['vol']:.1f}x", 
                help="ACTIVITY.\n• > 1.2x: Conviction\n• < 0.8x: Apathy")
            
            c4.metric("ADX", f"{m['adx']:.0f}", 
                delta="Trending" if m['adx']>25 else "Choppy",
                help="TREND STRENGTH.\n• > 25: Strong")
            
            c5.metric("RSI", f"{m['rsi']:.0f}",
                help="MOMENTUM.\n• > 70: Overbought\n• < 30: Oversold")
            
            st.divider()
            
            # --- MATRIX ---
            rows = [
                {"id": "breakout", "cond": "High Momentum", "z": "> 2.0 σ", "vol": "> 1.2x", "adx": "--", "verdict": "🚀 BREAKOUT"},
                {"id": "exhaustion", "cond": "Exhaustion", "z": "> 2.0 σ", "vol": "< 0.8x", "adx": "--", "verdict": "🛑 REVERSAL"},
                {"id": "anomaly", "cond": "Stat Outlier", "z": "> 2.0 σ", "vol": "0.8x - 1.2x", "adx": "--", "verdict": "⚠️ ANOMALY"},
                {"id": "trend", "cond": "Trending", "z": "1.0 - 2.0 σ", "vol": "0.8x - 1.2x", "adx": "> 25", "verdict": "🌊 RIDE TREND"},
                {"id": "sleep", "cond": "Normal / Chop", "z": "Any", "vol": "--", "adx": "< 20", "verdict": "😴 SLEEP"},
            ]
            
            html = ['<table class="matrix-table"><tr><th>Condition ⓘ</th><th>Z-Score ⓘ</th><th>Volume ⓘ</th><th>ADX ⓘ</th><th>Verdict ⓘ</th></tr>']
            for row in rows:
                css = f"signal-{row['id']}" if row['id'] == state else "plain-row"
                verdict = f"✅ {row['verdict']}" if row['id'] == state else row['verdict']
                html.append(f'<tr class="{css}"><td>{row["cond"]}</td><td>{row["z"]}</td><td>{row["vol"]}</td><td>{row["adx"]}</td><td>{verdict}</td></tr>')
            html.append('</table>')
            st.markdown("".join(html), unsafe_allow_html=True)
            
            st.divider()
            
            # --- CONCLUSION ---
            direction = "above" if m['z'] > 0 else "below"
            st.markdown("### 📝 Statistical Observations")
            st.info(f"""
            * **Macro Context:** The stock is in a **{regime}** regime.
            * **Rarity:** There is only a **{m['p']*100:.2f}% probability** of price being this far {direction} the average.
            * **Mean Reversion:** The 20-Day SMA is **${m['mu']:.2f}**.
            """)
            
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
