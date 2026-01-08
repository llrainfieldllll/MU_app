import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import percentileofscore, t
from tenacity import retry, stop_after_attempt, wait_fixed
from curl_cffi import requests as crequests 
import re
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Quant Scanner v7.0", layout="wide", page_icon="🛡️")

# --- 2. HIGH-CONTRAST CSS ---
st.markdown("""
<style>
    /* Table Styling */
    .matrix-table { width: 100%; border-collapse: collapse; font-family: 'Roboto Mono', monospace; font-size: 14px; margin-bottom: 20px; }
    .matrix-table th { background-color: #000000; color: #FFFFFF; border-bottom: 3px solid #444; padding: 12px; text-align: left; }
    .matrix-table td { padding: 12px; border-bottom: 1px solid #ddd; color: #000; font-weight: 500; }
    
    /* Signal Rows (Audit Compliant Colors) */
    .signal-bull { background-color: #e8f5e9; border-left: 6px solid #2e7d32; color: #1b5e20; } 
    .signal-bear { background-color: #ffebee; border-left: 6px solid #c62828; color: #b71c1c; } 
    .signal-neut { background-color: #f5f5f5; border-left: 6px solid #9e9e9e; color: #616161; }
    
    .plain-row { background-color: #ffffff; color: #999; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { color: #000 !important; font-weight: 700 !important; }
    div[data-testid="stMetricLabel"] { color: #444 !important; font-weight: 600 !important; }
    
    /* Audit Compliant Alerts */
    .alert-upside { padding: 15px; border-radius: 8px; font-weight: bold; margin-bottom: 10px; text-align: center; border: 2px solid #155724; background-color: #d4edda; color: #155724; }
    .alert-downside { padding: 15px; border-radius: 8px; font-weight: bold; margin-bottom: 10px; text-align: center; border: 2px solid #721c24; background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# --- 3. THE "NUCLEAR" DATA ENGINE (JSON API v8) ---
@st.cache_data(ttl=300, show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_data_nuclear(ticker):
    """
    Fetches data using the v8 JSON Chart API to bypass Yahoo 401 blocks.
    """
    try:
        # Use v8 Chart API (More robust than v7 Download)
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"
        r = crequests.get(url, impersonate="chrome110", timeout=10)
        
        if r.status_code != 200: return pd.DataFrame()

        data = r.json()
        if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
            return pd.DataFrame()

        result = data['chart']['result'][0]
        indicators = result['indicators']['quote'][0]
        
        timestamps = result['timestamp']
        closes = indicators.get('close', [])
        volumes = indicators.get('volume', [])
        
        df = pd.DataFrame({
            'Close': closes, 'Volume': volumes, 'Timestamp': timestamps
        })
        
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
        df['Date'] = df['Date'].dt.tz_localize(None)
        df.set_index('Date', inplace=True)
        df = df.dropna()
        
        return df
        
    except Exception as e:
        return pd.DataFrame()

# --- 4. MATH ENGINE (Updated with Rank Logic) ---
def check_shockwaves(closes):
    """
    Checks for Volatility Shocks (Sanitized Language).
    """
    alerts = []
    try:
        if len(closes) < 6: return []
        
        curr = closes.iloc[-1]
        prev = closes.iloc[-2]     # Yesterday
        week_ago = closes.iloc[-6] # 5 Days Ago
        
        # 1. DAILY CHECK
        daily_pct = (curr - prev) / prev
        if daily_pct <= -0.10:
            alerts.append(f'<div class="alert-downside">📉 VOLATILITY ALERT: -{abs(daily_pct):.1%} Drop (24h)</div>')
        elif daily_pct >= 0.10:
            alerts.append(f'<div class="alert-upside">⚡ VOLATILITY ALERT: +{daily_pct:.1%} Surge (24h)</div>')
            
        # 2. WEEKLY CHECK
        weekly_pct = (curr - week_ago) / week_ago
        if weekly_pct >= 0.15: 
            alerts.append(f'<div class="alert-upside">🚀 MOMENTUM ACCELERATION: +{weekly_pct:.1%} (5-Day)</div>')
        elif weekly_pct <= -0.15:
            alerts.append(f'<div class="alert-downside">⚠️ TREND DETERIORATION: {weekly_pct:.1%} (5-Day)</div>')
            
        return alerts
    except: return []

def calculate_metrics(df):
    try:
        if len(df) < 252: return None # Require 1 year for Rank
        closes = df['Close']
        window = 20
        curr = closes.iloc[-1]
        last_date = df.index[-1]
        
        # 1. Z-Score (Short Term Momentum)
        mu = closes.rolling(window).mean().iloc[-1]
        sigma = closes.rolling(window).std().iloc[-1]
        z = (curr - mu) / sigma if sigma > 0 else 0
        
        # 2. Percentile Rank (252-Day Long Term Rarity) [NEW FEATURE]
        # Measures where today's price sits relative to the last year distribution
        rank_252 = df['Close'].rolling(window=252).apply(
            lambda x: percentileofscore(x, x.iloc[-1]), raw=False
        ).iloc[-1]

        # 3. Regime (SMA)
        sma200 = closes.rolling(200).mean().iloc[-1]
        regime = "BULL" if curr > sma200 else "BEAR"
        
        return {
            "price": curr, "z": z, "rank": rank_252, 
            "mu": mu, "date": last_date, "regime": regime
        }
    except: return None

def get_signal_logic(z, rank):
    """
    Returns the Signal Status based on the v6.2 Z-Score + Rank Matrix.
    """
    if z > 2.0:
        if rank > 95: return "High Risk Extension", "bear", "extension"
        return "Momentum Stretched", "neut", "stretched"
    elif 1.0 <= z <= 2.0:
        return "Momentum Active", "bull", "trend"
    elif -1.0 < z < 1.0:
        return "Noise / Consolidation", "neut", "noise"
    elif -2.0 <= z <= -1.0:
        return "Watchlist (Cooling)", "neut", "cooling"
    elif z < -2.0:
        if rank < 5: return "Stat. Oversold (Prime)", "bull", "oversold"
        return "Downside Extension", "bear", "dump"
    return "Unknown", "neut", "unknown"

# --- 5. MAIN UI ---
def main():
    st.title("🛡️ Quant Scanner v7.0 (Audit Compliant)")
    st.caption("Disclaimer: For informational purposes only. Not financial advice.")
    
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
            df = fetch_data_nuclear(target)
            
            if df.empty:
                st.error("🛑 **Connection Blocked:** Yahoo refused the data connection. Try again in 1 min.")
                return
            
            m = calculate_metrics(df)
            if not m: 
                st.error(f"⚠️ **Insufficient History:** {target} requires at least 252 days of trading data.")
                return
            
            # --- GET SIGNAL ---
            status_text, status_color, status_id = get_signal_logic(m['z'], m['rank'])

            # --- ALERTS ---
            alerts = check_shockwaves(df['Close'])
            for alert_html in alerts:
                st.markdown(alert_html, unsafe_allow_html=True)

            # --- CONTEXT BADGE ---
            if m['regime'] == "BULL":
                st.success(f"🟢 **MARKET CONTEXT: BULL REGIME** (Price > 200 SMA). Favor Long Setups.")
            else:
                st.error(f"🔴 **MARKET CONTEXT: BEAR REGIME** (Price < 200 SMA). Favor Caution/Shorts.")

            # --- METRICS ---
            c1, c2, c3 = st.columns(3)
            
            c1.metric("Price", f"${m['price']:.2f}", m['date'].strftime('%Y-%m-%d'))
            
            c2.metric("Z-Score (20d)", f"{m['z']:.2f}σ", 
                delta="Extended" if abs(m['z'])>2 else "Normal", delta_color="inverse",
                help="Short Term Momentum Deviation")
            
            c3.metric("Rarity Rank (1-Year)", f"{m['rank']:.1f}%", 
                help="Historical Percentile Rank (0% = Yearly Low, 100% = Yearly High)")
            
            st.divider()
            
            # --- NEW LOGIC MATRIX (v6.2) ---
            rows = [
                {"id": "oversold", "cond": "Oversold (Prime)", "z": "< -2.0", "rank": "< 5%", "verdict": "⭐ STAT OVERSOLD"},
                {"id": "trend", "cond": "Momentum Active", "z": "1.0 to 2.0", "rank": "Any", "verdict": "⚡ RIDE TREND"},
                {"id": "noise", "cond": "Noise / Chop", "z": "-1.0 to 1.0", "rank": "20-80%", "verdict": "💤 WAIT"},
                {"id": "stretched", "cond": "Stretched", "z": "> 2.0", "rank": "Any", "verdict": "⚠️ CAUTION"},
                {"id": "extension", "cond": "Extension", "z": "> 2.0", "rank": "> 95%", "verdict": "🛑 HIGH RISK"},
            ]
            
            html = ['<table class="matrix-table"><tr><th>Condition</th><th>Z-Score</th><th>Rarity (Rank)</th><th>Stat Signal</th></tr>']
            for row in rows:
                # Determine CSS class based on signal match
                css_class = "plain-row"
                if row['id'] == status_id:
                    if status_color == "bull": css_class = "signal-bull"
                    elif status_color == "bear": css_class = "signal-bear"
                    else: css_class = "signal-neut"
                
                verdict_display = f"✅ {row['verdict']}" if row['id'] == status_id else row['verdict']
                
                html.append(f'<tr class="{css_class}"><td>{row["cond"]}</td><td>{row["z"]}</td><td>{row["rank"]}</td><td>{verdict_display}</td></tr>')
            html.append('</table>')
            st.markdown("".join(html), unsafe_allow_html=True)
            
            st.divider()
            
            # --- CHART ---
            st.markdown("### 1-Year Deviation Distribution")
            hist_data = df['Close'].rolling(20).mean() # Simplified for viz
            # Actually recalculate historical Z-scores for the chart
            rolling_mean = df['Close'].rolling(20).mean()
            rolling_std = df['Close'].rolling(20).std()
            historical_z = (df['Close'] - rolling_mean) / rolling_std
            valid_z = historical_z.tail(252).dropna()

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=valid_z, nbinsx=40, 
                marker_color='#333', opacity=0.8, name='History'
            ))
            fig.add_vline(x=m['z'], line_width=3, line_color="#0066FF")
            fig.add_annotation(x=m['z'], y=10, text="CURRENT", font=dict(color="#0066FF", size=14, weight="bold"))
            
            fig.update_layout(template="plotly_white", height=350, margin=dict(t=20, b=20), showlegend=False, xaxis_title="Z-Score (Standard Deviations)")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
