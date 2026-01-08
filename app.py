import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from curl_cffi import requests as crequests
from datetime import datetime
import pytz

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Quant Scanner v5.0 (Hybrid)", layout="wide", page_icon="🛡️")

# --- 2. CSS & STYLING ---
st.markdown("""
<style>
    .matrix-table { width: 100%; border-collapse: collapse; font-family: 'Roboto Mono', monospace; font-size: 14px; }
    .matrix-table th { background-color: #000; color: #FFF; padding: 10px; text-align: left; }
    .matrix-table td { padding: 10px; border-bottom: 1px solid #ddd; color: #000; font-weight: 500; }
    
    /* State Colors */
    .state-anomaly { background-color: #fff3cd; border-left: 5px solid #ffc107; color: #856404; }
    .state-breakout { background-color: #d4edda; border-left: 5px solid #28a745; color: #155724; }
    .state-exhaustion { background-color: #f8d7da; border-left: 5px solid #dc3545; color: #721c24; }
    .state-trend { background-color: #d1ecf1; border-left: 5px solid #17a2b8; color: #0c5460; }
    .state-neutral { background-color: #f8f9fa; border-left: 5px solid #6c757d; color: #343a40; }
    .plain { background-color: white; color: #ccc; }
    
    /* Alerts */
    .alert-box { padding: 10px; margin-bottom: 10px; border-radius: 5px; font-weight: bold; border: 1px solid; }
    .alert-up { background-color: #e6fffa; border-color: #2c7a7b; color: #2c7a7b; }
    .alert-down { background-color: #fff5f5; border-color: #c53030; color: #c53030; }
</style>
""", unsafe_allow_html=True)

# --- 3. THE "STEALTH HYBRID" ENGINE ---
def fetch_data_hybrid(ticker):
    """
    Attempts to fetch data using yfinance first (cleaner).
    Fails over to curl_cffi (Chrome 110 impersonation) if blocked.
    """
    # 1. Try Official API (yfinance)
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if not df.empty and len(df) > 100:
            if isinstance(df.columns, pd.MultiIndex):
                return df.xs('Close', axis=1, level=1)
            return df['Close']
    except:
        pass # Fail silently to backup
        
    # 2. Try Nuclear Option (curl_cffi)
    try:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"
        r = crequests.get(url, impersonate="chrome110", timeout=8)
        
        if r.status_code == 200:
            data = r.json()['chart']['result'][0]
            closes = data['indicators']['quote'][0]['close']
            timestamps = data['timestamp']
            
            df = pd.DataFrame({'Close': closes, 'Date': pd.to_datetime(timestamps, unit='s')})
            # Clean up: set index and drop NaNs
            return df.set_index('Date')['Close'].dropna()
    except:
        return None
        
    return None

# --- 4. MATH ENGINE (Refined) ---
def calculate_metrics(closes):
    try:
        if len(closes) < 200: return None
        
        curr_price = float(closes.iloc[-1])
        
        # 1. Z-Score (The Distance)
        roll = closes.rolling(20)
        mu = roll.mean().iloc[-1]
        sigma = roll.std().iloc[-1]
        
        # Avoid division by zero
        if sigma == 0: sigma = 0.001
        z_score = (curr_price - mu) / sigma
        
        # 2. Percentile Rank (The Rarity) - KEY FIX
        # We look at the Z-Score history of the last 6 months (126 days)
        past_z = (closes - closes.rolling(20).mean()) / closes.rolling(20).std()
        recent_z_history = past_z.tail(126).dropna()
        
        if not recent_z_history.empty:
            # Logic: What % of days had a LOWER Z-score than today?
            pct_rank = (recent_z_history < z_score).mean() * 100
        else:
            pct_rank = 50.0 # Default fallback
        
        # 3. Market Regime (The Trend)
        # Using EMA for faster reaction (Red Team Fix)
        ema50 = closes.ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = closes.ewm(span=200, adjust=False).mean().iloc[-1]
        
        regime = "NEUTRAL"
        if curr_price > ema200:
            regime = "BULL" if ema50 > ema200 else "RECOVERY"
        else:
            regime = "BEAR" if ema50 < ema200 else "WARNING"

        # 4. Stop Loss (Risk Management)
        stop_loss = curr_price - (3 * sigma)

        return {
            "price": curr_price,
            "z_score": z_score,
            "pct_rank": pct_rank,
            "regime": regime,
            "mu": mu,
            "prev_close": closes.iloc[-2],
            "stop_loss": stop_loss
        }
    except Exception as e:
        return None

def get_verdict(m):
    z = m['z_score']
    p = m['pct_rank']
    r = m['regime']
    
    # Logic combining Regime, Distance (Z), and Rarity (P)
    
    # EXTENSIONS (Overbought)
    if z > 2.0 and p > 95: return "exhaustion", "🛑 EXTENDED (Risk)"
    if z > 2.0: return "breakout", "🚀 BREAKOUT"
    
    # OVERSOLD (Dip Buying)
    if z < -2.0:
        if r == "BULL" and p < 5: return "anomaly", "⭐⭐⭐ PRIME BUY" # Confluence
        if r == "BULL": return "trend", "⭐⭐ DIP WATCH"
        if r == "BEAR": return "exhaustion", "⛔ FALLING KNIFE"
        
    # TRENDING
    if 1.0 < z < 2.0 and r == "BULL": return "trend", "🌊 RIDE TREND"
    if -2.0 < z < -1.0 and r == "BEAR": return "trend", "📉 DOWNTREND"
    
    return "neutral", "😴 CHOP"

# --- 5. UI ---
def main():
    st.title("🛡️ Quant Scanner v5.0 (Red Teamed)")
    
    # Sidebar
    ticker = st.sidebar.text_input("Ticker", "NVDA").upper()
    run = st.sidebar.button("Run Analysis", type="primary")
    
    if run and ticker:
        with st.spinner(f"Triangulating Data for {ticker}..."):
            closes = fetch_data_hybrid(ticker)
            
            if closes is None or closes.empty:
                st.error(f"Data Source Failure for {ticker}. Try again.")
                return

            m = calculate_metrics(closes)
            if not m:
                st.error("Insufficient historical data (Need 200+ days).")
                return
                
            state_id, verdict_text = get_verdict(m)
            
            # --- HEADER METRICS ---
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Price", f"${m['price']:.2f}")
            c2.metric("Regime", m['regime'], 
                      delta="Safe" if m['regime']=="BULL" else "Danger", 
                      delta_color="normal" if m['regime']=="BULL" else "inverse")
            c3.metric("Z-Score", f"{m['z_score']:.2f}σ", help="Standard Deviations from 20-Day Mean")
            
            # The Critical Metric: Percentile
            rank_display = f"{m['pct_rank']:.0f}%"
            c4.metric("Hist. Rank", rank_display, help="How rare is this price relative to the last 6 months?")
            
            c5.metric("Stop Loss", f"${m['stop_loss']:.2f}", help="3 Sigma below price")

            st.divider()
            
            # --- ALERTS ---
            daily_chg = (m['price'] - m['prev_close']) / m['prev_close']
            if daily_chg > 0.05:
                st.markdown(f'<div class="alert-box alert-up">⚡ HIGH VELOCITY: +{daily_chg:.1%} Today</div>', unsafe_allow_html=True)
            elif daily_chg < -0.05:
                st.markdown(f'<div class="alert-box alert-down">📉 HIGH VELOCITY: {daily_chg:.1%} Today</div>', unsafe_allow_html=True)

            # --- THE MATRIX ---
            rows = [
                {"id": "breakout", "cond": "Breakout", "z": "> 2.0", "p": "Any", "res": "🚀 BREAKOUT"},
                {"id": "exhaustion", "cond": "Extension", "z": "> 2.0", "p": "> 95%", "res": "🛑 REVERSAL RISK"},
                {"id": "anomaly", "cond": "Prime Oversold", "z": "< -2.0", "p": "< 5%", "res": "⭐⭐⭐ PRIME BUY"},
                {"id": "trend", "cond": "Trend", "z": "1.0 to 2.0", "p": "Any", "res": "🌊 RIDE TREND"},
                {"id": "neutral", "cond": "Noise", "z": "-1.0 to 1.0", "p": "20% - 80%", "res": "😴 WAIT"},
            ]
            
            html = ['<table class="matrix-table"><tr><th>Condition</th><th>Z-Score</th><th>Percentile (Rarity)</th><th>Verdict</th></tr>']
            for row in rows:
                css = f"state-{row['id']}" if row['id'] == state_id else "plain"
                res = f"✅ {row['res']}" if row['id'] == state_id else row['res']
                html.append(f'<tr class="{css}"><td>{row["cond"]}</td><td>{row["z"]}</td><td>{row["p"]}</td><td>{res}</td></tr>')
            html.append('</table>')
            st.markdown("".join(html), unsafe_allow_html=True)
            
            # --- CHART ---
            # Visualizing the Percentile Rank
            fig = go.Figure()
            
            # Histogram of past Z-scores
            past_z = (closes - closes.rolling(20).mean()) / closes.rolling(20).std()
            recent_z = past_z.tail(126).dropna()
            
            fig.add_trace(go.Histogram(
                x=recent_z, nbinsx=30, name="6-Month History",
                marker_color='#e0e0e0', opacity=0.7
            ))
            
            # Current Z-Score Line
            color = "red" if abs(m['z_score']) > 2 else "blue"
            fig.add_vline(x=m['z_score'], line_width=3, line_color=color)
            fig.add_annotation(x=m['z_score'], y=10, text=f"CURRENT<br>{m['z_score']:.2f}σ", 
                               showarrow=False, yshift=20, font=dict(color=color, weight="bold"))
            
            fig.update_layout(
                title="Is this Normal? (Current Z-Score vs. 6-Month History)",
                template="plotly_white", height=300, 
                xaxis_title="Z-Score", yaxis_title="Frequency (Days)",
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
