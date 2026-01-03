import streamlit as st

# --- CRITICAL: PAGE CONFIG MUST BE FIRST ---
st.set_page_config(page_title="Quant Scanner: Statistical Analyzer", layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
from tenacity import retry, stop_after_attempt, wait_fixed
import socket

# --- SAFETY: PREVENT HANGS ---
socket.setdefaulttimeout(5)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .matrix-table { width: 100%; border-collapse: collapse; font-size: 14px; margin-bottom: 20px; }
    .matrix-table th { background-color: #262730; color: white; padding: 10px; text-align: left; border-bottom: 2px solid #444; }
    .matrix-table td { padding: 10px; border-bottom: 1px solid #ddd; color: #333; }
    
    /* Highlight Classes */
    .highlight-blue { background-color: #e3f2fd; border-left: 5px solid #2196f3; font-weight: bold; }
    .highlight-green { background-color: #e8f5e9; border-left: 5px solid #4caf50; font-weight: bold; }
    .highlight-orange { background-color: #fff3e0; border-left: 5px solid #ff9800; font-weight: bold; }
    .highlight-red { background-color: #ffebee; border-left: 5px solid #f44336; font-weight: bold; }
    
    /* Faded Rows */
    .faded { color: #999 !important; opacity: 0.7; }
</style>
""", unsafe_allow_html=True)

# --- 1. MATRIX ENGINE (UI Logic) ---
def render_reference_matrix(z_score, vol_ratio, rsi):
    z_abs = abs(z_score)
    direction = "UP" if z_score > 0 else "DOWN"
    
    # Determine Active Key
    active_key = "normal"
    if z_abs >= 1.0 and z_abs < 2.0:
        active_key = "trending"
    elif z_abs >= 2.0:
        if vol_ratio > 1.2: active_key = "breakout"
        elif vol_ratio < 0.8: active_key = "exhaustion"
        else: active_key = "outlier"
    
    # Dynamic Text Logic
    breakout_label = "🟠 BREAKOUT (Up)" if direction == "UP" else "🟠 WATERFALL (Crash)"
    exhaustion_label = "🔴 TOP REVERSAL" if direction == "UP" else "🟢 BOTTOM BOUNCE"
    
    # Define Rows
    rows = [
        {"key": "normal", "cond": "Normal Noise", "z": "0.0 - 1.0 σ", "p": "> 19%", "vol": "Any", "rsi": "30 - 70", "verdict": "🔵 WAIT / NEUTRAL"},
        {"key": "trending", "cond": "Trending", "z": "1.0 - 2.0 σ", "p": "5% - 19%", "vol": "Normal", "rsi": "50 - 70", "verdict": "🟢 FOLLOW TREND"},
        {"key": "breakout", "cond": "High Momentum", "z": "> 2.0 σ", "p": "< 5%", "vol": "> 1.2x (High)", "rsi": "Extreme", "verdict": breakout_label},
        {"key": "exhaustion", "cond": "Exhaustion", "z": "> 2.0 σ", "p": "< 5%", "vol": "< 0.8x (Low)", "rsi": "Divergence", "verdict": exhaustion_label},
        {"key": "outlier", "cond": "Statistical Outlier", "z": "> 2.0 σ", "p": "< 5%", "vol": "Normal", "rsi": "Extreme", "verdict": "⚠️ ANOMALY (Caution)"}
    ]

    # Build HTML
    html_parts = ['<table class="matrix-table">']
    html_parts.append('<tr><th>Market Condition</th><th>Z-Score</th><th>P-Value (Rarity)</th><th>Volume</th><th>RSI</th><th>Verdict</th></tr>')
    
    for row in rows:
        if row["key"] == active_key:
            if "breakout" in active_key: theme = "highlight-orange"
            elif "exhaustion" in active_key: theme = "highlight-red" if direction == "UP" else "highlight-green"
            elif "trending" in active_key: theme = "highlight-green"
            elif "outlier" in active_key: theme = "highlight-orange"
            else: theme = "highlight-blue"
            
            row_html = (
                f'<tr class="{theme}">'
                f'<td>👉 {row["cond"]}</td>'
                f'<td>{row["z"]}</td>'
                f'<td>{row["p"]}</td>'
                f'<td>{row["vol"]}</td>'
                f'<td>{row["rsi"]}</td>'
                f'<td>{row["verdict"]}</td>'
                '</tr>'
            )
        else:
            row_html = (
                '<tr class="faded">'
                f'<td>{row["cond"]}</td>'
                f'<td>{row["z"]}</td>'
                f'<td>{row["p"]}</td>'
                f'<td>{row["vol"]}</td>'
                f'<td>{row["rsi"]}</td>'
                f'<td>{row["verdict"]}</td>'
                '</tr>'
            )
        html_parts.append(row_html)
            
    html_parts.append('</table>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)
    return active_key

# --- 2. DATA ENGINE (Red Team Approved) ---
@st.cache_data(ttl=900, show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_market_data(ticker):
    try:
        # Use download() with threads=False for maximum stability
        data = yf.download(ticker, period="6mo", interval="1d", progress=False, threads=False)
        
        if data.empty:
            return pd.DataFrame()

        # FIX: Flatten MultiIndex columns (e.g. ('Close', 'MU') -> 'Close')
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # FIX: Normalize column names to Title Case (Close, Volume)
        data.columns = [c.capitalize() for c in data.columns]

        # FIX: Ensure critical columns exist
        if 'Close' not in data.columns:
            if 'Adj close' in data.columns:
                data['Close'] = data['Adj close']
            else:
                return pd.DataFrame() # Fail if no price data

        # Fix Timezone
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # Fill gaps
        data = data.ffill().bfill()
        
        return data

    except Exception:
        return pd.DataFrame()

# --- 3. MATH ENGINE (Restored) ---
def calculate_metrics(df):
    try:
        # Sanity check
        if df.empty or 'Close' not in df.columns:
            return {"valid": False}
            
        prices = df['Close']
        volumes = df['Volume']
        current_price = prices.iloc[-1]
        current_volume = volumes.iloc[-1]
        
        # Z-Score Calc
        analysis_slice = prices.tail(20)
        mu = analysis_slice.mean()
        sigma = analysis_slice.std()
        
        if sigma == 0: 
            z_score = 0; p_value = 0.5
        else:
            z_score = (current_price - mu) / sigma
            if z_score > 0: p_value = 1 - t.cdf(z_score, df=5)
            else: p_value = t.cdf(z_score, df=5)
        
        # Volume Ratio
        vol_avg = volumes.tail(20).median()
        # Prevent div/0
        vol_ratio = (current_volume / vol_avg) if (vol_avg > 0 and not np.isnan(vol_avg)) else 1.0

        # RSI Calc
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        
        if len(avg_loss) > 0 and pd.notna(avg_loss.iloc[-1]):
            last_loss = avg_loss.iloc[-1]
            last_gain = avg_gain.iloc[-1]
            if last_loss == 0: 
                rsi = 100
            else:
                rs = last_gain / last_loss
                rsi = 100 - (100 / (1 + rs))
        else: 
            rsi = 50

        return {"price": current_price, "mu": mu, "z": z_score, "p": p_value, "vol": vol_ratio, "rsi": rsi, "valid": True}
    except Exception: 
        return {"valid": False}

# --- 4. MAIN UI ---
def main():
    st.title("🛡️ Quant Scanner: Statistical Analyzer")
    st.error("**LEGAL DISCLAIMER:** For Educational Purposes Only. Not financial advice.")
    
    with st.sidebar:
        # Input cleanup: Upper case and strip spaces
        ticker_input = st.text_input("Ticker Symbol", "MU")
        ticker = ticker_input.upper().strip()
        run_btn = st.button("Run Analysis", type="primary")

    if run_btn:
        with st.spinner(f"Scanning {ticker}..."):
            # 1. Fetch
            data = fetch_market_data(ticker)
            if data.empty: 
                st.error(f"Data fetch failed for '{ticker}'. Check spelling or Yahoo API status."); st.stop()
            
            # 2. Calculate (The previously missing function)
            m = calculate_metrics(data)
            if not m.get("valid", False): 
                st.error("Calculation error. Insufficient data points."); st.stop()

            # --- Logic: Fat Tail Banner ---
            is_fat_tail = abs(m['z']) > 2.0
            rarity_pct = m['p'] * 100

            if is_fat_tail:
                st.error(
                    f"🚨 FAT TAIL DETECTED: Price is {m['z']:.2f}σ from mean. "
                    f"Probability: {rarity_pct:.2f}%. This is a statistical outlier."
                )

            # --- Metrics Grid ---
            c1, c2, c3, c4, c5 = st.columns(5)
            
            c1.metric("Price", f"${m['price']:.2f}")
            c2.metric("Z-Score", f"{m['z']:.2f}σ", delta="Extreme" if is_fat_tail else "Normal", delta_color="inverse")
            c3.metric("Volume", f"{m['vol']:.1f}x")
            c4.metric("Rarity (P-Val)", f"{rarity_pct:.2f}%", help="Lower % means more rare/extreme.")
            
            # RSI Styling
            rsi_val = m['rsi']
            rsi_status = "Neutral"
            if rsi_val > 70: rsi_status = "Overbought"
            elif rsi_val < 30: rsi_status = "Oversold"
            
            c5.metric("RSI (14)", f"{rsi_val:.1f}", delta=rsi_status, delta_color="off")

            st.divider()

            # --- Matrix ---
            st.subheader("📊 Decision Matrix")
            render_reference_matrix(m['z'], m['vol'], m['rsi'])

            st.divider()

            # --- Observations ---
            gap = m['mu'] - m['price']
            direction_text = "above" if m['z'] > 0 else "below"
            
            st.markdown("### 📝 Statistical Observations")
            
            observation_text = f"""
            * **Rarity Analysis:** There is only a **{rarity_pct:.2f}% theoretical probability** of the price being this far {direction_text} the average.
            * **Mean Reversion Gap:** The 20-Day SMA is located at **${m['mu']:.2f}**. The distance between the current price and this statistical mean is **${gap:.2f}**.
            * **Momentum Context:** RSI is at **{m['rsi']:.1f}**. (Values >70 are typically Overbought, <30 are Oversold).
            """
            
            if is_fat_tail:
                st.warning(observation_text)
            else:
                st.info(observation_text)

            st.divider()

            # --- Visualization ---
            x = np.linspace(-4, 4, 1000)
            y = t.pdf(x, df=5)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#333'), name='Dist'))
            z = m['z']
            line_col = "#FF4B4B" if abs(z) >= 2 else "#2ECC71"
            fig.add_vline(x=z, line_width=2, line_dash="dash", line_color=line_col)
            fig.add_annotation(x=z, y=0.35, text=f"CURRENT<br>{z:.2f}σ", showarrow=True, arrowhead=2, font=dict(color=line_col))
            fig.update_layout(template="plotly_white", height=350, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
