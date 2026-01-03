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

# --- MATRIX ENGINE ---
def render_reference_matrix(z_score, vol_ratio, rsi):
    z_abs = abs(z_score)
    direction = "UP" if z_score > 0 else "DOWN"
    
    # 1. Determine Active Key
    active_key = "normal"
    if z_abs >= 1.0 and z_abs < 2.0:
        active_key = "trending"
    elif z_abs >= 2.0:
        if vol_ratio > 1.2: active_key = "breakout"
        elif vol_ratio < 0.8: active_key = "exhaustion"
        else: active_key = "outlier"
    
    # 2. Dynamic Text Logic
    breakout_label = "🟠 BREAKOUT (Up)" if direction == "UP" else "🟠 WATERFALL (Crash)"
    exhaustion_label = "🔴 TOP REVERSAL" if direction == "UP" else "🟢 BOTTOM BOUNCE"
    
    # 3. Define Rows
    rows = [
        {"key": "normal", "cond": "Normal Noise", "z": "0.0 - 1.0 σ", "p": "> 19%", "vol": "Any", "rsi": "30 - 70", "verdict": "🔵 WAIT / NEUTRAL"},
        {"key": "trending", "cond": "Trending", "z": "1.0 - 2.0 σ", "p": "5% - 19%", "vol": "Normal", "rsi": "50 - 70", "verdict": "🟢 FOLLOW TREND"},
        {"key": "breakout", "cond": "High Momentum", "z": "> 2.0 σ", "p": "< 5%", "vol": "> 1.2x (High)", "rsi": "Extreme", "verdict": breakout_label},
        {"key": "exhaustion", "cond": "Exhaustion", "z": "> 2.0 σ", "p": "< 5%", "vol": "< 0.8x (Low)", "rsi": "Divergence", "verdict": exhaustion_label},
        {"key": "outlier", "cond": "Statistical Outlier", "z": "> 2.0 σ", "p": "< 5%", "vol": "Normal", "rsi": "Extreme", "verdict": "⚠️ ANOMALY (Caution)"}
    ]

    # 4. Build HTML
    html_parts = ['<table class="matrix-table">']
    html_parts.append('<tr><th>Market Condition</th><th>Z-Score</th><th>P-Value (Rarity)</th><th>Volume</th><th>RSI</th><th>Verdict</th></tr>')
    
    for row in rows:
        if row["key"] == active_key:
            if "breakout" in active_key: theme = "highlight-orange"
            elif "exhaustion" in active_key: theme = "highlight-red" if direction == "UP" else "highlight-green"
            elif "trending" in active_key: theme = "highlight-green"
            elif "outlier" in active_key: theme = "highlight-orange"
            else: theme = "highlight-blue"
            
            # Active Row
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
            # Inactive Row
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

# --- ROBUST DATA ENGINE (Fixes MU/MultiIndex Issues) ---
@st.cache_data(ttl=900, show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_market_data(ticker):
    try:
        # 1. Use download() instead of Ticker().history() - it's often more stable for single tickers
        # threads=False helps prevent "hanging" on specific symbols
        data = yf.download(ticker, period="6mo", interval="1d", progress=False, threads=False)
        
        if data.empty:
            st.warning(f"Yahoo returned empty data for {ticker}. Try a different ticker.")
            return pd.DataFrame()

        # 2. CRITICAL FIX: Flatten MultiIndex Columns
        # If columns are like [('Close', 'MU'), ('Volume', 'MU')], this fixes it
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # 3. Rename columns to ensure standard Capitalization (Close, Volume)
        # Sometimes yahoo returns "close" (lowercase)
        data.columns = [c.capitalize() for c in data.columns]

        # 4. Ensure we have the necessary columns
        if 'Close' not in data.columns or 'Volume' not in data.columns:
             # Try mapping 'Adj close' to 'Close' if it exists
            if 'Adj close' in data.columns:
                data['Close'] = data['Adj close']
            else:
                return pd.DataFrame()

        # 5. Clean Timezone
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # 6. Fill NaNs instead of dropping (preserves recent data)
        data = data.ffill().bfill()
        
        return data

    except Exception as e:
        st.error(f"Detailed Error for {ticker}: {str(e)}")
        return pd.DataFrame()

# --- UI RENDERER ---
def main():
    st.title("🛡️ Quant Scanner: Statistical Analyzer")
    st.error("**LEGAL DISCLAIMER:** For Educational Purposes Only. Not financial advice.")
    
    with st.sidebar:
        ticker = st.text_input("Ticker Symbol", "MU").upper()
        run_btn = st.button("Run Analysis", type="primary")

    if run_btn:
        with st.spinner(f"Scanning {ticker}..."):
            data = fetch_market_data(ticker)
            if data.empty: 
                st.error("Data fetch failed. Ticker may be invalid."); st.stop()
            
            m = calculate_metrics(data)
            if not m.get("valid", False): 
                st.error("Calculation error."); st.stop()

            # --- 0. FAT TAIL BANNER (Restored) ---
            is_fat_tail = abs(m['z']) > 2.0
            rarity_pct = m['p'] * 100

            if is_fat_tail:
                st.error(
                    f"🚨 FAT TAIL DETECTED: Price is {m['z']:.2f}σ from mean. "
                    f"Probability: {rarity_pct:.2f}%. This is a statistical outlier."
                )

            # --- 1. KEY METRICS (Expanded to 5 Columns) ---
            c1, c2, c3, c4, c5 = st.columns(5)
            
            c1.metric("Price", f"${m['price']:.2f}")
            c2.metric("Z-Score", f"{m['z']:.2f}σ", delta="Extreme" if is_fat_tail else "Normal", delta_color="inverse")
            c3.metric("Volume", f"{m['vol']:.1f}x")
            c4.metric("Rarity (P-Val)", f"{rarity_pct:.2f}%", help="Lower % means more rare/extreme.")
            
            # RSI Logic for Color
            rsi_val = m['rsi']
            rsi_status = "Neutral"
            if rsi_val > 70: rsi_status = "Overbought"
            elif rsi_val < 30: rsi_status = "Oversold"
            
            c5.metric("RSI (14)", f"{rsi_val:.1f}", delta=rsi_status, delta_color="off")

            st.divider()

            # --- 2. MATRIX ---
            st.subheader("📊 Decision Matrix")
            render_reference_matrix(m['z'], m['vol'], m['rsi'])

            st.divider()

            # --- 3. OBSERVATIONS ---
            gap = m['mu'] - m['price']
            direction_text = "above" if m['z'] > 0 else "below"
            
            st.markdown("### 📝 Statistical Observations")
            
            observation_text = f"""
            * **Rarity Analysis:** There is only a **{rarity_pct:.2f}% theoretical probability** of the price being this far {direction_text} the average.
            * **Mean Reversion Gap:** The 20-Day SMA is located at **${m['mu']:.2f}**. The distance between the current price and this statistical mean is **${gap:.2f}**.
            * **Momentum Context:** RSI is at **{m['rsi']:.1f}**. (Values >70 are typically Overbought, <30 are Oversold).
            """
            
            if is_fat_tail:
                st.warning(observation_text) # Warning color if fat tail
            else:
                st.info(observation_text)

            st.divider()

            # --- 4. VISUALIZATION ---
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
