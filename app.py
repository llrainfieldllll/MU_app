import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import percentileofscore
from curl_cffi import requests as crequests

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Quant Scanner v14.1", page_icon="🛡️")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .matrix-table { width: 100%; border-collapse: collapse; font-family: 'Arial', sans-serif; margin-top: 20px; }
    .matrix-table th { background-color: #000; color: #fff; padding: 12px; text-align: left; font-size: 14px; }
    .matrix-table td { padding: 12px; border-bottom: 1px solid #ddd; color: #333; font-size: 14px; }
    .row-bull { background-color: #e6fffa; border-left: 5px solid #00cc99; font-weight: bold; }
    .row-bear { background-color: #fff5f5; border-left: 5px solid #ff3333; font-weight: bold; }
    .row-neut { background-color: #f9f9f9; border-left: 5px solid #999; font-weight: bold; }
    .row-plain { background-color: #fff; color: #666; }
    div[data-testid="stMetricValue"] { font-size: 32px !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# --- ROBUST DATA ENGINE (Patched) ---
@st.cache_data(ttl=300)
def fetch_data(ticker):
    try:
        session = crequests.Session()
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"
        r = session.get(url, headers=headers, impersonate="chrome110")
        
        if r.status_code != 200: return None, f"API Error: {r.status_code}"

        data = r.json()
        if 'chart' not in data or 'result' not in data['chart']: return None, "Data Error"
        if not data['chart']['result']: return None, "No data found"

        result = data['chart']['result'][0]
        timestamps = result.get('timestamp')
        quote = result.get('indicators', {}).get('quote', [{}])[0]
        
        closes = quote.get('close')
        if not timestamps or not closes: return None, "Empty dataset"

        # --- SENIOR DEV FIX: ARRAY ALIGNMENT ---
        # Ensure Volume matches Close length. If missing, fill with 0.
        volumes = quote.get('volume')
        if not volumes or len(volumes) != len(closes):
            volumes = [0] * len(closes)

        df = pd.DataFrame({
            'Date': pd.to_datetime(timestamps, unit='s'),
            'Close': closes,
            'Volume': volumes
        })
        
        df.set_index('Date', inplace=True)
        # Drop rows where Price is missing (Yahoo sometimes sends Nulls in the middle of arrays)
        df.dropna(subset=['Close'], inplace=True) 
        
        return df, None
    except Exception as e:
        return None, f"System Error: {str(e)}"

# --- QUANT ENGINE ---
def calculate_metrics(df):
    # 1. EMA Mean (20-day)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # 2. Volatility
    df['Std_20'] = df['Close'].rolling(20).std()
    
    # 3. Z-Score (Safe Division)
    df['Z_Score'] = np.where(df['Std_20'] > 0, (df['Close'] - df['EMA_20']) / df['Std_20'], 0)
    
    # 4. Volume Ratio (Safe Division)
    # If Volume is 0 (Index/ETF), Ratio stays 0 to prevent crashes
    df['Vol_Median'] = df['Volume'].rolling(20).median()
    df['Vol_Ratio'] = np.where(df['Vol_Median'] > 0, df['Volume'] / df['Vol_Median'], 0)

    # 5. Percentile Rank
    df['Z_Rank'] = df['Z_Score'].rolling(252).apply(
        lambda x: percentileofscore(x, x.iloc[-1]), raw=False
    )
    
    return df

def get_signal(z, rank, vol):
    if pd.isna(z): return "DATA ERROR", "neut", "none"
    safe_rank = 50 if pd.isna(rank) else rank

    if z > 2.0:
        if safe_rank > 95: return "REVERSAL RISK", "bear", "extreme"
        if vol > 1.5: return "BREAKOUT CONFIRMED", "bull", "breakout"
        return "CAUTION (Extended)", "neut", "extended"
    elif 1.0 <= z <= 2.0:
        return "RIDE TREND", "bull", "trend"
    elif -1.0 < z < 1.0:
        return "WAIT / NOISE", "neut", "noise"
    elif z < -2.0:
        if safe_rank < 5: return "PRIME OVERSOLD", "bull", "oversold"
        return "DOWNSIDE INERTIA", "bear", "downside"
    return "NEUTRAL", "neut", "none"

# --- MAIN UI ---
def main():
    st.title("🛡️ Quant Scanner v14.1")
    st.caption("Disclaimer: Not financial advice.")
    
    col_input, col_rest = st.columns([1, 4])
    with col_input:
        ticker = st.text_input("Ticker", "MU").upper()
        run = st.button("Run Analysis", type="primary")

    if run:
        with st.spinner(f"Analyzing {ticker}..."):
            df, err = fetch_data(ticker)
            
            if err:
                st.error(f"🛑 {err}")
            elif len(df) < 20:
                st.error(f"⚠️ Insufficient data (Need 20+ days)")
            else:
                df = calculate_metrics(df)
                cur = df.iloc[-1]
                
                # Logic Variables
                rank_val = cur['Z_Rank']
                rank_display = "N/A" if pd.isna(rank_val) else f"{rank_val:.1f}%"
                sig_txt, sig_col, sig_id = get_signal(cur['Z_Score'], cur['Z_Rank'], cur['Vol_Ratio'])
                
                # Metrics
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Price", f"${cur['Close']:.2f}")
                c2.metric("Regime", "BULL" if cur['Close'] > df['EMA_20'].iloc[-1] else "BEAR")
                c3.metric("Z-Score", f"{cur['Z_Score']:.2f}σ")
                c4.metric("Rank (1-Year)", rank_display)
                c5.metric("Vol Ratio", f"{cur['Vol_Ratio']:.1f}x")
                
                st.divider()

                # Signal Banner
                if sig_col == "bull": st.success(f"**STATISTICAL BIAS:** {sig_txt}")
                elif sig_col == "bear": st.error(f"**STATISTICAL BIAS:** {sig_txt}")
                else: st.warning(f"**STATISTICAL BIAS:** {sig_txt}")

                # Matrix & Chart
                c_matrix, c_chart = st.columns([3, 2])
                
                with c_matrix:
                    st.subheader("Decision Logic")
                    matrix_rows = [
                        {"id": "breakout", "cond": "Breakout", "z": "> 2.0", "rank": "Any", "vol": "> 1.5x", "out": "🚀 BUY BREAKOUT"},
                        {"id": "extreme", "cond": "Extension", "z": "> 2.0", "rank": "> 95%", "vol": "Any", "out": "🛑 SELL / TRIM"},
                        {"id": "oversold", "cond": "Prime Oversold", "z": "< -2.0", "rank": "< 5%", "vol": "Any", "out": "⭐ BUY DIP"},
                        {"id": "trend", "cond": "Trend", "z": "1.0 to 2.0", "rank": "Any", "vol": "> 1.0x", "out": "🌊 HOLD / ADD"},
                        {"id": "noise", "cond": "Noise", "z": "-1.0 to 1.0", "rank": "20-80%", "vol": "Any", "out": "😴 WAIT"},
                    ]
                    
                    html = """
<table class="matrix-table">
    <thead>
        <tr>
            <th>Condition</th>
            <th>Z-Score</th>
            <th>Rarity</th>
            <th>Volume</th>
            <th>Verdict</th>
        </tr>
    </thead>
    <tbody>
"""
                    for row in matrix_rows:
                        is_active = (row['id'] == sig_id)
                        if is_active:
                            if sig_col == "bull": css = "row-bull"
                            elif sig_col == "bear": css = "row-bear"
                            else: css = "row-neut"
                            icon = "✅ "
                        else:
                            css = "row-plain"
                            icon = ""
                        
                        html += f"""
        <tr class="{css}">
            <td>{row['cond']}</td>
            <td>{row['z']}</td>
            <td>{row['rank']}</td>
            <td>{row['vol']}</td>
            <td>{icon}{row['out']}</td>
        </tr>"""
                    
                    html += "</tbody></table>"
                    st.markdown(html, unsafe_allow_html=True)

                with c_chart:
                    st.subheader("Is this Normal?")
                    valid_z = df['Z_Score'].tail(252).dropna()
                    
                    if len(valid_z) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=valid_z, nbinsx=40, 
                            marker_color='#444', opacity=0.8, name='History'
                        ))
                        fig.add_vline(x=cur['Z_Score'], line_width=4, line_color="#0066FF")
                        fig.add_annotation(
                            x=cur['Z_Score'], y=10, text="NOW", 
                            font=dict(color="#0066FF", size=16, weight="bold")
                        )
                        fig.update_layout(
                            template="plotly_white", height=300, 
                            xaxis_title="Z-Score Deviation", showlegend=False,
                            margin=dict(t=20, b=20, l=20, r=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Insufficient data for chart.")

if __name__ == "__main__":
    main()
