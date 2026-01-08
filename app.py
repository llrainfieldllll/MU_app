import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import percentileofscore
from curl_cffi import requests as crequests
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Quant Scanner v9.1", page_icon="⚖️")

st.markdown("""
<style>
    .metric-card { background-color: #0E1117; border: 1px solid #303030; padding: 20px; border-radius: 10px; text-align: center; }
    .big-font { font-size: 24px !important; font-weight: bold; color: #ffffff; }
    .small-font { font-size: 14px !important; color: #888; }
</style>
""", unsafe_allow_html=True)

# --- DATA ENGINE (JSON API v8) ---
@st.cache_data(ttl=300)
def fetch_data(ticker):
    try:
        session = crequests.Session()
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"
        r = session.get(url, headers=headers, impersonate="chrome110")
        
        if r.status_code != 200: return None, "Data Source Error: Connection Refused"

        data = r.json()
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quote = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Date': pd.to_datetime(timestamps, unit='s'),
            'Close': quote['close'],
            'Volume': quote['volume'] 
        })
        df.set_index('Date', inplace=True)
        return df.dropna(), None
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

# --- QUANT ENGINE (EMA Logic) ---
def calculate_metrics(df):
    # 1. EMA Mean (20-day)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # 2. Volatility (Std Dev)
    df['Std_20'] = df['Close'].rolling(20).std()
    
    # 3. EMA Z-Score
    df['Z_Score'] = (df['Close'] - df['EMA_20']) / df['Std_20']
    
    # 4. Volume Ratio
    df['Vol_Median'] = df['Volume'].rolling(20).median()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_Median']

    # 5. Momentum Rarity (Z-Score Rank)
    df['Z_Rank'] = df['Z_Score'].rolling(252).apply(
        lambda x: percentileofscore(x, x.iloc[-1]), raw=False
    )
    
    return df

def get_signal(z, rank, vol):
    # LEGAL AUDIT FIX: 
    # Removed "BUY/SELL" commands. Replaced with descriptive Statistical States.
    
    if pd.isna(z) or pd.isna(rank): return "⚪ DATA INSUFFICIENT", "neut"

    if z > 2.0:
        if rank > 95: return "🛑 STATISTICAL EXTREME (High Risk)", "bear"
        if vol > 1.5: return "🚀 VOLATILITY EXPANSION (Vol Support)", "bull"
        return "⚠️ EXTENDED (Low Volume)", "neut"
    elif 1.0 <= z <= 2.0:
        if vol > 1.0: return "⚡ MOMENTUM ACTIVE", "bull"
        return "🌊 TREND FOLLOWING", "bull"
    elif -1.0 < z < 1.0:
        return "💤 CONSOLIDATION", "neut"
    elif z < -2.0:
        if vol > 2.0: return "📉 CLIMAX ACTION (Watch Reversal)", "bull"
        return "🩸 DOWNSIDE INERTIA", "bear"
    return "⚪ NEUTRAL", "neut"

# --- MAIN UI ---
def main():
    st.title("⚖️ Quant Scanner v9.1 (Compliance Ready)")
    
    # MANDATORY DISCLAIMER (Prominent Placement)
    st.info("⚠️ **DISCLAIMER:** This application is for **educational and informational purposes only**. "
            "It visualizes statistical data and does not constitute financial advice, buy/sell recommendations, "
            "or a solicitation to trade. Past statistical performance (Z-Score) does not guarantee future results.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker = st.text_input("Ticker", "MU").upper()
        if st.button("Run Statistical Analysis", type="primary"):
            with st.spinner(f"Querying Data for {ticker}..."):
                df, err = fetch_data(ticker)
                
                if err:
                    st.error(err)
                elif len(df) < 20:
                    st.error(f"Insufficient Data Depth for {ticker}")
                else:
                    df = calculate_metrics(df)
                    cur = df.iloc[-1]
                    
                    rank_val = cur['Z_Rank']
                    rank_display = "N/A" if pd.isna(rank_val) else f"{rank_val:.1f}%"
                    sig_txt, sig_col = get_signal(cur['Z_Score'], cur['Z_Rank'], cur['Vol_Ratio'])
                    
                    # --- METRICS ---
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="small-font">Last Price</div>
                        <div class="big-font">${cur['Close']:.2f}</div>
                    </div>
                    <div style="margin-top:10px;"></div>
                    <div class="metric-card">
                        <div class="small-font">EMA Deviation (Z)</div>
                        <div class="big-font">{cur['Z_Score']:.2f}σ</div>
                    </div>
                    <div style="margin-top:10px;"></div>
                    <div class="metric-card">
                        <div class="small-font">Relative Vol</div>
                        <div class="big-font">{cur['Vol_Ratio']:.1f}x</div>
                    </div>
                    <div style="margin-top:10px;"></div>
                    <div class="metric-card">
                        <div class="small-font">1-Year Rarity</div>
                        <div class="big-font">{rank_display}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.divider()
                    if sig_col == "bull": st.success(f"**Statistical Bias:** {sig_txt}")
                    elif sig_col == "bear": st.error(f"**Statistical Bias:** {sig_txt}")
                    else: st.warning(f"**Statistical Bias:** {sig_txt}")

    with col2:
        if 'df' in locals() and not err and len(df) > 20:
            st.subheader("Logic Matrix (Descriptive)")
            # LEGAL FIX: Renamed 'Verdict' to 'Model Output'
            # LEGAL FIX: Renamed 'BUY/SELL' to 'ACCUMULATE/DISTRIBUTE/WAIT'
            matrix = pd.DataFrame({
                "Condition": ["Vol Expansion", "Trend Following", "Extended", "Extreme", "Climax"],
                "EMA Z-Score": ["> 2.0", "1.0 to 2.0", "> 2.0", "> 2.0", "< -2.0"],
                "Volume": ["> 1.5x", "> 1.0x", "< 1.0x", "Any", "> 2.0x"],
                "Model Output": ["ACCUMULATION ZONE", "TREND BIAS", "CAUTION", "DISTRIBUTION RISK", "REVERSAL WATCH"]
            })
            st.table(matrix)

            # Chart
            st.subheader("1-Year EMA Deviation Distribution")
            valid_z = df['Z_Score'].tail(252).dropna()
            
            if len(valid_z) > 0:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=valid_z, nbinsx=40, marker_color='#444', name="History"))
                fig.add_vline(x=cur['Z_Score'], line_color="#0066FF", line_width=3)
                fig.add_annotation(x=cur['Z_Score'], y=10, text="CURRENT", font=dict(color="#0066FF", size=14, weight="bold"))
                
                fig.update_layout(
                    template="plotly_white", 
                    height=350, 
                    xaxis_title="Z-Score (Standard Deviations)",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
