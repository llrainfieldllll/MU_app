import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import percentileofscore
from curl_cffi import requests as crequests
from io import StringIO
from datetime import datetime, timedelta

# --- CONFIGURATION & STYLING ---
st.set_page_config(layout="wide", page_title="Quant Scanner Pro", page_icon="📊")

st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #303030;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    .big-font { font-size: 24px !important; font-weight: bold; color: #ffffff; }
    .small-font { font-size: 14px !important; color: #888; }
</style>
""", unsafe_allow_html=True)

# --- ROBUST DATA FETCHING (Anti-Blocking) ---
@st.cache_data(ttl=300)
def fetch_data(ticker):
    try:
        session = crequests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        }
        
        # Fetch 2 years (730 days) to ensure valid 252d rank calculation
        end_date = int(datetime.now().timestamp())
        start_date = int((datetime.now() - timedelta(days=730)).timestamp())
        
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_date}&period2={end_date}&interval=1d&events=history&includeAdjustedClose=true"
        
        response = session.get(url, headers=headers, impersonate="chrome110")
        
        if response.status_code != 200:
            return None, f"Error {response.status_code}: Could not fetch data."
            
        df = pd.read_csv(StringIO(response.text))
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df, None
    except Exception as e:
        return None, str(e)

# --- QUANT LOGIC ENGINE ---
def calculate_metrics(df):
    # 1. Z-Score (20-day Short Term Momentum)
    df['Mean_20'] = df['Close'].rolling(window=20).mean()
    df['Std_20'] = df['Close'].rolling(window=20).std()
    df['Z_Score'] = (df['Close'] - df['Mean_20']) / df['Std_20']
    
    # 2. Percentile Rank (252-day Long Term Rarity)
    # Calculates where today's price sits relative to the last trading year
    df['Rank_252'] = df['Close'].rolling(window=252).apply(
        lambda x: percentileofscore(x, x.iloc[-1]), raw=False
    )
    
    # 3. Regime Filter (SMA 200)
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    return df

def get_signal_status(z_score, rank):
    # Interpretation Matrix
    if z_score > 2.0:
        if rank > 95: return "🛑 STATISTICAL EXTENSION (High Risk)", "bear" # Rare + Stretched
        return "⚠️ MOMENTUM STRETCHED", "neut" # Stretched but not historically rare (Bull run)
    elif 1.0 <= z_score <= 2.0:
        return "⚡ MOMENTUM ACTIVE", "bull" # Sweet spot
    elif -1.0 < z_score < 1.0:
        return "💤 NOISE / CONSOLIDATION", "neut"
    elif -2.0 <= z_score <= -1.0:
        return "👀 WATCHLIST (Cooling)", "neut"
    elif z_score < -2.0:
        if rank < 5: return "⭐ STATISTICAL OVERSOLD (Prime)", "bull" # Rare + Dumped
        return "📉 DOWNSIDE EXTENSION", "bear" # Falling knife
    return "UNKNOWN", "neut"

# --- MAIN UI ---
st.title("🛡️ Quant Scanner v6.0 (Audit Compliant)")
st.caption("Disclaimer: For informational purposes only. Not financial advice.")

col1, col2 = st.columns([1, 3])

with col1:
    ticker = st.text_input("Ticker Symbol", value="MU").upper()
    if st.button("Run Analysis", type="primary"):
        with st.spinner(f"Fetching data for {ticker}..."):
            df, error = fetch_data(ticker)
            
            if error:
                st.error(error)
            else:
                df = calculate_metrics(df)
                latest = df.iloc[-1]
                
                # Metric Display Logic
                rank_val = latest['Rank_252']
                rank_str = "N/A" if pd.isna(rank_val) else f"{rank_val:.1f}%"
                
                status_text, status_color = get_signal_status(latest['Z_Score'], rank_val)
                
                # Cards
                st.markdown(f"""
                <div class="metric-card">
                    <div class="small-font">Price</div>
                    <div class="big-font">${latest['Close']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="small-font">Z-Score (20d)</div>
                    <div class="big-font">{latest['Z_Score']:.2f}σ</div>
                </div>
                <div class="metric-card">
                    <div class="small-font">Rarity (1y Rank)</div>
                    <div class="big-font">{rank_str}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.divider()
                if status_color == "bull": st.success(status_text)
                elif status_color == "bear": st.error(status_text)
                else: st.warning(status_text)

with col2:
    if 'df' in locals() and not error:
        # Reference Table
        st.subheader("Statistical Interpretation Matrix")
        matrix = pd.DataFrame({
            "Signal Type": ["Oversold (Prime)", "Momentum Active", "Noise", "Stretched", "Extension"],
            "Z-Score": ["< -2.0", "1.0 to 2.0", "-1.0 to 1.0", "> 2.0", "> 2.0"],
            "Rarity (Rank)": ["< 5%", "Any", "20-80%", "Any", "> 95%"],
            "Meaning": ["Price is abnormally low (Reversion likely)", "Strong Trend (Buy Strength)", "No Statistical Edge", "Getting Expensive", "Extreme Deviation (Pullback likely)"]
        })
        st.table(matrix)

        # Histogram
        st.subheader("1-Year Deviation Distribution")
        valid_history = df.tail(252).dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=valid_history['Z_Score'], nbinsx=40, 
            marker_color='#333', opacity=0.8, name='History'
        ))
        fig.add_vline(x=latest['Z_Score'], line_width=3, line_color="#0066FF")
        fig.add_annotation(x=latest['Z_Score'], y=10, text="CURRENT", font=dict(color="#0066FF", size=14, weight="bold"))
        
        fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Z-Score (Standard Deviations)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
