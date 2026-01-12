import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import percentileofscore, t
from curl_cffi import requests as crequests

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Quant Scanner v24.3 (Tech Context)", page_icon="🛡️")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    
    /* Hero Card Styling */
    .hero-card {
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .hero-title { font-size: 24px; font-weight: bold; margin-bottom: 5px; }
    .hero-subtitle { font-size: 16px; opacity: 0.95; font-weight: 500; letter-spacing: 0.5px; }
    
    /* Gradients */
    .hero-bull { background: linear-gradient(135deg, #00b09b, #96c93d); }
    .hero-bear { background: linear-gradient(135deg, #ff5f6d, #ffc371); }
    .hero-yellow { background: linear-gradient(135deg, #f7971e, #ffd200); color: #222 !important; }
    .hero-neut { background: linear-gradient(135deg, #8e9eab, #eef2f3); color: #333 !important; }

    /* Metric Styling */
    div[data-testid="stMetricValue"] { font-size: 20px !important; font-weight: 700 !important; font-family: 'Roboto Mono', monospace; }
    div[data-testid="stMetricLabel"] { font-size: 12px !important; color: #666; }
    
    .stExpander { border: none !important; box-shadow: none !important; background-color: transparent !important; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'data' not in st.session_state: st.session_state.data = None
if 'analyzed_ticker' not in st.session_state: st.session_state.analyzed_ticker = "MU" 

# --- DATA ENGINE ---
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
        
        closes = quote.get('close', [])
        highs = quote.get('high', [])
        lows = quote.get('low', [])
        opens = quote.get('open', [])
        volumes = quote.get('volume', [])

        if not timestamps or not closes: return None, "Empty dataset"
        
        target_len = len(closes)
        def pad_list(lst, target, fill_source):
            if not lst: return fill_source 
            if len(lst) < target: return lst + [lst[-1]] * (target - len(lst))
            return lst[:target]

        highs = pad_list(highs, target_len, closes)
        lows = pad_list(lows, target_len, closes)
        opens = pad_list(opens, target_len, closes)
        volumes = pad_list(volumes, target_len, [0]*target_len)

        df = pd.DataFrame({
            'Date': pd.to_datetime(timestamps, unit='s'),
            'Open': opens, 'High': highs, 'Low': lows, 'Close': closes, 'Volume': volumes
        })
        df.set_index('Date', inplace=True)
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=cols, inplace=True)
        return df, None
    except Exception as e:
        return None, f"System Error: {str(e)}"

# --- QUANT ENGINE ---
def calculate_metrics(df):
    df['Mean_20'] = df['Close'].rolling(window=20).mean()
    df['Std_20'] = df['Close'].rolling(window=20).std().fillna(0)
    
    df['Z_Close'] = np.where(df['Std_20'] > 0, (df['Close'] - df['Mean_20']) / df['Std_20'], 0)
    df['Z_High'] = np.where(df['Std_20'] > 0, (df['High'] - df['Mean_20']) / df['Std_20'], 0)
    
    df['Z_Wick'] = df['Z_High'] - df['Z_Close']
    df['Wick_Pct'] = (df['High'] - df['Close']) / df['Close']
    
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    df['Vol_Median'] = df['Volume'].rolling(20).median().fillna(0)
    df['Vol_Ratio'] = np.where(df['Vol_Median'] > 0, df['Volume'] / df['Vol_Median'], 0)

    df['Z_Rank'] = df['Z_Close'].rolling(252).apply(lambda x: percentileofscore(x, x.iloc[-1]), raw=False).fillna(50)
    
    return df

# --- NEW: DETAILED CONTEXT GENERATOR ---
def get_trend_regime(price, sma20, sma50, sma200):
    """
    Returns a Tuple: (Header Status, Detailed Context Description)
    """
    if pd.isna(sma50) or pd.isna(sma200): 
        return "INSUFFICIENT DATA", "Calculating..."
    
    # 1. MACRO BULLISH (Above 200)
    if price > sma200:
        # A. Strong Uptrend (Above 50)
        if price > sma50:
            if price > sma20:
                return "STRONG UPTREND", "Price > 20MA (High Momentum)"
            else:
                return "UPTREND (Pullback)", "Price between 20MA & 50MA"
        # B. Correction (Below 50)
        else:
            return "CORRECTION", "Price broken below 50MA"
            
    # 2. MACRO BEARISH (Below 200)
    else:
        # A. Recovery Attempt (Above 50)
        if price > sma50:
            return "RECOVERY ATTEMPT", "Price reclaimed 50MA (Below 200MA)"
        # B. Downtrend
        else:
            return "DOWNTREND", "Price below 50MA & 200MA"

# --- SIGNAL ENGINE ---
def get_signal(z, rank, vol_ratio, z_high, z_wick, wick_pct, open_price, close_price):
    if pd.isna(z): return "DATA ERROR", "neut", "none", "Error"
    safe_rank = 50 if pd.isna(rank) else rank

    is_red_candle = close_price < open_price
    rejection_threshold = 0.8 if is_red_candle else 1.2
    significant_size = wick_pct > 0.005 

    # Priority 0: Panic Override
    if z < -3.0: return "FLASH CRASH (Extreme)", "hero-bull", "oversold", "Z-Score < -3.0"

    # Priority 1: Rejection
    if significant_size and (z_wick > rejection_threshold) and (vol_ratio >= 0.5):
        return "PROFIT TAKING (Wick)", "hero-bear", "rejection", f"Rejection Wick: {z_wick:.2f}σ"

    # Priority 2: Extremes
    if z_high > 3.0: return "CLIMAX TOP", "hero-bear", "rejection", "Price Extended > 3.0σ"
    if z < -2.0 and safe_rank < 5: return "EXTREME OVERSOLD", "hero-bull", "oversold", "Rank < 5%"
    if z > 2.0 and vol_ratio > 1.5: return "BREAKOUT DETECTED", "hero-bull", "breakout", "Vol > 1.5x"
    
    # Priority 3: Trend
    if 1.0 <= z <= 2.0: return "POSITIVE TREND", "hero-bull", "trend", "Z-Score > 1.0"
    if z > 2.0: return "EXTENDED (Caution)", "hero-neut", "extended", "Z-Score > 2.0"
    if z < -2.0: return "NEGATIVE INERTIA", "hero-bear", "downside", "Z-Score < -2.0"
    
    return "NO SIGNAL", "hero-neut", "none", "Market Noise"

# --- MAIN UI ---
def main():
    with st.sidebar:
        st.header("checklist")
        st.checkbox("Macro Trend (200d)?")
        st.checkbox("Sector?")
        st.checkbox("Stop Loss?")
        st.divider()
        st.caption("v24.3 Tech Context")

    c_title, c_input = st.columns([1, 2])
    with c_title:
        st.title("🛡️ Quant v24.3")
    with c_input:
        ticker_input = st.text_input("", placeholder="Ticker...", label_visibility="collapsed").strip().upper()
        if ticker_input: st.session_state.analyzed_ticker = ticker_input

    target = st.session_state.analyzed_ticker
    if target:
        with st.spinner(f"Scanning {target}..."):
            df, err = fetch_data(target)
            if err:
                st.error(f"🛑 {err}")
            else:
                if len(df) < 200: st.warning("Need 200d data.")
                st.session_state.data = calculate_metrics(df)

    if st.session_state.data is not None:
        df = st.session_state.data
        cur = df.iloc[-1]
        
        # --- NEW CONTEXT CALL ---
        regime_title, regime_context = get_trend_regime(
            cur['Close'], cur['Mean_20'], cur['SMA_50'], cur['SMA_200']
        )
        
        sig_txt, sig_css, sig_id, sig_reason = get_signal(
            cur['Z_Close'], cur['Z_Rank'], cur['Vol_Ratio'], 
            cur['Z_High'], cur['Z_Wick'], cur['Wick_Pct'],
            cur['Open'], cur['Close']
        )
        
        # Override Logic for "No Signal"
        if sig_id == "none":
            if "CORRECTION" in regime_title:
                sig_css = "hero-yellow"
                sig_txt = f"WATCHLIST: {regime_title}"
                sig_reason = regime_context # Use the precise MA context
            elif "UPTREND" in regime_title:
                sig_css = "hero-bull"
                sig_txt = f"HOLD: {regime_title}"
                sig_reason = regime_context
            elif "DOWNTREND" in regime_title:
                sig_css = "hero-bear"
                sig_txt = f"AVOID: {regime_title}"
                sig_reason = regime_context
            else:
                sig_reason = regime_context

        # HERO CARD
        st.markdown(f"""
        <div class="hero-card {sig_css}">
            <div class="hero-title">{sig_txt}</div>
            <div class="hero-subtitle">{sig_reason}</div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Price", f"${cur['Close']:.2f}")
        c2.metric("Trend (20d)", f"${cur['Mean_20']:.2f}")
        c3.metric("Trend (50d)", f"${cur['SMA_50']:.2f}")
        c4.metric("Z-Score", f"{cur['Z_Close']:.2f}σ")
        c5.metric("Rank", f"{cur['Z_Rank']:.0f}%")
        c6.metric("Reach", f"${cur['High']:.2f}", delta=f"Wick: {cur['Z_Wick']:.2f}σ", delta_color="inverse")
        c7.metric("Vol Ratio", f"{cur['Vol_Ratio']:.1f}x")
        
        st.markdown("---")

        # Chart
        st.subheader("Price Behavior Distribution (Last 200 Days)")
        valid_z = df['Z_Close'].tail(200).dropna()
        if len(valid_z) > 0:
            p05, p95 = valid_z.quantile(0.05), valid_z.quantile(0.95)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=valid_z, nbinsx=40, histnorm='probability density', marker_color='#444', opacity=0.6))
            x_range = np.linspace(-4, 4, 100)
            fig.add_trace(go.Scatter(x=x_range, y=t.pdf(x_range, df=5), mode='lines', line=dict(color='#FF4B4B', width=2)))
            fig.add_vline(x=p05, line_width=1, line_color="#888", line_dash="dash")
            fig.add_vline(x=p95, line_width=1, line_color="#888", line_dash="dash")
            fig.add_vline(x=cur['Z_Close'], line_width=3, line_color="#0066FF")
            fig.add_annotation(x=cur['Z_Close'], y=0.35, text="TODAY", font=dict(color="#0066FF", weight="bold"))
            high_text_y = 0.55 if abs(cur['Z_High'] - cur['Z_Close']) < 0.5 else 0.25
            fig.add_vline(x=cur['Z_High'], line_width=1, line_color="#FF3333", line_dash="dot")
            fig.add_annotation(x=cur['Z_High'], y=high_text_y, text="HIGH", font=dict(color="#FF3333"))
            fig.update_layout(template="plotly_white", height=300, margin=dict(t=0,b=0,l=0,r=0), xaxis_title="Z-Score", yaxis_title=None, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
