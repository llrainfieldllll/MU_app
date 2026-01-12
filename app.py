import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import percentileofscore, t
from curl_cffi import requests as crequests

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Quant Scanner v24.6 (Purple Edition)", page_icon="🛡️")

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
    
    /* --- COLOR PALETTE --- */
    
    /* 1. BULLISH (Green): Strong Uptrend, Breakouts */
    .hero-bull { background: linear-gradient(135deg, #00b09b, #96c93d); }
    
    /* 2. BEARISH (Red): Rejections, Crashes, Downtrends */
    .hero-bear { background: linear-gradient(135deg, #ff5f6d, #ffc371); }
    
    /* 3. WATCHLIST (Yellow): Corrections, Dips */
    .hero-yellow { background: linear-gradient(135deg, #f7971e, #ffd200); color: #222 !important; }
    
    /* 4. EXTENDED (Purple): Running Hot, Caution (NEW) */
    .hero-purple { background: linear-gradient(135deg, #667eea, #764ba2); }
    
    /* 5. NEUTRAL (Gray): Noise */
    .hero-neut { background: linear-gradient(135deg, #8e9eab, #eef2f3); color: #333 !important; }

    /* Metric & Matrix Styling */
    div[data-testid="stMetricValue"] { font-size: 20px !important; font-weight: 700 !important; font-family: 'Roboto Mono', monospace; }
    div[data-testid="stMetricLabel"] { font-size: 12px !important; color: #666; }
    .stExpander { border: none !important; box-shadow: none !important; background-color: transparent !important; }
    .matrix-table { width: 100%; border-collapse: collapse; font-family: 'Arial', sans-serif; margin-top: 10px; }
    .matrix-table th { background-color: #333; color: #fff; padding: 10px; font-size: 12px; text-align: left; }
    .matrix-table td { padding: 8px; border-bottom: 1px solid #ddd; font-size: 13px; color: #333; }
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

def get_trend_regime(price, sma20, sma50, sma200):
    if pd.isna(sma50) or pd.isna(sma200): 
        return "INSUFFICIENT DATA", "Calculating..."
    
    if price > sma200:
        if price > sma50:
            if price > sma20: return "STRONG UPTREND", "Price > 20MA (High Momentum)"
            else: return "UPTREND (Pullback)", "Price between 20MA & 50MA"
        else: return "CORRECTION", "Price broken below 50MA"
    else:
        if price > sma50: return "RECOVERY ATTEMPT", "Price reclaimed 50MA (Below 200MA)"
        else: return "DOWNTREND", "Price below 50MA & 200MA"

def get_signal(z, rank, vol_ratio, z_high, z_wick, wick_pct, open_price, close_price):
    if pd.isna(z): return "DATA ERROR", "neut", "none", "Error"
    safe_rank = 50 if pd.isna(rank) else rank

    is_red_candle = close_price < open_price
    rejection_threshold = 0.8 if is_red_candle else 1.2
    significant_size = wick_pct > 0.005 

    # 1. Panic Override
    if z < -3.0: return "FLASH CRASH (Extreme)", "hero-bull", "oversold", "Z-Score < -3.0"

    # 2. Rejection
    if significant_size and (z_wick > rejection_threshold) and (vol_ratio >= 0.5):
        return "PROFIT TAKING (Wick)", "hero-bear", "rejection", f"Rejection Wick: {z_wick:.2f}σ"

    # 3. Extremes
    if z_high > 3.0: return "CLIMAX TOP", "hero-bear", "rejection", "Price Extended > 3.0σ"
    if z < -2.0 and safe_rank < 5: return "EXTREME OVERSOLD", "hero-bull", "oversold", "Rank < 5%"
    if z > 2.0 and vol_ratio > 1.5: return "BREAKOUT DETECTED", "hero-bull", "breakout", "Vol > 1.5x"
    
    # 4. Trend & Extension
    if 1.0 <= z <= 2.0: return "POSITIVE TREND", "hero-bull", "trend", "Z-Score > 1.0"
    
    # --- UPDATED: Use PURPLE for Extended ---
    if z > 2.0: return "EXTENDED (Caution)", "hero-purple", "extended", "Z-Score > 2.0 (Hot)"
    
    if z < -2.0: return "NEGATIVE INERTIA", "hero-bear", "downside", "Z-Score < -2.0"
    
    return "NO SIGNAL", "hero-neut", "none", "Market Noise"

def style_metric(label, value, color):
    return f"""
    <div style="display: flex; flex-direction: column; justify-content: center; align-items: flex-start; padding: 0px;">
        <span style="font-size: 12px; color: #666; margin-bottom: -5px;">{label}</span>
        <span style="font-size: 24px; font-weight: 700; color: {color}; font-family: 'Roboto Mono', monospace;">{value}</span>
    </div>
    """

# --- MAIN UI ---
def main():
    with st.sidebar:
        st.header("checklist")
        st.checkbox("Macro Trend (200d)?")
        st.checkbox("Sector?")
        st.checkbox("Stop Loss?")
        st.divider()
        st.caption("v24.6 Purple Edition")

    c_title, c_input = st.columns([1, 2])
    with c_title:
        st.title("🛡️ Quant v24.6")
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
                sig_css = "hero-yellow"; sig_txt = f"WATCHLIST: {regime_title}"; sig_reason = regime_context
            elif "UPTREND" in regime_title:
                sig_css = "hero-bull"; sig_txt = f"HOLD: {regime_title}"; sig_reason = regime_context
            elif "DOWNTREND" in regime_title:
                sig_css = "hero-bear"; sig_txt = f"AVOID: {regime_title}"; sig_reason = regime_context
            else:
                sig_reason = regime_context

        # Hero Card
        st.markdown(f"""
        <div class="hero-card {sig_css}">
            <div class="hero-title">{sig_txt}</div>
            <div class="hero-subtitle">{sig_reason}</div>
        </div>
        """, unsafe_allow_html=True)

        # Color Logic for Price
        price, sma20, sma50, sma200 = cur['Close'], cur['Mean_20'], cur['SMA_50'], cur['SMA_200']
        s20 = 0 if pd.isna(sma20) else sma20
        s50 = 0 if pd.isna(sma50) else sma50
        s200 = 0 if pd.isna(sma200) else sma200
        
        if price > s20: price_color = "#00CC96" 
        elif price > s50: price_color = "#FFA500"
        elif price > s200: price_color = "#FF4B4B"
        else: price_color = "#8B0000"

        # Metrics
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.markdown(style_metric("Price", f"${price:.2f}", price_color), unsafe_allow_html=True)
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

        # Hidden Matrix
        with st.expander("❓ View Signal Logic Matrix (Advanced)"):
            st.caption("How the 'Hero Signal' is calculated:")
            matrix_rows = [
                {"id": "breakout", "cond": "Breakout", "z": "> 2.0", "vol": "> 1.5x", "out": "🚀 BREAKOUT"},
                {"id": "extended", "cond": "Extension", "z": "> 2.0", "vol": "< 1.5x", "out": "⚠️ EXTENDED"},
                {"id": "rejection", "cond": "Profit Taking", "z": "Wick > 0.8σ", "vol": "> 0.5x", "out": "🔻 REJECTION"},
                {"id": "rejection", "cond": "Climax Top", "z": "High > 3.0", "vol": "Any", "out": "🔻 CLIMAX TOP"},
                {"id": "oversold", "cond": "Prime Oversold", "z": "< -2.0", "vol": "Any", "out": "⭐ OVERSOLD"},
                {"id": "trend", "cond": "Trend", "z": "1.0 to 2.0", "vol": "Any", "out": "🌊 UPTREND"},
            ]
            
            html = '<table class="matrix-table"><thead><tr><th>Condition</th><th>Z-Score</th><th>Vol</th><th>Signal</th></tr></thead><tbody>'
            for row in matrix_rows:
                is_active = False
                if row['id'] == sig_id: is_active = True
                elif row['id'] == "extended" and sig_id == "extended": is_active = True
                
                bg = "#e6fffa" if is_active else "transparent"
                fw = "bold" if is_active else "normal"
                icon = "✅ " if is_active else ""
                html += f'<tr style="background-color: {bg}; font-weight: {fw}"><td>{row["cond"]}</td><td>{row["z"]}</td><td>{row["vol"]}</td><td>{icon}{row["out"]}</td></tr>'
            html += '</tbody></table>'
            st.markdown(html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
