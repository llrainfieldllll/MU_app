import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import percentileofscore, t
from curl_cffi import requests as crequests

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Quant Scanner v23.0 (Pro Edition)", page_icon="🛡️")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .matrix-table { width: 100%; border-collapse: collapse; font-family: 'Arial', sans-serif; margin-top: 20px; }
    .matrix-table th { background-color: #000; color: #fff; padding: 12px; text-align: left; font-size: 14px; }
    .matrix-table td { padding: 12px; border-bottom: 1px solid #ddd; color: #333; font-size: 14px; }
    
    /* Signal Rows */
    .row-bull { background-color: #e6fffa; border-left: 5px solid #00cc99; font-weight: bold; }
    .row-bear { background-color: #fff5f5; border-left: 5px solid #ff3333; font-weight: bold; }
    .row-rejection { background-color: #fff0f0; border-left: 5px solid #cc0000; font-weight: bold; }
    .row-neut { background-color: #f9f9f9; border-left: 5px solid #999; font-weight: bold; }
    .row-plain { background-color: #fff; color: #666; }
    
    /* Metric Styling */
    div[data-testid="stMetricValue"] { font-size: 22px !important; font-weight: 700 !important; }
    div[data-testid="stMetricLabel"] { font-size: 13px !important; color: #555; }
    
    /* Trend Pills */
    .trend-pill { padding: 4px 12px; border-radius: 16px; font-size: 14px; font-weight: bold; color: white; display: inline-block; margin-right: 8px; }
    .pill-green { background-color: #00cc99; }
    .pill-yellow { background-color: #ffcc00; color: #333; }
    .pill-red { background-color: #ff3333; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'data' not in st.session_state: st.session_state.data = None
if 'analyzed_ticker' not in st.session_state: st.session_state.analyzed_ticker = "MU" 

# --- DATA ENGINE (Stability Patch) ---
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
        
        # --- ROBUST PADDING (Prevents Array Length Crashes) ---
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
        # Ensure numeric types
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=cols, inplace=True)
        
        return df, None
    except Exception as e:
        return None, f"System Error: {str(e)}"

# --- QUANT ENGINE (Logic Patch) ---
def calculate_metrics(df):
    df['Mean_20'] = df['Close'].rolling(window=20).mean()
    df['Std_20'] = df['Close'].rolling(window=20).std()
    
    # 1. Main Z-Score
    df['Z_Close'] = np.where(df['Std_20'] > 0, (df['Close'] - df['Mean_20']) / df['Std_20'], 0)
    
    # 2. Shadow Z-Score (High)
    df['Z_High'] = np.where(df['Std_20'] > 0, (df['High'] - df['Mean_20']) / df['Std_20'], 0)
    
    # 3. Z_Wick (Rejection Energy)
    df['Z_Wick'] = df['Z_High'] - df['Z_Close']
    
    # 4. Wick Percentage (Volatility Filter)
    df['Wick_Pct'] = (df['High'] - df['Close']) / df['Close']
    
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    df['Vol_Median'] = df['Volume'].rolling(20).median()
    df['Vol_Ratio'] = np.where(df['Vol_Median'] > 0, df['Volume'] / df['Vol_Median'], 0)

    # Rank Calculation
    df['Z_Rank'] = df['Z_Close'].rolling(252).apply(lambda x: percentileofscore(x, x.iloc[-1]), raw=False)
    
    return df

def get_trend_regime(price, sma50, sma200):
    if pd.isna(sma50) or pd.isna(sma200): return "INSUFFICIENT DATA", "pill-yellow"
    if price > sma200 and price > sma50: return "🟢 STRONG UPTREND", "pill-green"
    elif price > sma200 and price < sma50: return "🟡 CORRECTION", "pill-yellow"
    elif price < sma200 and price > sma50: return "🟡 RECOVERY", "pill-yellow"
    else: return "🔴 STRONG DOWNTREND", "pill-red"

# --- SIGNAL ENGINE (Soundness Patch) ---
def get_signal(z, rank, vol_ratio, z_high, z_wick, wick_pct, open_price, close_price):
    if pd.isna(z): return "DATA ERROR", "neut", "none"
    safe_rank = 50 if pd.isna(rank) else rank

    is_red_candle = close_price < open_price
    # Logic: Green candles get more room (1.2), Red candles are strict (0.8)
    rejection_threshold = 0.8 if is_red_candle else 1.2
    significant_size = wick_pct > 0.005 

    # --- FINAL LOGIC: MUST HAVE VOLUME ---
    if significant_size and (z_wick > rejection_threshold) and (vol_ratio >= 0.5):
        return "PROFIT TAKING (Wick)", "bear", "rejection"

    if z_high > 3.0: return "CLIMAX TOP", "bear", "rejection"
    if z < -2.0 and safe_rank < 5: return "EXTREME OVERSOLD", "bull", "oversold"
    if z > 2.0 and vol_ratio > 1.5: return "BREAKOUT DETECTED", "bull", "breakout"
    if z > 2.0 and safe_rank > 95: return "STATISTICAL EXTREME", "bear", "extreme"
    if 1.0 <= z <= 2.0: return "POSITIVE TREND", "bull", "trend"
    if z > 2.0: return "EXTENDED (Caution)", "neut", "extended"
    if z < -2.0: return "NEGATIVE INERTIA", "bear", "downside"
    
    return "NO SIGNAL", "neut", "none"

# --- MAIN UI ---
def main():
    with st.sidebar:
        st.header("🧠 Pre-Trade Checklist")
        st.write("Stop. Breathe. Check these before executing.")
        st.checkbox("Is the Macro Trend (200d) in my favor?")
        st.checkbox("Is the Sector/Market also moving this way?")
        st.checkbox("Do I have a predefined Stop Loss?")
        st.checkbox("Am I chasing a green candle?")
        st.divider()
        st.caption("v23.0 Pro Edition")

    st.title("🛡️ Quant Scanner v23.0")
    
    col_input, col_rest = st.columns([1, 4])
    with col_input:
        # Input Hygiene
        ticker_input = st.text_input("Ticker", placeholder="Enter Ticker (e.g. NVDA)").strip().upper()
        run = st.button("Run Analysis", type="primary")

    target_ticker = ticker_input if ticker_input else st.session_state.analyzed_ticker

    if run and target_ticker:
        st.session_state.analyzed_ticker = target_ticker
        with st.spinner(f"Scanning {target_ticker}..."):
            df, err = fetch_data(target_ticker)
            if err:
                st.error(f"🛑 {err}")
                st.session_state.data = None
            elif len(df) < 200:
                st.warning(f"⚠️ Warning: Found {len(df)} days. Need 200 days for Macro Trend.")
                st.session_state.data = calculate_metrics(df)
            else:
                st.session_state.data = calculate_metrics(df)

    if st.session_state.data is not None:
        st.caption(f"Showing Analysis for: **{st.session_state.analyzed_ticker}**")
        
        df = st.session_state.data
        cur = df.iloc[-1]
        
        regime_txt, regime_css = get_trend_regime(cur['Close'], cur['SMA_50'], cur['SMA_200'])
        
        sig_txt, sig_col, sig_id = get_signal(
            cur['Z_Close'], cur['Z_Rank'], cur['Vol_Ratio'], 
            cur['Z_High'], cur['Z_Wick'], cur['Wick_Pct'],
            cur['Open'], cur['Close']
        )
        
        rank_display = f"{cur['Z_Rank']:.1f}%" if not pd.isna(cur['Z_Rank']) else "N/A"
        
        # --- VISUAL LOGIC: DIM MACRO ON DANGER ---
        if "rejection" in sig_col or "bear" in sig_col:
            macro_opacity = "0.4"
            macro_msg = f"⚠️ MACRO IS {regime_txt} (BUT IGNORE IT)"
        else:
            macro_opacity = "1.0"
            macro_msg = regime_txt

        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px; opacity: {macro_opacity};">
            <span style="color: #666; font-weight: bold; margin-right: 10px;">MACRO CONTEXT:</span>
            <span class="trend-pill {regime_css}">{macro_msg}</span>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Price", f"${cur['Close']:.2f}")
        c2.metric("Trend (20d)", f"${cur['Mean_20']:.2f}")
        c3.metric("Trend (50d)", f"${cur['SMA_50']:.2f}")
        c4.metric("Z-Score (20d)", f"{cur['Z_Close']:.2f}σ", help="Current live price vs 20-day average.")
        c5.metric("Rank (Real)", rank_display, help="Percentile Rank of today's Z-Score.")
        c6.metric("Intraday Reach", f"${cur['High']:.2f}", 
                  delta=f"Wick: {cur['Z_Wick']:.2f}σ", delta_color="inverse", 
                  help=f"Max Z: {cur['Z_High']:.2f}σ | Wick Size: {cur['Z_Wick']:.2f}σ")
        c7.metric("Vol Ratio", f"{cur['Vol_Ratio']:.1f}x")
        
        st.divider()

        if sig_col == "bull": st.success(f"**STATISTICAL BIAS:** {sig_txt}")
        elif sig_col == "bear": st.error(f"**STATISTICAL BIAS:** {sig_txt}")
        else: st.warning(f"**STATISTICAL BIAS:** {sig_txt}")

        c_matrix, c_chart = st.columns([2, 3])
        
        with c_matrix:
            st.subheader("Signal Matrix")
            matrix_rows = [
                {"id": "breakout", "cond": "Breakout", "z": "> 2.0", "rank": "Any", "vol": "> 1.5x", "out": "🚀 BREAKOUT"},
                {"id": "extreme", "cond": "Extension", "z": "> 2.0", "rank": "> 95%", "vol": "< 1.5x", "out": "⚠️ EXTENDED"},
                {"id": "rejection", "cond": "Profit Taking", "z": "Wick > 0.8σ", "rank": "Any", "vol": "> 0.5x", "out": "🔻 REJECTION"},
                {"id": "rejection", "cond": "Climax Top", "z": "High > 3.0", "rank": "Any", "vol": "Any", "out": "🔻 CLIMAX TOP"},
                {"id": "oversold", "cond": "Prime Oversold", "z": "< -2.0", "rank": "< 5%", "vol": "Any", "out": "⭐ OVERSOLD"},
                {"id": "trend", "cond": "Trend", "z": "1.0 to 2.0", "rank": "Any", "vol": "Any", "out": "🌊 UPTREND"},
            ]
            
            html = '<table class="matrix-table"><thead><tr><th>Condition</th><th>Z-Score</th><th>Rarity</th><th>Vol</th><th>Signal</th></tr></thead><tbody>'
            for row in matrix_rows:
                is_active = False
                if row['id'] == sig_id:
                     if row['cond'] == "Profit Taking" and sig_txt == "PROFIT TAKING (Wick)": is_active = True
                     elif row['cond'] == "Climax Top" and sig_txt == "CLIMAX TOP": is_active = True
                     elif row['id'] != "rejection": is_active = True
                
                css = "row-bull" if is_active and sig_col=="bull" else "row-bear" if is_active and sig_col=="bear" else "row-rejection" if row['id']=="rejection" and is_active else "row-neut" if is_active else "row-plain"
                icon = "✅ " if is_active else ""
                html += f'<tr class="{css}"><td>{row["cond"]}</td><td>{row["z"]}</td><td>{row["rank"]}</td><td>{row["vol"]}</td><td>{icon}{row["out"]}</td></tr>'
            html += '</tbody></table>'
            st.markdown(html, unsafe_allow_html=True)

        with c_chart:
            st.subheader("Price Behavior Distribution (Last 200 Days)")
            valid_z = df['Z_Close'].tail(200).dropna()
            
            if len(valid_z) > 0:
                # --- DATA DRIVEN PERCENTILE LINES ---
                p05 = valid_z.quantile(0.05)
                p95 = valid_z.quantile(0.95)
                
                fig = go.Figure()
                
                # 1. Histogram (Real History)
                fig.add_trace(go.Histogram(
                    x=valid_z, nbinsx=40, histnorm='probability density',
                    marker_color='#444', opacity=0.6, name='Real History'
                ))
                
                # 2. Theoretical Curve
                x_range = np.linspace(-4, 4, 100)
                fig.add_trace(go.Scatter(x=x_range, y=t.pdf(x_range, df=5), 
                            mode='lines', line=dict(color='#FF4B4B', width=2), name='Theoretical Curve'))
                
                # 3. Percentile Lines (The Truth)
                fig.add_vline(x=p05, line_width=1, line_color="#888", line_dash="dash")
                fig.add_annotation(x=p05, y=0.42, text="5%", font=dict(color="#888", size=10))
                
                fig.add_vline(x=p95, line_width=1, line_color="#888", line_dash="dash")
                fig.add_annotation(x=p95, y=0.42, text="95%", font=dict(color="#888", size=10))

                # 4. Markers
                fig.add_vline(x=cur['Z_Close'], line_width=3, line_color="#0066FF")
                fig.add_annotation(x=cur['Z_Close'], y=0.35, text="TODAY", 
                                 font=dict(color="#0066FF", size=14, weight="bold"))
                
                # Collision Detection
                high_text_y = 0.55 if abs(cur['Z_High'] - cur['Z_Close']) < 0.5 else 0.25
                fig.add_vline(x=cur['Z_High'], line_width=1, line_color="#FF3333", line_dash="dot")
                fig.add_annotation(x=cur['Z_High'], y=high_text_y, text="HIGH", 
                                 font=dict(color="#FF3333", size=12))

                fig.update_layout(
                    template="plotly_white", height=300, margin=dict(t=10, b=20, l=20, r=20),
                    xaxis_title="Deviation from Normal (Z-Score)", 
                    yaxis_title="Frequency", 
                    legend=dict(orientation="h", y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("""<div style='text-align: center; color: #888; font-size: 12px;'>DISCLAIMER: Not Financial Advice.</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
