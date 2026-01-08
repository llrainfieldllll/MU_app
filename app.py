import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import percentileofscore, t, norm
from curl_cffi import requests as crequests

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Quant Scanner v18.0", page_icon="🚦")

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
    div[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 700 !important; }
    
    /* TREND BADGES */
    .trend-pill { padding: 4px 12px; border-radius: 16px; font-size: 14px; font-weight: bold; color: white; display: inline-block; margin-right: 8px; }
    .pill-green { background-color: #00cc99; }
    .pill-yellow { background-color: #ffcc00; color: #333; }
    .pill-red { background-color: #ff3333; }
</style>
""", unsafe_allow_html=True)

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
        
        closes = quote.get('close')
        if not timestamps or not closes: return None, "Empty dataset"

        volumes = quote.get('volume')
        if not volumes or len(volumes) != len(closes):
            volumes = [0] * len(closes)

        df = pd.DataFrame({
            'Date': pd.to_datetime(timestamps, unit='s'),
            'Close': closes,
            'Volume': volumes
        })
        
        df.set_index('Date', inplace=True)
        df.dropna(subset=['Close'], inplace=True)
        return df, None
    except Exception as e:
        return None, f"System Error: {str(e)}"

# --- QUANT ENGINE (Added MA Logic) ---
def calculate_metrics(df):
    # 1. Short Term (20-Day) for Z-Score
    df['Mean_20'] = df['Close'].rolling(window=20).mean()
    df['Std_20'] = df['Close'].rolling(window=20).std()
    df['Z_Score'] = np.where(df['Std_20'] > 0, (df['Close'] - df['Mean_20']) / df['Std_20'], 0)
    
    # 2. Long Term Trend (50 & 200 Day MA) -- NEW FEATURE
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # 3. Volume Ratio
    df['Vol_Median'] = df['Volume'].rolling(20).median()
    df['Vol_Ratio'] = np.where(df['Vol_Median'] > 0, df['Volume'] / df['Vol_Median'], 0)

    # 4. Empirical Rank
    df['Z_Rank'] = df['Z_Score'].rolling(252).apply(
        lambda x: percentileofscore(x, x.iloc[-1]), raw=False
    )
    
    # 5. Student-T Odds
    df['T_Prob'] = t.sf(np.abs(df['Z_Score']), df=5) * 2
    
    return df

def get_trend_regime(price, sma50, sma200):
    # Traffic Light Logic
    if pd.isna(sma50) or pd.isna(sma200): return "INSUFFICIENT DATA", "pill-yellow"
    
    if price > sma200 and price > sma50:
        return "🟢 STRONG UPTREND (Buy Dips)", "pill-green"
    elif price > sma200 and price < sma50:
        return "🟡 CORRECTION IN UPTREND", "pill-yellow"
    elif price < sma200 and price > sma50:
        return "🟡 RECOVERY ATTEMPT", "pill-yellow"
    else: # Price < 200 and Price < 50
        return "🔴 STRONG DOWNTREND (Short Rallies)", "pill-red"

def get_signal(z, rank, vol):
    if pd.isna(z): return "DATA ERROR", "neut", "none"
    safe_rank = 50 if pd.isna(rank) else rank

    if z < -2.0 and safe_rank < 5: return "PRIME OVERSOLD", "bull", "oversold"
    if z > 2.0 and vol > 1.5: return "BREAKOUT CONFIRMED", "bull", "breakout"
    if z > 2.0 and safe_rank > 95: return "REVERSAL RISK", "bear", "extreme"
    if 1.0 <= z <= 2.0: return "RIDE TREND", "bull", "trend"
    if z > 2.0: return "CAUTION (Extended)", "neut", "extended"
    if z < -2.0: return "DOWNSIDE INERTIA", "bear", "downside"
    return "NEUTRAL", "neut", "none"

# --- MAIN UI ---
def main():
    st.title("🚦 Quant Scanner v18.0 (Trend Aware)")
    
    col_input, col_rest = st.columns([1, 4])
    with col_input:
        ticker = st.text_input("Ticker", "MU").upper()
        run = st.button("Run Analysis", type="primary")

    if run:
        with st.spinner(f"Scanning Trend & Momentum for {ticker}..."):
            df, err = fetch_data(ticker)
            
            if err:
                st.error(f"🛑 {err}")
            elif len(df) < 200:
                st.warning(f"⚠️ Warning: Found {len(df)} days. Need 200 days for accurate Trend Analysis.")
                # We continue anyway, but SMA_200 will be NaN
                df = calculate_metrics(df)
            else:
                df = calculate_metrics(df)
            
            cur = df.iloc[-1]
            
            # 1. Calculate Trend Regime
            regime_txt, regime_css = get_trend_regime(cur['Close'], cur['SMA_50'], cur['SMA_200'])
            
            # 2. Calculate Momentum Signal
            rank_val = cur['Z_Rank']
            rank_display = "N/A" if pd.isna(rank_val) else f"{rank_val:.1f}%"
            sig_txt, sig_col, sig_id = get_signal(cur['Z_Score'], cur['Z_Rank'], cur['Vol_Ratio'])
            
            # --- TOP HEADER: MACRO TREND ---
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <span style="color: #666; font-weight: bold; margin-right: 10px;">MACRO TREND (200D):</span>
                <span class="trend-pill {regime_css}">{regime_txt}</span>
            </div>
            """, unsafe_allow_html=True)

            # --- MOMENTUM METRICS ---
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Price", f"${cur['Close']:.2f}")
            c2.metric("Trend (50d)", f"${cur['SMA_50']:.2f}", delta=f"{cur['Close']-cur['SMA_50']:.2f} vs 50d")
            c3.metric("Z-Score", f"{cur['Z_Score']:.2f}σ")
            c4.metric("Rank (Real)", rank_display)
            c5.metric("Vol Ratio", f"{cur['Vol_Ratio']:.1f}x")
            
            st.divider()

            # Signal Banner
            if sig_col == "bull": st.success(f"**SHORT-TERM MOMENTUM:** {sig_txt}")
            elif sig_col == "bear": st.error(f"**SHORT-TERM MOMENTUM:** {sig_txt}")
            else: st.warning(f"**SHORT-TERM MOMENTUM:** {sig_txt}")

            # Matrix & Chart
            c_matrix, c_chart = st.columns([2, 3])
            
            with c_matrix:
                st.subheader("Momentum Matrix")
                matrix_rows = [
                    {"id": "breakout", "cond": "Breakout", "z": "> 2.0", "rank": "Any", "vol": "> 1.5x", "out": "🚀 BUY BREAKOUT"},
                    {"id": "extreme", "cond": "Extension", "z": "> 2.0", "rank": "> 95%", "vol": "< 1.5x", "out": "🛑 SELL / TRIM"},
                    {"id": "oversold", "cond": "Prime Oversold", "z": "< -2.0", "rank": "< 5%", "vol": "Any", "out": "⭐ BUY DIP"},
                    {"id": "trend", "cond": "Trend", "z": "1.0 to 2.0", "rank": "Any", "vol": "Any", "out": "🌊 HOLD / ADD"},
                    {"id": "noise", "cond": "Noise", "z": "-1.0 to 1.0", "rank": "20-80%", "vol": "Any", "out": "😴 WAIT"},
                ]
                
                html = '<table class="matrix-table"><thead><tr><th>Condition</th><th>Z-Score</th><th>Rarity</th><th>Vol</th><th>Verdict</th></tr></thead><tbody>'
                for row in matrix_rows:
                    is_active = (row['id'] == sig_id)
                    css = "row-bull" if is_active and sig_col=="bull" else "row-bear" if is_active and sig_col=="bear" else "row-neut" if is_active else "row-plain"
                    icon = "✅ " if is_active else ""
                    html += f'<tr class="{css}"><td>{row["cond"]}</td><td>{row["z"]}</td><td>{row["rank"]}</td><td>{row["vol"]}</td><td>{icon}{row["out"]}</td></tr>'
                html += '</tbody></table>'
                st.markdown(html, unsafe_allow_html=True)

            with c_chart:
                st.subheader("Market Probability Model")
                valid_z = df['Z_Score'].tail(252).dropna()
                if len(valid_z) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=valid_z, nbinsx=40, histnorm='probability density', marker_color='#444', opacity=0.6, name='Real History'))
                    
                    x_range = np.linspace(-4, 4, 100)
                    y_t = t.pdf(x_range, df=5) 
                    fig.add_trace(go.Scatter(x=x_range, y=y_t, mode='lines', line=dict(color='#FF4B4B', width=2), name='Fat Tail Model'))
                    
                    fig.add_vline(x=cur['Z_Score'], line_width=3, line_color="#0066FF")
                    fig.add_annotation(x=cur['Z_Score'], y=0.35, text="NOW", font=dict(color="#0066FF", size=14, weight="bold"))
                    
                    fig.update_layout(template="plotly_white", height=350, margin=dict(t=30, b=20, l=20, r=20), xaxis_title="Z-Score Deviation", yaxis_title="Probability Density", legend=dict(orientation="h", y=1.02))
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
