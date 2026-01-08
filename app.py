import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import percentileofscore
from curl_cffi import requests as crequests

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Quant Scanner v13.0", page_icon="🧠")

# --- CUSTOM CSS (ADHD Optimized) ---
st.markdown("""
<style>
    /* FOCUS MODE TABLE */
    .matrix-table { width: 100%; border-collapse: collapse; font-family: 'Arial', sans-serif; margin-top: 10px; }
    .matrix-table th { background-color: #f0f2f6; color: #444; padding: 10px; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .matrix-table td { padding: 12px; border-bottom: 1px solid #eee; color: #333; transition: all 0.3s ease; }
    
    /* Active Rows (Spotlight) */
    .row-bull { background-color: #e6fffa; border-left: 6px solid #00cc99; font-weight: bold; color: #004d3b; transform: scale(1.01); box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .row-bear { background-color: #fff5f5; border-left: 6px solid #ff3333; font-weight: bold; color: #661a1a; transform: scale(1.01); box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .row-neut { background-color: #f8f9fa; border-left: 6px solid #adb5bd; font-weight: bold; color: #495057; }
    
    /* Inactive Rows (Dimmed) */
    .row-inactive { opacity: 0.35; filter: grayscale(100%); }
    
    /* METRICS */
    div[data-testid="stMetricValue"] { font-size: 36px !important; font-weight: 800 !important; font-family: 'Roboto', sans-serif; }
    div[data-testid="stMetricLabel"] { font-size: 14px !important; color: #666; font-weight: 500; text-transform: uppercase; }
    
    /* BADGES */
    .signal-badge { padding: 10px 20px; border-radius: 8px; font-weight: bold; font-size: 20px; text-align: center; margin-bottom: 20px; color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .badge-bull { background: linear-gradient(90deg, #00b09b, #96c93d); }
    .badge-bear { background: linear-gradient(90deg, #ff416c, #ff4b2b); }
    .badge-neut { background: linear-gradient(90deg, #606c88, #3f4c6b); }
</style>
""", unsafe_allow_html=True)

# --- ROBUST DATA ENGINE (Safe JSON Parsing) ---
@st.cache_data(ttl=300)
def fetch_data(ticker):
    try:
        session = crequests.Session()
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"
        r = session.get(url, headers=headers, impersonate="chrome110")
        
        if r.status_code != 200: 
            return None, f"API Error: {r.status_code}"

        data = r.json()
        
        # SENIOR FIX: Granular Key Check to prevent crashes
        if 'chart' not in data or 'result' not in data['chart']:
            return None, "Invalid API Response Structure"
            
        if not data['chart']['result']:
            return None, f"No data found for ticker '{ticker}'"

        result = data['chart']['result'][0]
        
        if 'timestamp' not in result or 'indicators' not in result:
             return None, "Empty dataset received"

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
        return None, f"System Error: {str(e)}"

# --- QUANT ENGINE (Math Safety Added) ---
def calculate_metrics(df):
    # 1. EMA Mean (20-day)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # 2. Volatility (Std)
    df['Std_20'] = df['Close'].rolling(20).std()
    
    # SENIOR FIX: Handle Division by Zero for Z-Score
    # If Std_20 is 0, Z-Score becomes 0 (flatline) instead of inf
    df['Z_Score'] = np.where(df['Std_20'] > 0, (df['Close'] - df['EMA_20']) / df['Std_20'], 0)
    
    # 3. Volume Ratio
    df['Vol_Median'] = df['Volume'].rolling(20).median()
    
    # SENIOR FIX: Handle Division by Zero for Volume
    # If Median Volume is 0, set Ratio to 0 to avoid massive spikes
    df['Vol_Ratio'] = np.where(df['Vol_Median'] > 0, df['Volume'] / df['Vol_Median'], 0)

    # 4. Percentile Rank
    df['Z_Rank'] = df['Z_Score'].rolling(252).apply(
        lambda x: percentileofscore(x, x.iloc[-1]), raw=False
    )
    
    return df

def get_signal(z, rank, vol):
    # SENIOR FIX: Explicit NaN handling
    if pd.isna(z): return "DATA ERROR", "neut", "none"
    
    # If Rank is NaN (New stock < 1 year), treat as "Neutral" 50% for logic safety
    # But we will display "N/A" in the UI
    safe_rank = 50 if pd.isna(rank) else rank

    if z > 2.0:
        if safe_rank > 95: return "🛑 REVERSAL RISK", "bear", "extreme"
        if vol > 1.5: return "🚀 BREAKOUT CONFIRMED", "bull", "breakout"
        return "⚠️ CAUTION (Extended)", "neut", "extended"
    elif 1.0 <= z <= 2.0:
        return "🌊 RIDE TREND", "bull", "trend"
    elif -1.0 < z < 1.0:
        return "💤 WAIT / NOISE", "neut", "noise"
    elif z < -2.0:
        if safe_rank < 5: return "⭐ PRIME OVERSOLD", "bull", "oversold"
        return "🩸 DOWNSIDE INERTIA", "bear", "downside"
    return "⚪ NEUTRAL", "neut", "none"

# --- MAIN UI ---
def main():
    st.title("🧠 Quant Scanner v13.0 (Bulletproof)")
    
    col_input, col_rest = st.columns([1, 5])
    with col_input:
        ticker = st.text_input("Ticker", "MU").upper()
        run = st.button("Run Scan", type="primary")

    if run:
        with st.spinner(f"Focusing on {ticker}..."):
            df, err = fetch_data(ticker)
            
            if err:
                st.error(f"🛑 {err}")
            elif len(df) < 20:
                st.error(f"⚠️ Insufficient data (Found {len(df)} days, need 20+)")
            else:
                df = calculate_metrics(df)
                cur = df.iloc[-1]
                
                # Logic Variables
                rank_val = cur['Z_Rank']
                rank_display = "N/A" if pd.isna(rank_val) else f"{rank_val:.1f}%"
                sig_txt, sig_col, sig_id = get_signal(cur['Z_Score'], cur['Z_Rank'], cur['Vol_Ratio'])
                
                # --- 1. SIGNAL BADGE ---
                badge_css = f"badge-{sig_col}"
                st.markdown(f'<div class="signal-badge {badge_css}">{sig_txt}</div>', unsafe_allow_html=True)
                
                # --- 2. METRICS ---
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Price", f"${cur['Close']:.2f}")
                m2.metric("Regime", "BULL" if cur['Close'] > df['EMA_20'].iloc[-1] else "BEAR")
                m3.metric("Z-Score", f"{cur['Z_Score']:.2f}σ")
                m4.metric("Rank (1-Year)", rank_display)
                m5.metric("Vol Ratio", f"{cur['Vol_Ratio']:.1f}x")
                
                st.write("") 

                # --- 3. FOCUS MATRIX & CHART ---
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
                                <th>Action</th>
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
                            icon = "👈"
                        else:
                            css = "row-inactive"
                            icon = ""
                            
                        html += f"""
                            <tr class="{css}">
                                <td>{row['cond']}</td>
                                <td>{row['z']}</td>
                                <td>{row['rank']}</td>
                                <td>{row['vol']}</td>
                                <td>{row['out']} {icon}</td>
                            </tr>
                        """
                    
                    html += "</tbody></table>"
                    st.markdown(html, unsafe_allow_html=True)

                with c_chart:
                    st.subheader("Is it Normal?")
                    valid_z = df['Z_Score'].tail(252).dropna()
                    
                    if len(valid_z) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=valid_z, nbinsx=40, 
                            marker_color='#e0e0e0', opacity=0.8, name='History'
                        ))
                        
                        fig.add_vline(x=cur['Z_Score'], line_width=4, line_color="#000")
                        fig.add_annotation(
                            x=cur['Z_Score'], y=10, 
                            text="NOW", 
                            font=dict(color="#000", size=16, weight="black")
                        )
                        
                        fig.update_layout(
                            template="plotly_white", 
                            height=280, 
                            xaxis_title="Z-Score Deviation",
                            showlegend=False,
                            margin=dict(t=20, b=20, l=20, r=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Insufficient data for histogram.")

if __name__ == "__main__":
    main()
