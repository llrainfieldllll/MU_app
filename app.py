import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import percentileofscore, t, norm
from curl_cffi import requests as crequests

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Quant Scanner v17.0", page_icon="📐")

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
</style>
""", unsafe_allow_html=True)

# --- ROBUST DATA ENGINE ---
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

# --- QUANT ENGINE (Student-T Enhanced) ---
def calculate_metrics(df):
    # 1. Standard Z-Score (SMA Basis)
    df['Mean_20'] = df['Close'].rolling(window=20).mean()
    df['Std_20'] = df['Close'].rolling(window=20).std()
    df['Z_Score'] = np.where(df['Std_20'] > 0, (df['Close'] - df['Mean_20']) / df['Std_20'], 0)
    
    # 2. Volume Ratio
    df['Vol_Median'] = df['Volume'].rolling(20).median()
    df['Vol_Ratio'] = np.where(df['Vol_Median'] > 0, df['Volume'] / df['Vol_Median'], 0)

    # 3. Empirical Rank (Real History)
    df['Z_Rank'] = df['Z_Score'].rolling(252).apply(
        lambda x: percentileofscore(x, x.iloc[-1]), raw=False
    )
    
    # 4. Student-T Probability (Theoretical Fat Tail)
    # Degrees of Freedom (df) = 5 is standard for daily stock returns
    # We calculate the "Survival Function" (sf) which is 1 - CDF to get the tail probability
    # Multiplied by 2 for two-tailed test (Extreme up OR Extreme down)
    df['T_Prob'] = t.sf(np.abs(df['Z_Score']), df=5) * 2
    
    return df

def get_signal(z, rank, vol):
    if pd.isna(z): return "DATA ERROR", "neut", "none"
    safe_rank = 50 if pd.isna(rank) else rank

    # Logic: Breakout > Trend > Reversal
    if z < -2.0 and safe_rank < 5: return "PRIME OVERSOLD", "bull", "oversold"
    if z > 2.0 and vol > 1.5: return "BREAKOUT CONFIRMED", "bull", "breakout"
    if z > 2.0 and safe_rank > 95: return "REVERSAL RISK", "bear", "extreme"
    if 1.0 <= z <= 2.0: return "RIDE TREND", "bull", "trend"
    if z > 2.0: return "CAUTION (Extended)", "neut", "extended"
    if z < -2.0: return "DOWNSIDE INERTIA", "bear", "downside"
    
    return "NEUTRAL", "neut", "none"

# --- MAIN UI ---
def main():
    st.title("📐 Quant Scanner v17.0 (Fat Tail Viz)")
    st.caption("Includes Student-T Distribution Analysis (df=5)")
    
    col_input, col_rest = st.columns([1, 4])
    with col_input:
        ticker = st.text_input("Ticker", "MU").upper()
        run = st.button("Run Analysis", type="primary")

    if run:
        with st.spinner(f"Modeling Fat Tails for {ticker}..."):
            df, err = fetch_data(ticker)
            
            if err:
                st.error(f"🛑 {err}")
            elif len(df) < 20:
                st.error(f"⚠️ Insufficient data (Need 20+ days)")
            else:
                df = calculate_metrics(df)
                cur = df.iloc[-1]
                
                # Metrics Formatting
                rank_val = cur['Z_Rank']
                rank_display = "N/A" if pd.isna(rank_val) else f"{rank_val:.1f}%"
                
                # Probability Formatting (Student-T)
                # Convert 0.05 prob to "1 in 20"
                prob_val = cur['T_Prob']
                if prob_val > 0:
                    days_odds = int(1 / prob_val)
                    prob_str = f"1 in {days_odds} Days"
                else:
                    prob_str = "Extreme"

                sig_txt, sig_col, sig_id = get_signal(cur['Z_Score'], cur['Z_Rank'], cur['Vol_Ratio'])
                
                # --- METRICS ---
                # Added "Theo. Odds" to show Student-T output
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Price", f"${cur['Close']:.2f}")
                c2.metric("Z-Score", f"{cur['Z_Score']:.2f}σ")
                c3.metric("Rank (Real)", rank_display)
                c4.metric("Odds (Theory)", prob_str, help="Theoretical frequency based on Student-T (Fat Tail) distribution")
                c5.metric("Vol Ratio", f"{cur['Vol_Ratio']:.1f}x")
                
                st.divider()

                # Signal Banner
                if sig_col == "bull": st.success(f"**STATISTICAL BIAS:** {sig_txt}")
                elif sig_col == "bear": st.error(f"**STATISTICAL BIAS:** {sig_txt}")
                else: st.warning(f"**STATISTICAL BIAS:** {sig_txt}")

                # Matrix & Chart
                c_matrix, c_chart = st.columns([2, 3]) # Gave chart more space for details
                
                with c_matrix:
                    st.subheader("Decision Logic")
                    matrix_rows = [
                        {"id": "breakout", "cond": "Breakout", "z": "> 2.0", "rank": "Any", "vol": "> 1.5x", "out": "🚀 BUY BREAKOUT"},
                        {"id": "extreme", "cond": "Extension", "z": "> 2.0", "rank": "> 95%", "vol": "< 1.5x", "out": "🛑 SELL / TRIM"},
                        {"id": "oversold", "cond": "Prime Oversold", "z": "< -2.0", "rank": "< 5%", "vol": "Any", "out": "⭐ BUY DIP"},
                        {"id": "trend", "cond": "Trend", "z": "1.0 to 2.0", "rank": "Any", "vol": "Any", "out": "🌊 HOLD / ADD"},
                        {"id": "noise", "cond": "Noise", "z": "-1.0 to 1.0", "rank": "20-80%", "vol": "Any", "out": "😴 WAIT"},
                    ]
                    
                    html = '<table class="matrix-table">'
                    html += '<thead><tr><th>Condition</th><th>Z-Score</th><th>Rarity</th><th>Volume</th><th>Verdict</th></tr></thead><tbody>'
                    
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
                        html += f'<tr class="{css}"><td>{row["cond"]}</td><td>{row["z"]}</td><td>{row["rank"]}</td><td>{row["vol"]}</td><td>{icon}{row["out"]}</td></tr>'
                    html += '</tbody></table>'
                    st.markdown(html, unsafe_allow_html=True)

                with c_chart:
                    st.subheader("Fat Tails: Theory vs Reality")
                    valid_z = df['Z_Score'].tail(252).dropna()
                    
                    # 1. Histogram (Reality)
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=valid_z, nbinsx=40, histnorm='probability density',
                        marker_color='#444', opacity=0.6, name='Real History'
                    ))
                    
                    # 2. Student-T Curve (Fat Tail Theory)
                    x_range = np.linspace(-4, 4, 100)
                    # df=5 is the standard finance setting for "Fat Tails"
                    y_t = t.pdf(x_range, df=5) 
                    fig.add_trace(go.Scatter(
                        x=x_range, y=y_t, mode='lines', 
                        line=dict(color='#FF4B4B', width=2), name='Fat Tail Model (T)'
                    ))

                    # 3. Normal Curve (Thin Tail Theory - For Comparison)
                    y_norm = norm.pdf(x_range)
                    fig.add_trace(go.Scatter(
                        x=x_range, y=y_norm, mode='lines', 
                        line=dict(color='#00CC96', width=2, dash='dot'), name='Normal Model (Z)'
                    ))

                    # 4. Current Marker
                    fig.add_vline(x=cur['Z_Score'], line_width=3, line_color="#0066FF")
                    fig.add_annotation(
                        x=cur['Z_Score'], y=0.35, text="NOW", 
                        font=dict(color="#0066FF", size=14, weight="bold")
                    )
                    
                    fig.update_layout(
                        template="plotly_white", height=350, 
                        xaxis_title="Z-Score Deviation", yaxis_title="Probability Density",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(t=30, b=20, l=20, r=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
