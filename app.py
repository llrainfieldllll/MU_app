import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from tenacity import retry, stop_after_attempt, wait_fixed
import logging

# --- CONFIGURATION ---
st.set_page_config(page_title="Quant-Audit: Pro", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ROBUST DATA ENGINE ---
@st.cache_data(ttl=900, show_spinner=False) # Reduced TTL to 15min for fresher market data
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_market_data(ticker: str) -> pd.DataFrame:
    try:
        # PURE YFINANCE (Internal Spoofing)
        # Note: We fetch '1y' to ensure smooth RSI/SMA calculations even if there are holidays
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="6mo", interval="1d")
        
        # Validation 1: No Data
        if data.empty:
            return pd.DataFrame()

        # Validation 2: Timezone Cleanup
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # Validation 3: Length Check
        # Need 34 days min (20 for SMA + 14 for RSI)
        if len(data) < 34: 
            return pd.DataFrame()

        return data
    except Exception as e:
        logger.error(f"Data Fetch Error: {e}")
        raise e

# --- MATH ENGINE ---
def calculate_metrics(df: pd.DataFrame):
    prices = df['Close']
    
    # 1. Determine "Current" Context
    # We use the very last candle available (could be live today or yesterday close)
    current_price = float(prices.iloc[-1])
    
    # 2. Slice for SMA (Last 20 points)
    # logic: The SMA includes the current price in its calculation window
    analysis_slice = prices.tail(20)
    mu = analysis_slice.mean()    
    sigma = analysis_slice.std()
    
    # Safety: Zero Volatility
    if sigma == 0:
        z_score = 0
    else:
        z_score = (current_price - mu) / sigma
        
    # 3. Probability (One-Tailed Test)
    # Answers: "What is the % chance of being ABOVE this Z?" (if Z>0)
    # Answers: "What is the % chance of being BELOW this Z?" (if Z<0)
    if z_score > 0:
        p_value = 1 - norm.cdf(z_score) 
    else:
        p_value = norm.cdf(z_score)
        
    # 4. Price Magnet (Reversion)
    price_adjustment = mu - current_price

    # 5. RSI (Wilder's Smoothing is standard, but Simple is safer for pure arrays)
    # We use Simple RSI here for stability
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    
    last_gain = gain.iloc[-1]
    last_loss = loss.iloc[-1]

    if last_loss == 0 and last_gain == 0:
        rsi = 50.0 # Flatline
    elif last_loss == 0:
        rsi = 100.0 # Vertical Up
    else:
        rs = last_gain / last_loss
        rsi = 100 - (100 / (1 + rs))

    return {
        "price": current_price,
        "mu": mu,
        "sigma": sigma,
        "z_score": z_score,
        "p_value": p_value,
        "price_adjustment": price_adjustment,
        "rsi": rsi
    }

# --- UI RENDERER ---
def main():
    st.title("🛡️ Quant Auditor: Statistical Deep Dive")
    st.markdown("#### Market Mean Reversion Scanner")
    
    with st.sidebar:
        st.header("Search")
        ticker = st.text_input("Ticker Symbol", "MU", help="e.g. NVDA, SPY, AAPL").upper()
        run_btn = st.button("Run Audit", type="primary")
        st.info("💡 **Tip:** Z-Scores > 2.0 often signal a statistical extreme.")

    if run_btn:
        with st.spinner(f"Auditing volatility profile for {ticker}..."):
            try:
                # 1. DATA
                data = fetch_market_data(ticker)
                
                if data.empty:
                    st.error(f"❌ **Connection Failed.** Could not retrieve data for '{ticker}'.\n\nPossible reasons:\n1. Invalid Ticker\n2. Yahoo Finance is blocking Cloud IP (Try refreshing later).")
                    st.stop()

                # 2. MATH
                m = calculate_metrics(data)

                # 3. KPI DISPLAY
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Live Price", f"${m['price']:.2f}")
                
                col2.metric("20-Day SMA", f"${m['mu']:.2f}", 
                           delta=f"{m['price_adjustment']:.2f}", delta_color="inverse")
                
                col3.metric("Z-Score (Sigma)", f"{m['z_score']:.2f}σ", 
                           delta_color="off")
                
                col4.metric("Rarity (P-Value)", f"{m['p_value']*100:.2f}%",
                           help="The theoretical probability of the price being this far from the mean.")

                st.divider()

                # 4. PLOTLY VISUALIZATION (Pixel-Perfect Match)
                x = np.linspace(-4, 4, 1000)
                y = norm.pdf(x, 0, 1)
                
                fig = go.Figure()

                # Black Bell Curve
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode='lines', 
                    name='Normal Dist', 
                    line=dict(color='black', width=2.5)
                ))
                
                # Dynamic Coloring based on Z
                z = m['z_score']
                is_extreme = abs(z) >= 2.0
                line_color = "#FF4B4B" if is_extreme else "#222222" # Red if extreme, Black/Dark if normal
                
                # Red Dashed Line
                fig.add_vline(x=z, line_width=2, line_dash="dash", line_color="red")

                # Shaded Tail Area
                if z > 0:
                    fill_x = x[x >= z]
                    fill_y = y[x >= z]
                    direction_text = "Above"
                else:
                    fill_x = x[x <= z]
                    fill_y = y[x <= z]
                    direction_text = "Below"
                
                fig.add_trace(go.Scatter(
                    x=fill_x, y=fill_y, 
                    fill='tozeroy', 
                    fillcolor='rgba(255, 0, 0, 0.4)', 
                    line=dict(width=0),
                    name=f"Tail Risk"
                ))

                # Annotation (Current MU)
                fig.add_annotation(
                    x=z, y=max(y)*0.5, # Place at half height
                    text=f"Current: {z:.2f}σ", 
                    showarrow=True,
                    arrowhead=2,
                    ax=40 if z > 0 else -40, # Dynamic arrow direction
                    ay=-40,
                    font=dict(color="red", size=14, family="Arial Black")
                )

                # Layout: Clean White Background
                fig.update_layout(
                    title=dict(text=f"<b>{ticker} Distribution at ${m['price']:.2f}</b>", x=0.5),
                    xaxis_title="Standard Deviations (σ)",
                    yaxis_title="Probability Density",
                    template="plotly_white",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=500,
                    margin=dict(l=20, r=20, t=80, b=20),
                    xaxis=dict(range=[-4.5, 4.5], showgrid=True, gridcolor='#F0F0F0'),
                    yaxis=dict(showgrid=True, gridcolor='#F0F0F0')
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # 5. TEXT NARRATIVE
                st.subheader('Analysis of the "Tail" Risk')
                
                prob_pct = m['p_value'] * 100
                
                # Narrative Logic
                if abs(z) >= 2.0:
                    status = "EXTREME OUTLIER"
                    context = "Buying Climax / Parabolic" if z > 0 else "Selling Climax / Crash"
                elif abs(z) >= 1.0:
                    status = "ELEVATED"
                    context = "Trending Strong"
                else:
                    status = "NORMAL"
                    context = "Noise / Chop"

                st.info(f"""
                **1. {prob_pct:.2f}% Probability:** Statistically, there is only a **{prob_pct:.2f}% chance** of the stock being this far {direction_text.upper()} its 20-Day SMA.
                
                **2. Status: {status}** The stock is currently at **{z:.2f}σ**. ({context}).
                
                **3. Price Magnet:** The 20-Day SMA (**${m['mu']:.2f}**) is the statistical gravity. To revert to the mean, the price must adjust by **${m['price_adjustment']:.2f}**.
                """)

            except Exception as e:
                st.error(f"System Error: {str(e)}")

if __name__ == "__main__":
    main()
