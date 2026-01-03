import streamlit as st
# --- CRITICAL: PAGE CONFIG MUST BE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Quant Scanner: Reference Matrix", layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
from tenacity import retry, stop_after_attempt, wait_fixed
import socket

# --- SAFETY: PREVENT HANGS ---
# Set a global timeout so the app doesn't freeze if Yahoo is blocking
socket.setdefaulttimeout(5)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .matrix-table { width: 100%; border-collapse: collapse; font-size: 14px; margin-bottom: 20px; }
    .matrix-table th { background-color: #262730; color: white; padding: 10px; text-align: left; border-bottom: 2px solid #444; }
    .matrix-table td { padding: 10px; border-bottom: 1px solid #ddd; color: #333; }
    
    /* Highlight Classes */
    .highlight-blue { background-color: #e3f2fd; border-left: 5px solid #2196f3; font-weight: bold; }
    .highlight-green { background-color: #e8f5e9; border-left: 5px solid #4caf50; font-weight: bold; }
    .highlight-orange { background-color: #fff3e0; border-left: 5px solid #ff9800; font-weight: bold; }
    .highlight-red { background-color: #ffebee; border-left: 5px solid #f44336; font-weight: bold; }
    
    /* Faded Rows */
    .faded { color: #999 !important; opacity: 0.7; }
</style>
""", unsafe_allow_html=True)

# --- MATRIX ENGINE ---
def render_reference_matrix(z_score, vol_ratio, rsi):
    z_abs = abs(z_score)
    direction = "UP" if z_score > 0 else "DOWN"
    
    # 1. Determine Active Key
    active_key = "normal"
    if z_abs >= 1.0 and z_abs < 2.0:
        active_key = "trending"
    elif z_abs >= 2.0:
        if vol_ratio > 1.2: active_key = "breakout"
        elif vol_ratio < 0.8: active_key = "exhaustion"
        else: active_key = "outlier"
    
    # 2. Dynamic Text Logic
    breakout_label = "🟠 BREAKOUT (Up)" if direction == "UP" else "🟠 WATERFALL (Crash)"
    exhaustion_label = "🔴 TOP REVERSAL" if direction == "UP" else "🟢 BOTTOM BOUNCE"
    
    # 3. Define Rows
    rows = [
        {"key": "normal", "cond": "Normal Noise", "z": "0.0 - 1.0 σ", "vol": "Any", "rsi": "30 - 70", "verdict": "🔵 WAIT / NEUTRAL"},
        {"key": "trending", "cond": "Trending", "z": "1.0 - 2.0 σ", "vol": "Normal", "rsi": "50 - 70", "verdict": "🟢 FOLLOW TREND"},
        {"key": "breakout", "cond": "High Momentum", "z": "> 2.0 σ", "vol": "> 1.2x (High)", "rsi": "Extreme", "verdict": breakout_label},
        {"key": "exhaustion", "cond": "Exhaustion", "z": "> 2.0 σ", "vol": "< 0.8x (Low)", "rsi": "Divergence", "verdict": exhaustion_label},
        {"key": "outlier", "cond": "Statistical Outlier", "z": "> 2.0 σ", "vol": "Normal", "rsi": "Extreme", "verdict": "⚠️ ANOMALY (Caution)"}
    ]

    # 4. Build HTML
    html = '<table class="matrix-table">'
    html += '<tr><th>Market Condition</th><th>Z-Score Range</th><th>Volume</th><th>RSI</th><th>Verdict</th></tr>'
    
    for row in rows:
        if row["key"] == active_key:
            # Assign color theme
            if "breakout" in active_key: theme = "highlight-orange"
            elif "exhaustion" in active_key: theme = "highlight-red" if direction == "UP" else "highlight-green"
            elif "trending" in active_key: theme = "highlight-green"
            elif "outlier" in active_key: theme = "highlight-orange"
            else: theme = "highlight-blue"
            
            html += f'<tr class="{theme}"><td>👉 {row["cond"]}</td><td>{row["z"]}</td><td>{row["vol"]}</td><td>{row["rsi"]}</td><td>{row["verdict"]}</td></tr>'
        else:
            html += f'<tr class="faded"><td>{row["cond"]}</td><td>{row["z"]}</td><td>{
