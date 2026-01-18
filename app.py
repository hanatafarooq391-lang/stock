import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- Page Config ---
st.set_page_config(page_title="StockPulse AI Terminal", layout="wide", page_icon="🚀")

st.title("🚀 StockPulse: Hybrid AI Advisor")
st.markdown("##### Ensemble Voting | Deep Learning Forecast | Technical Advice")

# --- Sidebar ---
st.sidebar.header("🕹️ Control Panel")
ticker = st.sidebar.text_input("Enter Ticker (e.g. SYS.KA, AAPL, BTC-USD)", value="SYS.KA").upper()
days = st.sidebar.slider("Historical Lookback", 200, 1000, 500)
sensitivity = st.sidebar.slider("Anomaly Sensitivity", 0.01, 0.10, 0.03)

# --- 1. LSTM Engine (Anomaly + Forecast) ---
def run_advanced_lstm(df_input, contam):
    data = df_input[['Close']].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    win = 10 
    for i in range(win, len(scaled_data)):
        X.append(scaled_data[i-win:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(win, 1)),
        Dropout(0.1),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
    
    preds = model.predict(X)
    mse = np.abs(scaled_data[win:].flatten() - preds.flatten())
    thresh = np.percentile(mse, 100 * (1 - contam))
    anoms = np.ones(len(df_input))
    anoms[win:] = np.where(mse > thresh, -1, 1)
    
    last_batch = scaled_data[-win:].reshape(1, win, 1)
    future_preds = []
    curr = last_batch
    for _ in range(5):
        nxt = model.predict(curr)[0]
        future_preds.append(nxt)
        curr = np.append(curr[:, 1:, :], [[nxt]], axis=1)
        
    future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    return anoms, future_prices.flatten()

# --- 2. Data & News Engine ---
@st.cache_data(show_spinner="Fetching Market Intelligence...")
def get_market_intelligence(symbol, days_val):
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(period=f"{days_val}d")
        if df.empty: return None, None
        
        # RSI Calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/loss)))
        
        # News with Multi-Level Key Checking
        news = []
        try:
            raw_news = tk.news
            for n in raw_news[:5]:
                # Deep extraction logic for new API format
                title = n.get('title') or n.get('headline')
                if not title and 'content' in n:
                    title = n['content'].get('title') or n['content'].get('headline')
                
                link = n.get('link') or (n.get('content').get('canonicalUrl')['url'] if 'content' in n else "#")
                publisher = n.get('publisher') or (n.get('content').get('provider')['displayName'] if 'content' in n else "Market News")
                
                if title: # Sirf tab add karein agar title mil jaye
                    news.append({'title': title, 'link': link, 'publisher': publisher})
        except:
            news = []
        
        df = df.copy()
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Close'].rolling(21).std()
        return df.dropna(), news
    except: return None, None

df, news_feed = get_market_intelligence(ticker, days)

if df is not None:
    # Analysis
    scaler_ml = StandardScaler()
    scaled_ml = scaler_ml.fit_transform(df[['Close', 'Returns', 'Volatility']])
    df['IF_Anom'] = IsolationForest(contamination=sensitivity).fit_predict(scaled_ml)
    df['SVM_Anom'] = OneClassSVM(nu=sensitivity).fit_predict(scaled_ml)
    
    with st.spinner("Analyzing Sequences with LSTM..."):
        lstm_anoms, forecast = run_advanced_lstm(df, sensitivity)
        df['LSTM_Anom'] = lstm_anoms

    # --- TOP: ADVISOR ---
    st.subheader("🤖 AI Investment Advisor")
    last_row = df.iloc[-1]
    anom_count = [last_row['IF_Anom'], last_row['SVM_Anom'], last_row['LSTM_Anom']].count(-1)
    rsi_val = last_row['RSI']
    
    if anom_count >= 2:
        advice, color, reason = "🚫 AVOID / DON'T BUY", "#e74c3c", "Unstable Price Patterns detected by multiple AI models."
    elif rsi_val < 35:
        advice, color, reason = "✅ STRONG BUY", "#27ae60", "Stock is oversold (Low RSI) and patterns are stable."
    elif rsi_val > 70:
        advice, color, reason = "⚠️ SELL / HIGH RISK", "#f39c12", "Stock is overbought. High chance of price drop."
    else:
        advice, color, reason = "⚖️ NEUTRAL (HOLD)", "#3498db", "No clear signal. Market is in normal range."

    st.markdown(f"""<div style="padding:15px; border-radius:10px; border: 2px solid {color}; background-color: rgba(0,0,0,0.02);">
        <h2 style="color:{color}; margin:0;">{advice}</h2>
        <p><b>Reason:</b> {reason} | <b>RSI:</b> {rsi_val:.2f}</p>
        </div>""", unsafe_allow_html=True)

    # --- FORECAST ---
    st.write("###")
    st.subheader("🚀 5-Day Forecast")
    f_cols = st.columns(5)
    f_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 6)]
    for i in range(5):
        diff = ((forecast[i] - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
        f_cols[i].metric(f_dates[i].strftime('%d %b'), f"{forecast[i]:.2f}", f"{diff:.2f}%")

    # --- MAIN CONTENT ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📊 Price Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Actual', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=f_dates, y=forecast, name='Forecast', line=dict(color='#3498db', dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📰 Market Headlines")
        if news_feed:
            for item in news_feed:
                st.markdown(f"**[{item['title']}]({item['link']})**")
                st.caption(f"Source: {item['publisher']}")
                st.divider()
        else:
            st.info("No news headlines found. Tickers like 'AAPL' or 'TSLA' usually show more news.")

    with st.expander("📊 Technical Analysis Log"):
        st.dataframe(df[['Close', 'RSI', 'IF_Anom', 'LSTM_Anom']].sort_values(by='Date', ascending=False))

else:
    st.error("Data Fetch Error.")