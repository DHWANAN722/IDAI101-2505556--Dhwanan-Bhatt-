import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Crypto Volatility Visualizer")

st.title("🚀 Crypto Volatility Visualizer")
st.markdown("### Simulating and Exploring Market Swings")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

pattern = st.sidebar.selectbox(
    "Select Pattern",
    ["Real Data", "Simulated Wave", "Random Shock"]
)

amplitude = st.sidebar.slider("Amplitude", 1, 50, 10)
frequency = st.sidebar.slider("Frequency", 1, 20, 5)
drift = st.sidebar.slider("Drift", -10, 10, 1)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("crypto.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.dropna()
    df.rename(columns={"Close": "Price"}, inplace=True)
    return df

df = load_data()

# ---------------- SIMULATION ----------------
if pattern == "Simulated Wave":
    t = np.linspace(0, 10, len(df))
    df["Price"] = amplitude * np.sin(frequency * t) + drift * t + 100

elif pattern == "Random Shock":
    noise = np.random.normal(0, amplitude, len(df))
    df["Price"] = 100 + np.cumsum(noise) + drift

# ---------------- METRICS ----------------
volatility = np.std(df["Price"])
avg_price = np.mean(df["Price"])

col1, col2 = st.columns(2)
col1.metric("📊 Volatility Index", round(volatility, 2))
col2.metric("💰 Average Price", round(avg_price, 2))

# ---------------- PRICE GRAPH ----------------
fig_price = px.line(df, x="Timestamp", y="Price",
                    title="Price Over Time",
                    template="plotly_dark")
st.plotly_chart(fig_price, use_container_width=True)

# ---------------- HIGH LOW ----------------
fig_hl = go.Figure()
fig_hl.add_trace(go.Scatter(x=df["Timestamp"], y=df["High"],
                            mode='lines', name='High'))
fig_hl.add_trace(go.Scatter(x=df["Timestamp"], y=df["Low"],
                            mode='lines', name='Low'))
fig_hl.update_layout(title="High vs Low Comparison",
                     template="plotly_dark")
st.plotly_chart(fig_hl, use_container_width=True)

# ---------------- VOLUME ----------------
fig_vol = px.bar(df, x="Timestamp", y="Volume",
                 title="Volume Analysis",
                 template="plotly_dark")
st.plotly_chart(fig_vol, use_container_width=True)

# ---------------- ROLLING VOLATILITY ----------------
df["Rolling Volatility"] = df["Price"].rolling(window=10).std()

fig_roll = px.line(df, x="Timestamp", y="Rolling Volatility",
                   title="Rolling Volatility (10-period)",
                   template="plotly_dark")
st.plotly_chart(fig_roll, use_container_width=True)
