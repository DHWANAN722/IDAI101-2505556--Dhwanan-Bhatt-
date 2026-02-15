import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide", page_title="Crypto Volatility Visualizer")

# ---------------- FULL DARK + BLUE THEME ----------------
st.markdown("""
<style>

/* Main background */
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #00BFFF !important;
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #00BFFF;
}

/* Paragraph text */
p, label {
    color: #d1e8ff;
}

/* Slider track */
.stSlider > div > div > div > div {
    background-color: #00BFFF;
}

/* Dropdown styling */
div[data-baseweb="select"] > div {
    background-color: #1f2937 !important;
    color: #00BFFF !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🚀 Crypto Volatility Visualizer")
st.markdown("### Real-Time Volatility Simulation & Analysis")
st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

pattern = st.sidebar.selectbox(
    "Select Pattern",
    ["Real Data", "Simulated Wave", "Random Shock"]
)

amplitude = st.sidebar.slider("Amplitude (Volatility Size)", 1, 50, 10)
frequency = st.sidebar.slider("Frequency (Swing Speed)", 1, 20, 5)
drift = st.sidebar.slider("Drift (Long-Term Trend)", -10, 10, 1)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("crypto.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.dropna()
    return df

df = load_data()

# ---------------- SIMULATION LOGIC ----------------
if pattern == "Simulated Wave":
    t = np.linspace(0, 10, len(df))
    df["Close"] = amplitude * np.sin(frequency * t) + drift * t + 100

elif pattern == "Random Shock":
    noise = np.random.normal(0, amplitude, len(df))
    df["Close"] = 100 + np.cumsum(noise) + drift

# ---------------- METRICS ----------------
volatility = np.std(df["Close"])
avg_price = np.mean(df["Close"])
max_price = np.max(df["Close"])
min_price = np.min(df["Close"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("📊 Volatility Index", round(volatility, 2))
m2.metric("💰 Average Price", round(avg_price, 2))
m3.metric("📈 Max Price", round(max_price, 2))
m4.metric("📉 Min Price", round(min_price, 2))

st.divider()

# ---------------- PRICE GRAPH ----------------
fig_price = px.line(df, x="Timestamp", y="Close")
fig_price.update_traces(line=dict(color="#1f77ff", width=2))
fig_price.update_layout(
    title="Price Over Time",
    plot_bgcolor="white",
    paper_bgcolor="#0e1117",
    font=dict(color="#00BFFF")
)

# ---------------- ROLLING VOLATILITY ----------------
df["Rolling Volatility"] = df["Close"].rolling(window=10).std()

fig_roll = px.line(df, x="Timestamp", y="Rolling Volatility")
fig_roll.update_traces(line=dict(color="#00BFFF", width=2))
fig_roll.update_layout(
    title="Rolling Volatility (10-Period)",
    plot_bgcolor="white",
    paper_bgcolor="#0e1117",
    font=dict(color="#00BFFF")
)

# ---------------- HIGH vs LOW ----------------
fig_hl = go.Figure()
fig_hl.add_trace(go.Scatter(
    x=df["Timestamp"], y=df["High"],
    mode="lines", name="High",
    line=dict(color="#4da6ff")
))
fig_hl.add_trace(go.Scatter(
    x=df["Timestamp"], y=df["Low"],
    mode="lines", name="Low",
    line=dict(color="#99ccff")
))
fig_hl.update_layout(
    title="High vs Low Comparison",
    plot_bgcolor="white",
    paper_bgcolor="#0e1117",
    font=dict(color="#00BFFF")
)

# ---------------- VOLUME ----------------
fig_vol = px.bar(df, x="Timestamp", y="Volume")
fig_vol.update_traces(marker_color="#1f77ff")
fig_vol.update_layout(
    title="Volume Analysis",
    plot_bgcolor="white",
    paper_bgcolor="#0e1117",
    font=dict(color="#00BFFF")
)

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)
col1.plotly_chart(fig_price, use_container_width=True)
col2.plotly_chart(fig_roll, use_container_width=True)

st.divider()

col3, col4 = st.columns(2)
col3.plotly_chart(fig_hl, use_container_width=True)
col4.plotly_chart(fig_vol, use_container_width=True)
