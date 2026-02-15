import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CryptoVol Analytics Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for that WOW aesthetic
st.markdown("""
<style>
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        box-shadow: 5px 0 15px rgba(0,0,0,0.3);
    }
    
    /* Headers with glow effect */
    h1, h2, h3 {
        color: #ffffff !important;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        font-family: 'Helvetica Neue', sans-serif;
        letter-spacing: 1px;
    }
    
    /* Metrics container */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #00ff88 !important;
        text-shadow: 0 0 10px rgba(0,255,136,0.5);
    }
    
    /* Cards and containers */
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff 0%, #0099ff 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.6);
    }
    
    /* Sidebar text */
    .sidebar .sidebar-content {
        color: white;
    }
    
    /* Success/Info boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: white;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d4ff 0%, #0099ff 100%);
    }
    
    /* Slider */
    .stSlider [data-baseweb="slider"] {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        color: white !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    """Load and prepare cryptocurrency data"""
    df = pd.read_csv('crypto.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    df['Date'] = df['Timestamp'].dt.date
    
    # Calculate additional metrics
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Range_Pct'] = (df['Price_Range'] / df['Close']) * 100
    df['Volume_MA7'] = df['Volume'].rolling(window=7).mean()
    df['Price_MA7'] = df['Close'].rolling(window=7).mean()
    df['Price_MA30'] = df['Close'].rolling(window=30).mean()
    df['Volatility_7D'] = df['Daily_Return'].rolling(window=7).std()
    df['Volatility_30D'] = df['Daily_Return'].rolling(window=30).std()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# Advanced mathematical pattern generation
def generate_wave_pattern(length, amplitude, frequency, phase, noise_level, drift):
    """Generate mathematical wave patterns with noise and drift"""
    x = np.linspace(0, 10, length)
    
    # Sine wave component
    sine_wave = amplitude * np.sin(frequency * x + phase)
    
    # Cosine wave component for complexity
    cosine_wave = (amplitude * 0.3) * np.cos(frequency * 1.5 * x + phase * 0.5)
    
    # Random noise
    noise = np.random.normal(0, noise_level, length)
    
    # Linear drift
    drift_component = drift * x
    
    # Combine all components
    pattern = sine_wave + cosine_wave + noise + drift_component
    
    return pattern

def calculate_volatility_index(data):
    """Calculate custom volatility index"""
    returns = data.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    return volatility * 100

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Main app
def main():
    # Load data
    df = load_data()
    
    # Sidebar - Advanced Controls
    with st.sidebar:
        st.markdown("# 🎛️ Control Panel")
        st.markdown("---")
        
        # Theme toggle
        theme_col1, theme_col2 = st.columns(2)
        with theme_col1:
            if st.button("🌙 Dark"):
                st.session_state.theme = 'dark'
        with theme_col2:
            if st.button("☀️ Light"):
                st.session_state.theme = 'light'
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "📍 Navigation",
            ["🏠 Dashboard Overview", 
             "📊 Advanced Analytics", 
             "🎨 Pattern Generator",
             "📈 Volatility Analyzer",
             "🔬 Deep Dive Analysis"],
            label_visibility="visible"
        )
        
        st.markdown("---")
        
        # Date range filter
        st.markdown("### 📅 Date Range")
        date_range = st.date_input(
            "Select Period",
            value=(df['Date'].min(), df['Date'].max()),
            min_value=df['Date'].min(),
            max_value=df['Date'].max()
        )
        
        st.markdown("---")
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters", expanded=False):
            price_range = st.slider(
                "Price Range ($)",
                float(df['Close'].min()),
                float(df['Close'].max()),
                (float(df['Close'].min()), float(df['Close'].max())),
                step=100.0
            )
            
            volume_filter = st.slider(
                "Minimum Volume",
                0,
                int(df['Volume'].max()),
                0,
                step=1000
            )
        
        st.markdown("---")
        
        # Pattern generator controls (for Pattern Generator page)
        if page == "🎨 Pattern Generator":
            st.markdown("### 🎚️ Pattern Controls")
            
            pattern_type = st.selectbox(
                "Pattern Type",
                ["Sine Wave", "Cosine Wave", "Combined Waves", "Random Walk", "Trending"]
            )
            
            amplitude = st.slider("Amplitude", 1.0, 50.0, 10.0, 0.5)
            frequency = st.slider("Frequency", 0.1, 5.0, 1.0, 0.1)
            phase = st.slider("Phase Shift", 0.0, 2*np.pi, 0.0, 0.1)
            noise_level = st.slider("Noise Level", 0.0, 10.0, 1.0, 0.1)
            drift = st.slider("Drift (Trend)", -5.0, 5.0, 0.0, 0.1)
        
        st.markdown("---")
        
        # Info box
        st.info("💡 **Pro Tip**: Use the advanced filters to drill down into specific market conditions!")
        
        # Stats preview
        with st.expander("📊 Quick Stats", expanded=True):
            filtered_df = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]
            st.metric("Data Points", len(filtered_df))
            st.metric("Date Range", f"{(date_range[1] - date_range[0]).days} days")
    
    # Filter data based on sidebar selections
    filtered_df = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]
    filtered_df = filtered_df[
        (filtered_df['Close'] >= price_range[0]) & 
        (filtered_df['Close'] <= price_range[1]) &
        (filtered_df['Volume'] >= volume_filter)
    ]
    
    # Page routing
    if page == "🏠 Dashboard Overview":
        show_dashboard_overview(filtered_df, df)
    elif page == "📊 Advanced Analytics":
        show_advanced_analytics(filtered_df)
    elif page == "🎨 Pattern Generator":
        show_pattern_generator(amplitude, frequency, phase, noise_level, drift, pattern_type)
    elif page == "📈 Volatility Analyzer":
        show_volatility_analyzer(filtered_df)
    elif page == "🔬 Deep Dive Analysis":
        show_deep_dive(filtered_df)

def show_dashboard_overview(filtered_df, full_df):
    """Main dashboard with overview metrics and charts"""
    
    # Header with animation effect
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, rgba(0,212,255,0.3) 0%, rgba(148,0,255,0.3) 100%); border-radius: 15px; margin-bottom: 20px;'>
            <h1 style='font-size: 48px; margin: 0;'>📈 CryptoVol Analytics Pro</h1>
            <p style='font-size: 20px; color: #ffffff; margin: 10px 0 0 0;'>Advanced Cryptocurrency Volatility Analysis Dashboard</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Row
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_price = filtered_df['Close'].iloc[-1]
        prev_price = filtered_df['Close'].iloc[0]
        price_change = ((current_price - prev_price) / prev_price) * 100
        st.metric(
            "Current Price",
            f"${current_price:,.2f}",
            f"{price_change:+.2f}%",
            delta_color="normal"
        )
    
    with col2:
        vol_index = calculate_volatility_index(filtered_df['Close'])
        st.metric(
            "Volatility Index",
            f"{vol_index:.2f}%",
            "High" if vol_index > 50 else "Moderate"
        )
    
    with col3:
        avg_volume = filtered_df['Volume'].mean()
        st.metric(
            "Avg Volume",
            f"{avg_volume:,.0f}",
            "BTC"
        )
    
    with col4:
        high_52w = full_df['High'].max()
        st.metric(
            "52W High",
            f"${high_52w:,.2f}",
            "Peak"
        )
    
    with col5:
        low_52w = full_df['Low'].min()
        st.metric(
            "52W Low",
            f"${low_52w:,.2f}",
            "Bottom"
        )
    
    st.markdown("---")
    
    # Main chart section with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Action", "📊 Volume Analysis", "🎯 Technical Indicators", "🔥 Heatmap"])
    
    with tab1:
        # Advanced candlestick chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Bitcoin Price with Bollinger Bands', 'Daily Returns'),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=filtered_df['Timestamp'],
                open=filtered_df['Open'],
                high=filtered_df['High'],
                low=filtered_df['Low'],
                close=filtered_df['Close'],
                name='OHLC',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff0055'
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=filtered_df['Timestamp'],
                y=filtered_df['BB_Upper'],
                name='BB Upper',
                line=dict(color='rgba(255, 255, 0, 0.5)', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=filtered_df['Timestamp'],
                y=filtered_df['BB_Middle'],
                name='BB Middle',
                line=dict(color='rgba(255, 255, 255, 0.7)', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=filtered_df['Timestamp'],
                y=filtered_df['BB_Lower'],
                name='BB Lower',
                line=dict(color='rgba(255, 255, 0, 0.5)', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(255, 255, 0, 0.1)'
            ),
            row=1, col=1
        )
        
        # Daily returns bar chart
        colors = ['#00ff88' if val > 0 else '#ff0055' for val in filtered_df['Daily_Return']]
        fig.add_trace(
            go.Bar(
                x=filtered_df['Timestamp'],
                y=filtered_df['Daily_Return'],
                name='Daily Return %',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=800,
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig, width='stretch')
    
    with tab2:
        # Volume analysis
        fig_vol = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Trading Volume Over Time', 'Volume vs Price Correlation'),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )
        
        # Volume bars
        fig_vol.add_trace(
            go.Bar(
                x=filtered_df['Timestamp'],
                y=filtered_df['Volume'],
                name='Volume',
                marker=dict(
                    color=filtered_df['Volume'],
                    colorscale='Viridis',
                    showscale=True
                )
            ),
            row=1, col=1
        )
        
        # 7-day moving average
        fig_vol.add_trace(
            go.Scatter(
                x=filtered_df['Timestamp'],
                y=filtered_df['Volume_MA7'],
                name='7D MA',
                line=dict(color='#ff6b6b', width=3)
            ),
            row=1, col=1
        )
        
        # Scatter: Volume vs Price Change
        fig_vol.add_trace(
            go.Scatter(
                x=filtered_df['Volume'],
                y=filtered_df['Daily_Return'],
                mode='markers',
                name='Vol vs Returns',
                marker=dict(
                    size=8,
                    color=filtered_df['Close'],
                    colorscale='Plasma',
                    showscale=True,
                    opacity=0.6
                )
            ),
            row=2, col=1
        )
        
        fig_vol.update_layout(
            template='plotly_dark',
            height=700,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        fig_vol.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.1)')
        fig_vol.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig_vol, width='stretch')
    
    with tab3:
        # Technical indicators
        col_a, col_b = st.columns(2)
        
        with col_a:
            # RSI
            fig_rsi = go.Figure()
            
            fig_rsi.add_trace(go.Scatter(
                x=filtered_df['Timestamp'],
                y=filtered_df['RSI'],
                name='RSI',
                line=dict(color='#00d4ff', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 212, 255, 0.3)'
            ))
            
            # Overbought/Oversold lines
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            
            fig_rsi.update_layout(
                title="RSI (Relative Strength Index)",
                template='plotly_dark',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig_rsi, width='stretch')
        
        with col_b:
            # Moving averages
            fig_ma = go.Figure()
            
            fig_ma.add_trace(go.Scatter(
                x=filtered_df['Timestamp'],
                y=filtered_df['Close'],
                name='Close Price',
                line=dict(color='white', width=1)
            ))
            
            fig_ma.add_trace(go.Scatter(
                x=filtered_df['Timestamp'],
                y=filtered_df['Price_MA7'],
                name='7D MA',
                line=dict(color='#00ff88', width=2)
            ))
            
            fig_ma.add_trace(go.Scatter(
                x=filtered_df['Timestamp'],
                y=filtered_df['Price_MA30'],
                name='30D MA',
                line=dict(color='#ff0055', width=2)
            ))
            
            fig_ma.update_layout(
                title="Moving Averages Comparison",
                template='plotly_dark',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_ma, width='stretch')
    
    with tab4:
        # Heatmap of correlations
        st.markdown("### 🔥 Correlation Heatmap")
        
        # Select numeric columns for correlation
        corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'Price_Range', 'Volatility_7D']
        corr_data = filtered_df[corr_cols].corr()
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_data.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig_heatmap.update_layout(
            title="Feature Correlation Matrix",
            template='plotly_dark',
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_heatmap, width='stretch')

def show_advanced_analytics(filtered_df):
    """Advanced analytics page with statistical analysis"""
    
    st.markdown("# 📊 Advanced Analytics Suite")
    st.markdown("### Deep Statistical Analysis & Market Insights")
    
    st.markdown("---")
    
    # Distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=filtered_df['Close'],
            nbinsx=50,
            name='Price Distribution',
            marker=dict(
                color='#00d4ff',
                line=dict(color='white', width=1)
            )
        ))
        
        fig_dist.update_layout(
            title="Price Distribution Analysis",
            template='plotly_dark',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title="Price ($)",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig_dist, width='stretch')
    
    with col2:
        # Returns distribution
        fig_ret_dist = go.Figure()
        
        fig_ret_dist.add_trace(go.Histogram(
            x=filtered_df['Daily_Return'].dropna(),
            nbinsx=50,
            name='Returns Distribution',
            marker=dict(
                color='#ff0055',
                line=dict(color='white', width=1)
            )
        ))
        
        fig_ret_dist.update_layout(
            title="Daily Returns Distribution",
            template='plotly_dark',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig_ret_dist, width='stretch')
    
    # Statistical metrics
    st.markdown("### 📈 Statistical Summary")
    
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.metric("Mean Price", f"${filtered_df['Close'].mean():,.2f}")
        st.metric("Median Price", f"${filtered_df['Close'].median():,.2f}")
    
    with col_b:
        st.metric("Std Deviation", f"${filtered_df['Close'].std():,.2f}")
        st.metric("Variance", f"{filtered_df['Close'].var():,.0f}")
    
    with col_c:
        st.metric("Skewness", f"{filtered_df['Daily_Return'].skew():.3f}")
        st.metric("Kurtosis", f"{filtered_df['Daily_Return'].kurtosis():.3f}")
    
    with col_d:
        st.metric("Max Drawdown", f"{filtered_df['Daily_Return'].min():.2f}%")
        st.metric("Max Gain", f"{filtered_df['Daily_Return'].max():.2f}%")
    
    st.markdown("---")
    
    # Volatility analysis over time
    st.markdown("### 📉 Volatility Evolution")
    
    fig_vol_evolution = go.Figure()
    
    fig_vol_evolution.add_trace(go.Scatter(
        x=filtered_df['Timestamp'],
        y=filtered_df['Volatility_7D'],
        name='7-Day Volatility',
        line=dict(color='#00ff88', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 136, 0.2)'
    ))
    
    fig_vol_evolution.add_trace(go.Scatter(
        x=filtered_df['Timestamp'],
        y=filtered_df['Volatility_30D'],
        name='30-Day Volatility',
        line=dict(color='#ff0055', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 85, 0.2)'
    ))
    
    fig_vol_evolution.update_layout(
        title="Rolling Volatility Analysis",
        template='plotly_dark',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_vol_evolution, width='stretch')
    
    # Price range analysis
    st.markdown("### 📏 Intraday Range Analysis")
    
    fig_range = go.Figure()
    
    fig_range.add_trace(go.Scatter(
        x=filtered_df['Timestamp'],
        y=filtered_df['Price_Range_Pct'],
        mode='lines+markers',
        name='Daily Range %',
        line=dict(color='#ffd700', width=2),
        marker=dict(size=4, color='#ffd700')
    ))
    
    fig_range.update_layout(
        title="Daily Price Range as % of Close",
        template='plotly_dark',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis_title="Range %"
    )
    
    st.plotly_chart(fig_range, width='stretch')

def show_pattern_generator(amplitude, frequency, phase, noise_level, drift, pattern_type):
    """Mathematical pattern generator page"""
    
    st.markdown("# 🎨 Mathematical Pattern Generator")
    st.markdown("### Simulate Market Behavior with Mathematical Functions")
    
    st.markdown("---")
    
    # Generate patterns
    length = 200
    x_vals = np.linspace(0, 10, length)
    
    if pattern_type == "Sine Wave":
        pattern = amplitude * np.sin(frequency * x_vals + phase) + noise_level * np.random.randn(length) + drift * x_vals
    elif pattern_type == "Cosine Wave":
        pattern = amplitude * np.cos(frequency * x_vals + phase) + noise_level * np.random.randn(length) + drift * x_vals
    elif pattern_type == "Combined Waves":
        pattern = generate_wave_pattern(length, amplitude, frequency, phase, noise_level, drift)
    elif pattern_type == "Random Walk":
        steps = np.random.randn(length) * noise_level + drift * 0.1
        pattern = np.cumsum(steps) + amplitude
    else:  # Trending
        pattern = drift * x_vals + noise_level * np.random.randn(length) + amplitude * np.sin(frequency * x_vals)
    
    # Create comparison: stable vs volatile
    stable_pattern = 2 * np.sin(0.5 * x_vals) + 0.5 * np.random.randn(length)
    volatile_pattern = 15 * np.sin(2 * x_vals + 1) + 5 * np.random.randn(length) + 0.5 * x_vals
    
    # Main pattern chart
    fig_pattern = go.Figure()
    
    fig_pattern.add_trace(go.Scatter(
        x=x_vals,
        y=pattern,
        mode='lines',
        name=f'{pattern_type}',
        line=dict(color='#00d4ff', width=3)
    ))
    
    fig_pattern.update_layout(
        title=f"Generated Pattern: {pattern_type}",
        template='plotly_dark',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_title="Time",
        yaxis_title="Value"
    )
    
    st.plotly_chart(fig_pattern, width='stretch')
    
    # Pattern statistics
    st.markdown("### 📊 Pattern Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{np.mean(pattern):.2f}")
    with col2:
        st.metric("Std Dev", f"{np.std(pattern):.2f}")
    with col3:
        st.metric("Volatility Index", f"{calculate_volatility_index(pd.Series(pattern)):.2f}%")
    with col4:
        st.metric("Range", f"{np.ptp(pattern):.2f}")
    
    st.markdown("---")
    
    # Stable vs Volatile comparison
    st.markdown("### ⚖️ Stable vs Volatile Market Simulation")
    
    fig_comparison = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Stable Market (Low Volatility)', 'Volatile Market (High Volatility)'),
        horizontal_spacing=0.1
    )
    
    fig_comparison.add_trace(
        go.Scatter(
            x=x_vals,
            y=stable_pattern,
            mode='lines',
            name='Stable',
            line=dict(color='#00ff88', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.3)'
        ),
        row=1, col=1
    )
    
    fig_comparison.add_trace(
        go.Scatter(
            x=x_vals,
            y=volatile_pattern,
            mode='lines',
            name='Volatile',
            line=dict(color='#ff0055', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 85, 0.3)'
        ),
        row=1, col=2
    )
    
    fig_comparison.update_layout(
        template='plotly_dark',
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig_comparison, width='stretch')
    
    # Comparison metrics
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 🟢 Stable Market Metrics")
        st.metric("Volatility", f"{np.std(stable_pattern):.2f}", "Low")
        st.metric("Price Range", f"{np.ptp(stable_pattern):.2f}", "Narrow")
    
    with col_b:
        st.markdown("#### 🔴 Volatile Market Metrics")
        st.metric("Volatility", f"{np.std(volatile_pattern):.2f}", "High")
        st.metric("Price Range", f"{np.ptp(volatile_pattern):.2f}", "Wide")
    
    # Mathematical explanation
    with st.expander("📚 Mathematical Background", expanded=False):
        st.markdown("""
        ### Wave Functions Used:
        
        **Sine Wave:** `y = A * sin(f * x + φ) + noise + drift`
        - A: Amplitude (swing size)
        - f: Frequency (swing speed)
        - φ: Phase shift
        
        **Combined Pattern:**
        - Primary sine wave
        - Secondary cosine wave (30% amplitude, 1.5x frequency)
        - Random Gaussian noise
        - Linear drift component
        
        **Volatility Calculation:**
        - Standard deviation of returns
        - Annualized using √252 (trading days)
        """)

def show_volatility_analyzer(filtered_df):
    """Dedicated volatility analysis page"""
    
    st.markdown("# 📈 Volatility Analyzer")
    st.markdown("### Comprehensive Volatility Metrics & Analysis")
    
    st.markdown("---")
    
    # Volatility metrics grid
    st.markdown("### 🎯 Volatility Metrics Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        daily_vol = filtered_df['Daily_Return'].std()
        st.metric("Daily Volatility", f"{daily_vol:.2f}%")
    
    with col2:
        weekly_vol = filtered_df['Volatility_7D'].mean()
        st.metric("7D Avg Volatility", f"{weekly_vol:.2f}%")
    
    with col3:
        monthly_vol = filtered_df['Volatility_30D'].mean()
        st.metric("30D Avg Volatility", f"{monthly_vol:.2f}%")
    
    with col4:
        annual_vol = calculate_volatility_index(filtered_df['Close'])
        st.metric("Annualized Vol", f"{annual_vol:.2f}%")
    
    with col5:
        avg_range = filtered_df['Price_Range_Pct'].mean()
        st.metric("Avg Daily Range", f"{avg_range:.2f}%")
    
    st.markdown("---")
    
    # Volatility regime classification
    st.markdown("### 🎨 Volatility Regime Classification")
    
    # Classify days by volatility
    filtered_df['Volatility_Regime'] = pd.cut(
        filtered_df['Volatility_7D'],
        bins=[0, 2, 5, 10, 100],
        labels=['Very Low', 'Low', 'Medium', 'High']
    )
    
    regime_counts = filtered_df['Volatility_Regime'].value_counts()
    
    fig_regime = go.Figure(data=[
        go.Pie(
            labels=regime_counts.index,
            values=regime_counts.values,
            hole=0.4,
            marker=dict(
                colors=['#00ff88', '#00d4ff', '#ffd700', '#ff0055'],
                line=dict(color='white', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=14, color='white')
        )
    ])
    
    fig_regime.update_layout(
        title="Distribution of Volatility Regimes",
        template='plotly_dark',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True
    )
    
    st.plotly_chart(fig_regime, width='stretch')
    
    # Volatility surface
    st.markdown("### 🌊 Volatility Surface")
    
    # Create volatility surface data
    window_sizes = [7, 14, 30, 60, 90]
    vol_surface = pd.DataFrame()
    
    for window in window_sizes:
        if len(filtered_df) >= window:
            vol_surface[f'{window}D'] = filtered_df['Daily_Return'].rolling(window=window).std()
    
    fig_surface = go.Figure()
    
    for col in vol_surface.columns:
        fig_surface.add_trace(go.Scatter(
            x=filtered_df['Timestamp'],
            y=vol_surface[col],
            name=col,
            mode='lines',
            line=dict(width=2)
        ))
    
    fig_surface.update_layout(
        title="Multi-Period Volatility Surface",
        template='plotly_dark',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x unified',
        yaxis_title="Volatility (%)"
    )
    
    st.plotly_chart(fig_surface, width='stretch')
    
    # High volatility events
    st.markdown("### ⚠️ High Volatility Events")
    
    high_vol_threshold = filtered_df['Volatility_7D'].quantile(0.9)
    high_vol_events = filtered_df[filtered_df['Volatility_7D'] > high_vol_threshold]
    
    fig_events = go.Figure()
    
    fig_events.add_trace(go.Scatter(
        x=filtered_df['Timestamp'],
        y=filtered_df['Close'],
        name='Price',
        line=dict(color='white', width=1)
    ))
    
    fig_events.add_trace(go.Scatter(
        x=high_vol_events['Timestamp'],
        y=high_vol_events['Close'],
        mode='markers',
        name='High Vol Events',
        marker=dict(
            size=12,
            color='#ff0055',
            symbol='star',
            line=dict(color='white', width=2)
        )
    ))
    
    fig_events.update_layout(
        title=f"High Volatility Events (Top 10% - Threshold: {high_vol_threshold:.2f}%)",
        template='plotly_dark',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig_events, width='stretch')

def show_deep_dive(filtered_df):
    """Deep dive analysis with advanced metrics"""
    
    st.markdown("# 🔬 Deep Dive Analysis")
    st.markdown("### Advanced Market Microstructure & Analytics")
    
    st.markdown("---")
    
    # Price momentum analysis
    st.markdown("### 🚀 Momentum Analysis")
    
    # Calculate momentum indicators
    filtered_df['Momentum_1D'] = filtered_df['Close'].pct_change(1) * 100
    filtered_df['Momentum_7D'] = filtered_df['Close'].pct_change(7) * 100
    filtered_df['Momentum_30D'] = filtered_df['Close'].pct_change(30) * 100
    
    fig_momentum = go.Figure()
    
    fig_momentum.add_trace(go.Scatter(
        x=filtered_df['Timestamp'],
        y=filtered_df['Momentum_7D'],
        name='7D Momentum',
        line=dict(color='#00ff88', width=2)
    ))
    
    fig_momentum.add_trace(go.Scatter(
        x=filtered_df['Timestamp'],
        y=filtered_df['Momentum_30D'],
        name='30D Momentum',
        line=dict(color='#ff0055', width=2)
    ))
    
    fig_momentum.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="Zero Line")
    
    fig_momentum.update_layout(
        title="Price Momentum Indicators",
        template='plotly_dark',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x unified',
        yaxis_title="Momentum (%)"
    )
    
    st.plotly_chart(fig_momentum, width='stretch')
    
    # Volume profile
    st.markdown("### 📊 Volume Profile Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Volume by price level
        price_bins = pd.cut(filtered_df['Close'], bins=20)
        volume_profile = filtered_df.groupby(price_bins)['Volume'].sum()
        
        fig_vol_profile = go.Figure()
        
        fig_vol_profile.add_trace(go.Bar(
            y=[str(interval.mid) for interval in volume_profile.index],
            x=volume_profile.values,
            orientation='h',
            marker=dict(
                color=volume_profile.values,
                colorscale='Viridis',
                showscale=True
            ),
            name='Volume by Price'
        ))
        
        fig_vol_profile.update_layout(
            title="Volume Distribution by Price Level",
            template='plotly_dark',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title="Total Volume",
            yaxis_title="Price Level ($)"
        )
        
        st.plotly_chart(fig_vol_profile, width='stretch')
    
    with col2:
        # VWAP (Volume Weighted Average Price)
        filtered_df['VWAP'] = (filtered_df['Close'] * filtered_df['Volume']).cumsum() / filtered_df['Volume'].cumsum()
        
        fig_vwap = go.Figure()
        
        fig_vwap.add_trace(go.Scatter(
            x=filtered_df['Timestamp'],
            y=filtered_df['Close'],
            name='Close Price',
            line=dict(color='white', width=1)
        ))
        
        fig_vwap.add_trace(go.Scatter(
            x=filtered_df['Timestamp'],
            y=filtered_df['VWAP'],
            name='VWAP',
            line=dict(color='#ffd700', width=3, dash='dash')
        ))
        
        fig_vwap.update_layout(
            title="Price vs VWAP",
            template='plotly_dark',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_vwap, width='stretch')
    
    # Market efficiency metrics
    st.markdown("### 🎯 Market Efficiency Metrics")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        # Sharpe ratio (simplified, assuming 0% risk-free rate)
        sharpe = (filtered_df['Daily_Return'].mean() / filtered_df['Daily_Return'].std()) * np.sqrt(252)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        # Max drawdown
        cumulative = (1 + filtered_df['Daily_Return']/100).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_dd = drawdown.min()
        st.metric("Max Drawdown", f"{max_dd:.2f}%")
    
    with col_b:
        # Win rate
        win_rate = (filtered_df['Daily_Return'] > 0).sum() / len(filtered_df['Daily_Return']) * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Average win/loss
        avg_win = filtered_df[filtered_df['Daily_Return'] > 0]['Daily_Return'].mean()
        st.metric("Avg Win", f"{avg_win:.2f}%")
    
    with col_c:
        # Profit factor
        total_gains = filtered_df[filtered_df['Daily_Return'] > 0]['Daily_Return'].sum()
        total_losses = abs(filtered_df[filtered_df['Daily_Return'] < 0]['Daily_Return'].sum())
        profit_factor = total_gains / total_losses if total_losses != 0 else 0
        st.metric("Profit Factor", f"{profit_factor:.2f}")
        
        # Average loss
        avg_loss = filtered_df[filtered_df['Daily_Return'] < 0]['Daily_Return'].mean()
        st.metric("Avg Loss", f"{avg_loss:.2f}%")
    
    # Summary data table
    st.markdown("### 📋 Detailed Data Table")
    
    display_df = filtered_df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'Volatility_7D']].copy()
    display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(
        display_df.tail(50).style.background_gradient(cmap='coolwarm', subset=['Daily_Return']),
        width='stretch',
        height=400
    )

if __name__ == "__main__":
    main()
