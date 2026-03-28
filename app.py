"""
🌾 Adaptive Crop Yield Prediction & Farmer Profit Optimization System
Streamlit Dashboard — Main Entry Point
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np

from pipeline import CropYieldPipeline
from recommendations import generate_recommendations
from profit_estimator import estimate_profit
from visualizations import (
    yield_vs_rainfall,
    correlation_heatmap,
    predicted_vs_actual,
    soil_nutrients_vs_yield,
    yield_distribution_by_crop,
)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 AgroIntel — Smart Farming Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.3rem 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #bbf7d0;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(5,150,105,0.15);
    }
    .metric-card .value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #059669;
        margin: 0.3rem 0;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Profit cards */
    .profit-positive { border-left: 5px solid #10b981; }
    .profit-negative { border-left: 5px solid #ef4444; }

    /* Recommendation cards */
    .rec-card {
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 5px solid;
        background: #fff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .rec-high   { border-color: #ef4444; background: #fef2f2; }
    .rec-medium { border-color: #f59e0b; background: #fffbeb; }
    .rec-low    { border-color: #10b981; background: #f0fdf4; }
    .rec-card h4 { margin: 0 0 0.3rem; font-size: 1rem; }
    .rec-card p  { margin: 0; font-size: 0.9rem; color: #374151; }

    /* SHAP explanation */
    .shap-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        border-radius: 8px;
        font-size: 0.9rem;
    }
    .shap-positive { background: #f0fdf4; color: #065f46; }
    .shap-negative { background: #fef2f2; color: #991b1b; }
    .shap-icon { margin-right: 0.6rem; font-size: 1.1rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    section[data-testid="stSidebar"] .stButton>button {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        width: 100%;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(5,150,105,0.3);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
    }

    /* About section */
    .about-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load and Train Pipeline in Memory ──────────────────────────────────────────
@st.cache_resource(show_spinner="Training model in memory...")
def get_pipeline():
    """Reads base dataset and trains ML models dynamically in-memory without .pkl files."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "crop_data.csv")
    df = pd.read_csv(data_path)
    pipeline = CropYieldPipeline()
    pipeline.fit(df)
    return pipeline, df

pipeline, training_df = get_pipeline()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌾 AgroIntel</h1>
    <p>Adaptive Crop Yield Prediction & Farmer Profit Optimization System</p>
</div>
""", unsafe_allow_html=True)


# ── Input Method ───────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Input Method")
input_mode = st.sidebar.radio("Choose Mode:", ["Batch Upload (CSV/Excel)", "Manual Entry Form"])
st.sidebar.markdown("---")


# ── State Initialization ───────────────────────────────────────────────────────
if "prediction_df" not in st.session_state:
    st.session_state.prediction_df = None
if "single_input" not in st.session_state:
    st.session_state.single_input = None

predict_btn = False
uploaded_file = None

# Sidebar Content Based on Mode
if input_mode == "Manual Entry Form":
    st.sidebar.markdown("### 🧑‍🌾 Farm & Soil Data")
    
    crop_type = st.sidebar.selectbox("🌱 Crop Type", ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean"])
    season = st.sidebar.selectbox("📅 Season", ["Kharif", "Rabi", "Zaid"])
    irrigation = st.sidebar.selectbox("💧 Irrigation Type", ["Rainfed", "Drip", "Sprinkler", "Flood"])

    st.sidebar.markdown("#### 🌦️ Weather Conditions")
    rainfall = st.sidebar.slider("Rainfall (mm)", 100, 3000, 800, step=50)
    temperature = st.sidebar.slider("Temperature (°C)", 5, 45, 25)
    humidity = st.sidebar.slider("Humidity (%)", 20, 100, 60)

    st.sidebar.markdown("#### 🧪 Soil Nutrients")
    nitrogen = st.sidebar.slider("Nitrogen (N)", 5, 140, 60)
    phosphorus = st.sidebar.slider("Phosphorus (P)", 5, 100, 45)
    potassium = st.sidebar.slider("Potassium (K)", 5, 110, 50)
    ph = st.sidebar.slider("Soil pH", 4.5, 8.5, 6.5, step=0.1)

    st.sidebar.markdown("#### 📐 Farm Area")
    area = st.sidebar.slider("Area (hectares)", 0.5, 20.0, 5.0, step=0.5)

    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button("🚀 Predict Yield & Optimize", use_container_width=True)
    
    if predict_btn:
        st.session_state.single_input = {
            "Crop_Type": crop_type, "Season": season, "Irrigation_Type": irrigation,
            "Rainfall_mm": rainfall, "Temperature_C": temperature, "Humidity_pct": humidity,
            "Nitrogen": nitrogen, "Phosphorus": phosphorus, "Potassium": potassium,
            "pH": ph, "Area_ha": area
        }
        st.session_state.prediction_df = None  # Clear batch results
        
elif input_mode == "Batch Upload (CSV/Excel)":
    st.sidebar.markdown("### 📁 Upload Dataset")
    st.sidebar.markdown("Upload a CSV or Excel file containing farm attributes for batch valuations.")
    
    uploaded_file = st.sidebar.file_uploader("Upload dataset", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)
                
            st.sidebar.success(f"Loaded {len(df_input)} rows.")
            
            if st.sidebar.button("🚀 Run Batch Valuation", use_container_width=True):
                preds, X_scaled = pipeline.predict(df_input)
                df_input["Predicted_Yield_ton_per_ha"] = np.round(preds, 2)
                st.session_state.prediction_df = df_input
                st.session_state.single_input = None  # Clear single results
                st.session_state.X_scaled = X_scaled
                
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")

# ── Main Content Tabs ──────────────────────────────────────────────────────────
tabs = st.tabs(["🌾 Yield Valuation", "💡 Recommendations", "💰 Profit Estimation", "📊 Data Insights", "ℹ️ About"])

# ── SINGLE ENTRY VIEW ──
if st.session_state.single_input is not None and input_mode == "Manual Entry Form":
    input_data = st.session_state.single_input
    
    df_single = pd.DataFrame([input_data])
    preds, X_scaled = pipeline.predict(df_single)
    predicted_yield = float(preds[0])
    
    explanations = pipeline.explain_prediction(X_scaled)
    recs = generate_recommendations(input_data, predicted_yield)
    profit = estimate_profit(input_data, predicted_yield, recs)

    with tabs[0]:
        st.markdown("## 🎯 Predicted Crop Yield (Manual)")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="metric-card"><div class="label">Predicted Yield</div><div class="value">{predicted_yield:.2f} t/ha</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><div class="label">Total Production</div><div class="value">{predicted_yield * input_data["Area_ha"]:.1f} tons</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><div class="label">Crop &amp; Area</div><div class="value">{input_data["Crop_Type"]}</div><div class="label">{input_data["Area_ha"]} hectares</div></div>', unsafe_allow_html=True)
        
        st.markdown("### 🔍 Key Influencing Factors (SHAP)")
        for exp in explanations:
            css_class = "shap-positive" if exp["direction"] == "increases" else "shap-negative"
            icon = "📈" if exp["direction"] == "increases" else "📉"
            st.markdown(f'<div class="shap-item {css_class}"><span class="shap-icon">{icon}</span>{exp["text"]}</div>', unsafe_allow_html=True)
            
    with tabs[1]:
        st.markdown("## 💡 Actionable Recommendations")
        for rec in recs:
            st.markdown(f'<div class="rec-card rec-{rec["severity"]}"><h4>{rec["icon"]} {rec["title"]}</h4><p><strong>Category:</strong> {rec["category"]} | <strong>Priority:</strong> {rec["severity"].upper()}</p><p>{rec["action"]}</p></div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("## 💰 Financial Valuation")
        st.info(f"💡 Revenue is calculated assuming an average market price of **₹{profit['market_price_per_ton']:,.0f} per ton** for **{input_data['Crop_Type']}**.")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="label">Revenue</div><div class="value">₹{profit["revenue"]:,.0f}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="label">Total Cost</div><div class="value">₹{profit["total_cost"]:,.0f}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card profit-{"positive" if profit["net_profit"] >=0 else "negative"}"><div class="label">Net Profit</div><div class="value">₹{profit["net_profit"]:,.0f}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="label">Profit / Hectare</div><div class="value">₹{profit["profit_per_ha"]:,.0f}</div></div>', unsafe_allow_html=True)

    with tabs[3]:
        st.markdown("## 📊 Historical Data Insights")
        st.plotly_chart(yield_vs_rainfall(training_df), use_container_width=True)
        col_a, col_b = st.columns(2)
        col_a.plotly_chart(correlation_heatmap(training_df), use_container_width=True)
        col_b.info("Predicted vs Actual requires test data metrics. Currently showing historical insights.")
        st.plotly_chart(yield_distribution_by_crop(training_df), use_container_width=True)
        
# ── BATCH PREDICTION VIEW ──
elif st.session_state.prediction_df is not None and input_mode == "Batch Upload (CSV/Excel)":
    pred_df = st.session_state.prediction_df
    X_scaled = st.session_state.X_scaled
    
    avg_yield = pred_df["Predicted_Yield_ton_per_ha"].mean()
    total_area = pred_df["Area_ha"].sum() if "Area_ha" in pred_df.columns else len(pred_df)
    total_production = (pred_df["Predicted_Yield_ton_per_ha"] * pred_df.get("Area_ha", 1)).sum()
    
    with tabs[0]:
        st.markdown("## 🎯 Batch Valuation Results")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="metric-card"><div class="label">Average Yield</div><div class="value">{avg_yield:.2f} t/ha</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><div class="label">Total Production</div><div class="value">{total_production:.1f} tons</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><div class="label">Total Farms Analyzed</div><div class="value">{len(pred_df)}</div></div>', unsafe_allow_html=True)
        
        st.markdown("### 📋 Prediction Table")
        st.dataframe(pred_df, use_container_width=True)
        
        st.markdown("### 📊 Global Feature Importance (Batch)")
        st.plotly_chart(pipeline.generate_global_importance(X_scaled), use_container_width=True)
        
    with tabs[1]:
        st.markdown("## 💡 Aggregate Recommendations")
        st.markdown("Reviewing the batch to identify widespread issues...")
        
        # Simple aggregate logic
        low_n = len(pred_df[pred_df["Nitrogen"] < 40]) if "Nitrogen" in pred_df.columns else 0
        low_p = len(pred_df[pred_df["Phosphorus"] < 30]) if "Phosphorus" in pred_df.columns else 0
        low_k = len(pred_df[pred_df["Potassium"] < 30]) if "Potassium" in pred_df.columns else 0
        
        if low_n > 0:
            st.warning(f"🧪 **{low_n} farms ({low_n/len(pred_df)*100:.1f}%)** have critically **Low Nitrogen**. Consider bulk purchasing Urea.")
        if low_p > 0:
            st.warning(f"🧪 **{low_p} farms ({low_p/len(pred_df)*100:.1f}%)** have **Low Phosphorus**. Consider applying DAP.")
        if low_k > 0:
            st.info(f"🧪 **{low_k} farms ({low_k/len(pred_df)*100:.1f}%)** have **Low Potassium**.")
            
        st.markdown("For individual farm recommendations, please use the Manual mode.")

    with tabs[2]:
        st.markdown("## 💰 Batch Financial Valuation")
        st.info("💡 Revenue estimations are based on current average market prices per ton (e.g. Rice: ₹22,000/ton, Wheat: ₹25,000/ton, Cotton: ₹55,000/ton).")
        # Calculate profit for the whole batch
        total_rev = 0
        total_cost = 0
        for i, row in pred_df.iterrows():
            rec = generate_recommendations(row.to_dict(), row["Predicted_Yield_ton_per_ha"])
            prof = estimate_profit(row.to_dict(), row["Predicted_Yield_ton_per_ha"], rec)
            total_rev += prof["revenue"]
            total_cost += prof["total_cost"]
            
        net_profit = total_rev - total_cost
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="label">Total Revenue</div><div class="value">₹{total_rev:,.0f}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="label">Total Cost</div><div class="value">₹{total_cost:,.0f}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card profit-{"positive" if net_profit >=0 else "negative"}"><div class="label">Total Net Profit</div><div class="value">₹{net_profit:,.0f}</div></div>', unsafe_allow_html=True)
        
    with tabs[3]:
        st.markdown("## 📊 Batch Data Insights")
        if "Rainfall_mm" in pred_df.columns:
            st.plotly_chart(yield_vs_rainfall(pred_df), use_container_width=True)
        if "Crop_Type" in pred_df.columns:
            st.plotly_chart(yield_distribution_by_crop(pred_df), use_container_width=True)

# ── DEFAULT VIEW ──
else:
    with tabs[0]:
        st.markdown("## 👋 Welcome to AgroIntel!")
        st.markdown("""
        Select your **Input Method** from the sidebar:
        
        - 📁 **Batch Upload (CSV/Excel)**: Upload an entire dataset for bulk prediction, portfolio financial valuation, and aggregate insights.
        - 🧑‍🌾 **Manual Entry Form**: Enter specific farm conditions for detailed SHAP explainability and individual recommendations.
        """)
        
    with tabs[3]:
        st.markdown("## 📊 Model Training Data Insights")
        st.plotly_chart(yield_vs_rainfall(training_df), use_container_width=True)

# ── ABOUT TAB ──
with tabs[4]:
    st.markdown("## ℹ️ About This System")
    st.markdown("""
    <div class="about-card">
        <h4>🌾 AgroIntel — Empowering Sustainable Agriculture</h4>
        <p>AgroIntel is an end-to-end machine learning platform designed to bridge the gap between advanced data science and practical farming. By providing highly accurate predictive models and actionable agronomic advice, we aim to augment farmer decision-making.</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="about-card">
            <h4>🌟 Value to Society</h4>
            <ul>
                <li><strong>Enhancing Food Security:</strong> Accurate predictions help stabilize national food supply chains.</li>
                <li><strong>Economic Upliftment:</strong> Data-driven profit estimations ensure farmers maximize their income and mitigate financial risks.</li>
                <li><strong>Sustainable Farming:</strong> Targeted recommendations for fertilizer and irrigation prevent resource waste and protect soil health.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="about-card">
            <h4>📈 System Performance</h4>
            <ul>
                <li><strong>Model Architecture:</strong> Extreme Gradient Boosting (XGBoost Regressor) trained on multi-dimensional climatic and soil data.</li>
                <li><strong>Accuracy Metrics:</strong> The core predictive model achieves a high R&sup2; Score of <strong>~0.87</strong>.</li>
                <li><strong>Interpretability:</strong> Global and local feature importances are isolated using advanced SHAP (SHapley Additive exPlanations) values to ensure AI transparency.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
