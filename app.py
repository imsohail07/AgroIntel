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
    temperature_vs_yield,
    avg_yield_by_season,
    rainfall_distribution,
    irrigation_yield_comparison,
    crop_area_pie,
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

    /* Sidebar — force readable text in both light and dark mode */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1b15 0%, #132a1c 100%) !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] span {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stButton>button {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white !important;
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
    .about-card h4 {
        color: #1e293b !important;
        margin: 0 0 0.5rem;
    }
    .about-card p,
    .about-card li,
    .about-card strong {
        color: #334155 !important;
    }
    .about-card strong {
        color: #0f172a !important;
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

        # ── Before vs After Revenue Comparison ──
        st.markdown("### 📊 Revenue: Before vs After Recommendations")
        st.markdown("_This chart shows your **current estimated revenue** vs the **projected revenue if you implement all AgroIntel recommendations**._")
        
        before_rev = profit["revenue"]
        after_rev = profit["improved_profit"] + profit["total_cost"]  # improved revenue
        before_profit = profit["net_profit"]
        after_profit = profit["improved_profit"]
        
        import plotly.graph_objects as go
        fig_ba = go.Figure(data=[
            go.Bar(name="Before (Current)", x=["Revenue", "Net Profit"], y=[before_rev, before_profit],
                   marker_color=["#f97316", "#f97316"], text=[f"₹{before_rev:,.0f}", f"₹{before_profit:,.0f}"], textposition="outside"),
            go.Bar(name="After (Optimized)", x=["Revenue", "Net Profit"], y=[after_rev, after_profit],
                   marker_color=["#10b981", "#10b981"], text=[f"₹{after_rev:,.0f}", f"₹{after_profit:,.0f}"], textposition="outside"),
        ])
        fig_ba.update_layout(
            barmode="group", template="plotly_white", height=420,
            title=f"Financial Impact of Implementing Recommendations ({input_data['Crop_Type']})",
            yaxis_title="Amount (₹)",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_ba, use_container_width=True)
        
        if profit["improvement_pct"] > 0:
            st.success(f"💡 **Potential Uplift:** By following our recommendations, you could increase your profit by **₹{profit['profit_uplift']:,.0f}** (+{profit['improvement_pct']}%).")
        else:
            st.info("✅ Your current parameters are already optimized. No improvement actions needed.")
        
        # ── Cross-Crop Profit Comparison ──
        st.markdown("### 🌾 Profit Comparison Across All Crops")
        st.markdown("_Estimated profit if the **same farm conditions** were applied to different crops (for the same area and market prices)._")
        from src.profit_estimator import MARKET_PRICES, BASE_COST_PER_HA
        crop_names = list(MARKET_PRICES.keys())
        crop_revenues = []
        crop_costs = []
        crop_profits = []
        for crop in crop_names:
            rev = predicted_yield * input_data["Area_ha"] * MARKET_PRICES[crop]
            cost = BASE_COST_PER_HA[crop] * input_data["Area_ha"]
            crop_revenues.append(rev)
            crop_costs.append(cost)
            crop_profits.append(rev - cost)
        
        fig_crop = go.Figure(data=[
            go.Bar(name="Revenue", x=crop_names, y=crop_revenues, marker_color="#3b82f6",
                   text=[f"₹{v:,.0f}" for v in crop_revenues], textposition="outside"),
            go.Bar(name="Cost", x=crop_names, y=crop_costs, marker_color="#ef4444",
                   text=[f"₹{v:,.0f}" for v in crop_costs], textposition="outside"),
            go.Bar(name="Net Profit", x=crop_names, y=crop_profits, marker_color="#10b981",
                   text=[f"₹{v:,.0f}" for v in crop_profits], textposition="outside"),
        ])
        fig_crop.update_layout(
            barmode="group", template="plotly_white", height=450,
            title="Revenue, Cost & Net Profit by Crop Type",
            xaxis_title="Crop", yaxis_title="Amount (₹)",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_crop, use_container_width=True)

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
        st.markdown("A full portfolio-level analysis of your uploaded farms, with **business impact** for each issue detected.")
        n = len(pred_df)
        found_any = False
        
        # ── 1. Nitrogen ──
        low_n = len(pred_df[pred_df["Nitrogen"] < 40]) if "Nitrogen" in pred_df.columns else 0
        if low_n > 0:
            found_any = True
            st.error(f"""🧪 **Low Nitrogen — {low_n} farms ({low_n/n*100:.1f}%)**  
            ⚙️ **Action:** Bulk purchase Urea @ 50–80 kg/ha for affected farms.  
            💼 **Business Impact:** Nitrogen is the single largest driver of vegetative growth. Correcting this deficiency can boost yields by **10–15%**, directly increasing gross revenue by an estimated ₹2,000–₹4,000 per hectare per season.""")
        
        # ── 2. Phosphorus ──
        low_p = len(pred_df[pred_df["Phosphorus"] < 30]) if "Phosphorus" in pred_df.columns else 0
        if low_p > 0:
            found_any = True
            st.error(f"""🧪 **Low Phosphorus — {low_p} farms ({low_p/n*100:.1f}%)**  
            ⚙️ **Action:** Apply DAP (Diammonium Phosphate) @ 40–60 kg/ha.  
            💼 **Business Impact:** Phosphorus deficiency stunts root development, reducing both yield quality and quantity. Fertilizer correction costs ~₹1,200/ha but recovers an estimated **₹3,000–₹5,000/ha** in additional revenue.""")
        
        # ── 3. Potassium ──
        low_k = len(pred_df[pred_df["Potassium"] < 30]) if "Potassium" in pred_df.columns else 0
        if low_k > 0:
            found_any = True
            st.warning(f"""🧪 **Low Potassium — {low_k} farms ({low_k/n*100:.1f}%)**  
            ⚙️ **Action:** Apply MOP (Muriate of Potash) @ 30–50 kg/ha.  
            💼 **Business Impact:** Low Potassium weakens plant immunity, increasing vulnerability to diseases and pests. Correction reduces crop-loss risk by up to **20%**, protecting the existing revenue baseline.""")
        
        # ── 4. Soil pH ──
        if "pH" in pred_df.columns:
            acidic = len(pred_df[pred_df["pH"] < 5.5])
            alkaline = len(pred_df[pred_df["pH"] > 7.5])
            if acidic > 0:
                found_any = True
                st.error(f"""⚗️ **Acidic Soil (pH < 5.5) — {acidic} farms ({acidic/n*100:.1f}%)**  
                ⚙️ **Action:** Apply agricultural lime (CaCO₃) @ 2–4 tons/ha.  
                💼 **Business Impact:** Acidic soil locks out N, P, K nutrients. Liming is a one-time cost of ~₹3,000/ha that unlocks **₹8,000–₹12,000/ha of hidden yield potential** over 2–3 seasons.""")
            if alkaline > 0:
                found_any = True
                st.warning(f"""⚗️ **Alkaline Soil (pH > 7.5) — {alkaline} farms ({alkaline/n*100:.1f}%)**  
                ⚙️ **Action:** Apply gypsum (CaSO₄) @ 2–5 tons/ha.  
                💼 **Business Impact:** High pH causes iron and zinc deficiency. Correcting it can improve grain quality, potentially qualifying for premium market pricing (+5–10% revenue uplift).""")
        
        # ── 5. Low Rainfall / Irrigation ──
        if "Rainfall_mm" in pred_df.columns:
            low_rain = len(pred_df[pred_df["Rainfall_mm"] < 600])
            if low_rain > 0:
                found_any = True
                st.error(f"""💧 **Low Rainfall (< 600mm) — {low_rain} farms ({low_rain/n*100:.1f}%)**  
                ⚙️ **Action:** Deploy drip or sprinkler irrigation systems on affected farms.  
                💼 **Business Impact:** Farms without sufficient water lose up to **40% of potential yield**. A ₹5,000/ha irrigation investment can recover ₹15,000–₹25,000/ha in saved revenue per season.""")
        
        # ── 6. High Humidity → Disease Risk ──
        if "Humidity_pct" in pred_df.columns:
            high_hum = len(pred_df[pred_df["Humidity_pct"] > 85])
            if high_hum > 0:
                found_any = True
                st.warning(f"""🦠 **High Humidity (> 85%) — {high_hum} farms ({high_hum/n*100:.1f}%)**  
                ⚙️ **Action:** Increase monitoring for fungal diseases (blast, blight). Apply preventive fungicide.  
                💼 **Business Impact:** Unchecked fungal infections in humid conditions cause 15–30% crop loss. Preventive fungicide costs ~₹800/ha but prevents potential losses of ₹5,000–₹10,000/ha.""")
        
        # ── 7. High Temperature Stress ──
        if "Temperature_C" in pred_df.columns:
            high_temp = len(pred_df[pred_df["Temperature_C"] > 38])
            if high_temp > 0:
                found_any = True
                st.warning(f"""🌡️ **Heat Stress (> 38°C) — {high_temp} farms ({high_temp/n*100:.1f}%)**  
                ⚙️ **Action:** Use mulching and adjust sowing dates to avoid peak heat. Consider heat-tolerant crop varieties.  
                💼 **Business Impact:** Every 1°C above optimum during the flowering stage can reduce grain yield by 5–8%. Switching to heat-tolerant varieties stabilizes revenue in volatile climates.""")
        
        # ── 8. Crop Diversification ──
        if "Crop_Type" in pred_df.columns:
            dominant_crop = pred_df["Crop_Type"].value_counts()
            if len(dominant_crop) > 0:
                top_crop = dominant_crop.index[0]
                top_pct = dominant_crop.iloc[0] / n * 100
                if top_pct > 50:
                    found_any = True
                    st.info(f"""🌱 **Low Crop Diversity — {top_pct:.1f}% farms grow {top_crop}**  
                    ⚙️ **Action:** Encourage crop rotation and diversification (e.g., alternate with legumes like Soybean for natural nitrogen fixation).  
                    💼 **Business Impact:** Monoculture increases soil degradation and market-price risk. Diversification reduces input costs by 10–15% over 3 years (natural nutrient replenishment) and protects against single-commodity price crashes.""")
        
        # ── 9. Underperforming Yield Alert ──
        if "Predicted_Yield_ton_per_ha" in pred_df.columns and "Crop_Type" in pred_df.columns:
            median_yields = {"Rice": 4.2, "Wheat": 3.2, "Maize": 4.8, "Cotton": 1.9, "Sugarcane": 65.0, "Soybean": 2.3}
            underperf_count = 0
            for _, row in pred_df.iterrows():
                crop = row.get("Crop_Type", "")
                pred_y = row.get("Predicted_Yield_ton_per_ha", 0)
                med = median_yields.get(crop, 0)
                if med > 0 and pred_y < med * 0.7:
                    underperf_count += 1
            if underperf_count > 0:
                found_any = True
                st.error(f"""📉 **Underperforming Farms — {underperf_count} farms ({underperf_count/n*100:.1f}%) predicted below 70% of median yield**  
                ⚙️ **Action:** Conduct a detailed soil test and consider switching to higher-yield crop alternatives (e.g., Maize instead of Cotton, Soybean instead of Wheat).  
                💼 **Business Impact:** Each underperforming farm is leaving 30%+ potential revenue on the table. Strategic crop switching alone can uplift per-hectare income by ₹8,000–₹15,000 without additional infrastructure cost.""")
        
        if not found_any:
            st.success("✅ **All Parameters Look Good!** No widespread issues found across the uploaded dataset. Current farming practices are within optimal ranges.")
        
        st.markdown("---")
        st.markdown("_For detailed individual farm analysis with SHAP feature explanations, switch to **Manual Entry** mode._")

    with tabs[2]:
        st.markdown("## 💰 Batch Financial Valuation")
        st.info("💡 Revenue estimations are based on current average market prices per ton (e.g. Rice: ₹22,000/ton, Wheat: ₹25,000/ton, Cotton: ₹55,000/ton).")
        # Calculate profit for the whole batch — use safe defaults for missing columns
        total_rev = 0.0
        total_cost = 0.0
        for i, row in pred_df.iterrows():
            row_dict = row.to_dict()
            # Ensure required keys have safe default values
            if "Crop_Type" not in row_dict or pd.isna(row_dict.get("Crop_Type")):
                row_dict["Crop_Type"] = "Rice"
            if "Area_ha" not in row_dict or pd.isna(row_dict.get("Area_ha")):
                row_dict["Area_ha"] = 1.0
            pred_yield = row.get("Predicted_Yield_ton_per_ha", 0)
            if pd.isna(pred_yield):
                pred_yield = 0.0
            rec = generate_recommendations(row_dict, pred_yield)
            prof = estimate_profit(row_dict, pred_yield, rec)
            rev = prof["revenue"] if not pd.isna(prof["revenue"]) else 0.0
            cost = prof["total_cost"] if not pd.isna(prof["total_cost"]) else 0.0
            total_rev += rev
            total_cost += cost
            
        net_profit = total_rev - total_cost
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="label">Total Revenue</div><div class="value">₹{total_rev:,.0f}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="label">Total Cost</div><div class="value">₹{total_cost:,.0f}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card profit-{"positive" if net_profit >=0 else "negative"}"><div class="label">Total Net Profit</div><div class="value">₹{net_profit:,.0f}</div></div>', unsafe_allow_html=True)
        
        # ── Per-Crop Profit Comparison Bar Chart ──
        st.markdown("### 🌾 Profit Comparison Across Crops")
        import plotly.graph_objects as go
        from src.profit_estimator import MARKET_PRICES, BASE_COST_PER_HA
        
        # Aggregate revenue & cost per crop from the batch
        crop_rev_map = {}
        crop_cost_map = {}
        crop_improved_rev_map = {}
        for i, row in pred_df.iterrows():
            row_dict = row.to_dict()
            crop = row_dict.get("Crop_Type", "Rice")
            if pd.isna(crop): crop = "Rice"
            area = row_dict.get("Area_ha", 1.0)
            if pd.isna(area): area = 1.0
            pred_yield = row.get("Predicted_Yield_ton_per_ha", 0)
            if pd.isna(pred_yield): pred_yield = 0.0
            rec = generate_recommendations(row_dict, pred_yield)
            prof = estimate_profit(row_dict, pred_yield, rec)
            rev = prof["revenue"] if not pd.isna(prof["revenue"]) else 0.0
            cost = prof["total_cost"] if not pd.isna(prof["total_cost"]) else 0.0
            imp_rev = (prof["improved_profit"] + prof["total_cost"]) if not pd.isna(prof.get("improved_profit", 0)) else rev
            crop_rev_map[crop] = crop_rev_map.get(crop, 0) + rev
            crop_cost_map[crop] = crop_cost_map.get(crop, 0) + cost
            crop_improved_rev_map[crop] = crop_improved_rev_map.get(crop, 0) + imp_rev
        
        crops = list(crop_rev_map.keys())
        revenues = [crop_rev_map[c] for c in crops]
        costs = [crop_cost_map[c] for c in crops]
        profits = [crop_rev_map[c] - crop_cost_map[c] for c in crops]
        
        fig_crop = go.Figure(data=[
            go.Bar(name="Revenue", x=crops, y=revenues, marker_color="#3b82f6",
                   text=[f"₹{v:,.0f}" for v in revenues], textposition="outside"),
            go.Bar(name="Cost", x=crops, y=costs, marker_color="#ef4444",
                   text=[f"₹{v:,.0f}" for v in costs], textposition="outside"),
            go.Bar(name="Net Profit", x=crops, y=profits, marker_color="#10b981",
                   text=[f"₹{v:,.0f}" for v in profits], textposition="outside"),
        ])
        fig_crop.update_layout(
            barmode="group", template="plotly_white", height=450,
            title="Revenue, Cost & Net Profit by Crop Type",
            xaxis_title="Crop", yaxis_title="Amount (₹)",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_crop, use_container_width=True)
        
        # ── Before vs After Revenue Comparison ──
        st.markdown("### 📊 Revenue: Before vs After Recommendations")
        st.markdown("_Compares **current revenue** against **projected revenue after implementing AgroIntel's recommendations** for each crop._")
        
        before_vals = [crop_rev_map[c] for c in crops]
        after_vals = [crop_improved_rev_map[c] for c in crops]
        uplift_vals = [a - b for a, b in zip(after_vals, before_vals)]
        
        fig_ba = go.Figure(data=[
            go.Bar(name="Before (Current Revenue)", x=crops, y=before_vals, marker_color="#f97316",
                   text=[f"₹{v:,.0f}" for v in before_vals], textposition="outside"),
            go.Bar(name="After (Optimized Revenue)", x=crops, y=after_vals, marker_color="#10b981",
                   text=[f"₹{v:,.0f}" for v in after_vals], textposition="outside"),
        ])
        fig_ba.update_layout(
            barmode="group", template="plotly_white", height=450,
            title="Revenue Before vs After Action (Per Crop)",
            xaxis_title="Crop", yaxis_title="Revenue (₹)",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_ba, use_container_width=True)
        
        # Show total uplift
        total_uplift = sum(uplift_vals)
        if total_uplift > 0:
            st.success(f"💡 **Total Revenue Uplift Potential:** By following all AgroIntel recommendations, the portfolio's revenue could increase by **₹{total_uplift:,.0f}**.")
        else:
            st.info("✅ Current parameters are already well-optimized. No significant improvement actions needed.")
        
    with tabs[3]:
        st.markdown("## 📊 Batch Data Insights")
        # Row 1: Yield vs Rainfall + Temperature
        col_a, col_b = st.columns(2)
        if "Rainfall_mm" in pred_df.columns and "Yield_ton_per_ha" in pred_df.columns:
            col_a.plotly_chart(yield_vs_rainfall(pred_df), use_container_width=True)
        fig_temp = temperature_vs_yield(pred_df)
        if fig_temp:
            col_b.plotly_chart(fig_temp, use_container_width=True)
        # Row 2: Distribution + Pie chart
        col_c, col_d = st.columns(2)
        if "Crop_Type" in pred_df.columns and "Yield_ton_per_ha" in pred_df.columns:
            col_c.plotly_chart(yield_distribution_by_crop(pred_df), use_container_width=True)
        fig_pie = crop_area_pie(pred_df)
        if fig_pie:
            col_d.plotly_chart(fig_pie, use_container_width=True)
        # Row 3: Irrigation comparison + Avg Yield by Season
        col_e, col_f = st.columns(2)
        fig_irr = irrigation_yield_comparison(pred_df)
        if fig_irr:
            col_e.plotly_chart(fig_irr, use_container_width=True)
        fig_season = avg_yield_by_season(pred_df)
        if fig_season:
            col_f.plotly_chart(fig_season, use_container_width=True)
        # Row 4: Correlation Heatmap + Rainfall histogram
        col_g, col_h = st.columns(2)
        col_g.plotly_chart(correlation_heatmap(pred_df), use_container_width=True)
        fig_rain = rainfall_distribution(pred_df)
        if fig_rain:
            col_h.plotly_chart(fig_rain, use_container_width=True)
        # Row 5: Soil nutrients
        if all(c in pred_df.columns for c in ["Nitrogen", "Phosphorus", "Potassium"]):
            st.plotly_chart(soil_nutrients_vs_yield(pred_df), use_container_width=True)

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
        # Row 1
        col_a, col_b = st.columns(2)
        col_a.plotly_chart(yield_vs_rainfall(training_df), use_container_width=True)
        fig_temp = temperature_vs_yield(training_df)
        if fig_temp:
            col_b.plotly_chart(fig_temp, use_container_width=True)
        # Row 2
        col_c, col_d = st.columns(2)
        col_c.plotly_chart(yield_distribution_by_crop(training_df), use_container_width=True)
        fig_pie = crop_area_pie(training_df)
        if fig_pie:
            col_d.plotly_chart(fig_pie, use_container_width=True)
        # Row 3
        col_e, col_f = st.columns(2)
        fig_irr = irrigation_yield_comparison(training_df)
        if fig_irr:
            col_e.plotly_chart(fig_irr, use_container_width=True)
        fig_season = avg_yield_by_season(training_df)
        if fig_season:
            col_f.plotly_chart(fig_season, use_container_width=True)
        # Row 4
        col_g, col_h = st.columns(2)
        col_g.plotly_chart(correlation_heatmap(training_df), use_container_width=True)
        fig_rain = rainfall_distribution(training_df)
        if fig_rain:
            col_h.plotly_chart(fig_rain, use_container_width=True)
        st.plotly_chart(soil_nutrients_vs_yield(training_df), use_container_width=True)

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
