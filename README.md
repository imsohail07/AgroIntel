# 🌾 AgroIntel

**AgroIntel** is an end-to-end, AI-driven Smart Farming Assistant developed to help farmers maximize crop yield and optimize their financial returns. By leveraging an Extreme Gradient Boosting (XGBoost) Machine Learning algorithm alongside real-world agricultural metrics (soil nutrients, weather patterns, and farm area), AgroIntel delivers actionable agronomic recommendations and detailed profit estimations.

## 🌟 Key Features

*   **🌱 Advanced Yield Prediction:** Utilizes an XGBoost Regressor to predict crop yield (tons per hectare) with an accuracy R² score of ~0.87.
*   **🧠 Explainable AI (SHAP):** Transparent predictions. The system breaks down *exactly* why a specific yield was predicted by displaying the positive or negative influence of each feature (e.g., Rainfall, Nitrogen, pH).
*   **💡 Actionable Recommendations:** A robust rule-based engine that converts raw data into prioritized farming advice (e.g., "Add 50kg of Urea due to critical Nitrogen shortage").
*   **💰 Financial Valuation:** Calculates estimated Cultivation Cost, Total Revenue (based on aggregated market prices), and Net Profit. It also calculates potential profit *uplift* if recommendations are followed.
*   **📁 Batch Processing:** Upload hundreds of farm records simultaneously via CSV or Excel to receive portfolio-level financial valuations and aggregate problem detection.
*   **📊 Interactive Data Insights:** Built-in historical data views using Plotly for visual correlation, feature distribution, and crop analysis.

## 🚀 Tech Stack

*   **Machine Learning:** Scikit-learn, XGBoost, SHAP (SHapley Additive exPlanations)
*   **Data Processing:** Pandas, NumPy
*   **Frontend/UI:** Streamlit
*   **Visualizations:** Plotly Express & Graph Objects

## ⚙️ How It Works (Methodology)

AgroIntel operates completely **in-memory**. When the application launches, it dynamically trains the entire predictive pipeline using the base dataset, avoiding the need for opaque, pre-saved `.pkl` files.

1.  **Data Ingestion:** Reads farm parameters (N-P-K levels, pH, rainfall, temperature, humidity, crop type, season, area).
2.  **Feature Engineering:** Computes a custom `Soil_Fertility_Index` and `Weather_Index`.
3.  **Model Inference:** The scaled features are fed into the XGBoost Regressor.
4.  **Explainability:** The SHAP TreeExplainer generates the precise percentage impact of the top features.
5.  **Output Generation:** Aggregates recommendations and matches predicted yield with real-world market prices to estimate net profit.

## 💻 Running the Project Locally

### Prerequisites
*   Python 3.9+
*   Pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/imsohail07/AgroIntel.git
    cd AgroIntel
    ```

2.  **Create a Virtual Environment (Optional but recommended):**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # Windows
    # source venv/bin/activate  # macOS/Linux
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the Application:**
    ```bash
    streamlit run app.py
    ```

5.  Open `http://localhost:8501` in your browser.

## 🌍 Value to Society

*   **Enhancing Food Security:** Accurate predictions help stabilize national food supply chains.
*   **Economic Upliftment:** Data-driven estimations ensure farmers maximize their income and mitigate financial risks.
*   **Sustainable Farming:** Targeted recommendations for fertilizer and irrigation prevent resource waste and protect soil health.
