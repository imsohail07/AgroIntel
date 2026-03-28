import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def yield_vs_rainfall(df: pd.DataFrame):
    fig = px.scatter(
        df, x="Rainfall_mm", y="Yield_ton_per_ha",
        color="Crop_Type",
        title="Crop Yield vs Rainfall",
        labels={"Rainfall_mm": "Rainfall (mm)", "Yield_ton_per_ha": "Yield (t/ha)"},
        opacity=0.6,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=450, legend_title_text="Crop", margin=dict(l=10, r=10, t=40, b=10))
    return fig

def correlation_heatmap(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig.update_layout(title="Feature Correlation Heatmap", height=500, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
    return fig

def predicted_vs_actual(y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", marker=dict(color="#6366f1", opacity=0.5, size=6), name="Predictions"))
    # Perfect prediction line
    if len(y_test) > 0 and len(y_pred) > 0:
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    else:
        mn, mx = 0, 10
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", line=dict(color="#ef4444", dash="dash", width=2), name="Perfect Prediction"))
    fig.update_layout(title="Predicted vs Actual Yield", xaxis_title="Actual Yield (t/ha)", yaxis_title="Predicted Yield (t/ha)", template="plotly_white", height=450, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def soil_nutrients_vs_yield(df: pd.DataFrame):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Nitrogen vs Yield", "Phosphorus vs Yield", "Potassium vs Yield"))
    colors = ["#6366f1", "#10b981", "#f59e0b"]
    for i, (col, color) in enumerate(zip(["Nitrogen", "Phosphorus", "Potassium"], colors)):
        if col in df.columns and "Yield_ton_per_ha" in df.columns:
            fig.add_trace(go.Scatter(x=df[col], y=df["Yield_ton_per_ha"], mode="markers", marker=dict(color=color, opacity=0.4, size=4), name=col), row=1, col=i + 1)
    fig.update_layout(title="Soil Nutrients vs Yield", height=380, template="plotly_white", showlegend=False, margin=dict(l=10, r=10, t=60, b=10))
    fig.update_xaxes(title_text="kg/ha")
    fig.update_yaxes(title_text="Yield (t/ha)", col=1)
    return fig

def yield_distribution_by_crop(df: pd.DataFrame):
    fig = px.box(df, x="Crop_Type", y="Yield_ton_per_ha", color="Crop_Type", title="Yield Distribution by Crop Type", labels={"Crop_Type": "Crop", "Yield_ton_per_ha": "Yield (t/ha)"}, template="plotly_white", color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig
