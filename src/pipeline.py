"""
End-to-end ML Pipeline for Crop Yield Prediction
Runs entirely in memory without saving .pkl files.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
import shap
import plotly.graph_objects as go


class CropYieldPipeline:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.explainer = None
        self.feature_names = None
        
        self.categorical_cols = ["Crop_Type", "Season", "Irrigation_Type"]
        self.numeric_cols = [
            "Rainfall_mm", "Temperature_C", "Humidity_pct", 
            "Nitrogen", "Phosphorus", "Potassium", "pH", "Area_ha",
            "Soil_Fertility_Index", "Weather_Index"
        ]

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Soil_Fertility_Index"] = (df["Nitrogen"] + df["Phosphorus"] + df["Potassium"]) / 3.0
        df["Weather_Index"] = (
            df["Rainfall_mm"] * 0.004
            + (100 - (df["Temperature_C"] - 25).abs()) * 0.03
            + df["Humidity_pct"] * 0.03
        )
        return df

    def fit(self, df: pd.DataFrame):
        df = df.copy()

        # 1. Handle missing numeric values
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
                
        # 2. Handle missing categorical values
        for col in self.categorical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
                
        # 3. Feature engineering
        df = self._engineer_features(df)
        
        # 4. Encode categoricals
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Convert to string to avoid mixed type issues
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            
        self.feature_names = self.numeric_cols + self.categorical_cols
        
        # 5. Extract features and scale
        X = df[self.feature_names].values.astype(float)
        X[:, :len(self.numeric_cols)] = self.scaler.fit_transform(X[:, :len(self.numeric_cols)])
        
        # Ensure target exists
        if "Yield_ton_per_ha" not in df.columns:
            raise ValueError("Training data must contain 'Yield_ton_per_ha'.")
            
        y = df["Yield_ton_per_ha"].values
        
        # 6. Train model and setup explainability
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, df: pd.DataFrame):
        """
        Predict on new data.
        Returns: predictions array, feature array X
        """
        df = df.copy()
        
        # 1. Ensure base numeric columns exist and handle missing
        base_numeric = [c for c in self.numeric_cols if c not in ["Soil_Fertility_Index", "Weather_Index"]]
        for col in base_numeric:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
            elif col not in df.columns:
                df[col] = 0.0 # fallback if a column is completely missing
                
        # 2. Ensure base categoricals exist and handle missing
        for col in self.categorical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif col not in df.columns:
                df[col] = "Rice" # fallback
                
        # 3. Feature engineering
        df = self._engineer_features(df)
        
        # 4. Encode categoricals (handle unseen labels safely)
        for col in self.categorical_cols:
            classes = list(self.label_encoders[col].classes_)
            # map unseen to the first known class
            df[col] = df[col].astype(str).map(lambda x: x if x in classes else classes[0])
            df[col] = self.label_encoders[col].transform(df[col])
            
        # 5. Extract features and scale
        X = df[self.feature_names].values.astype(float)
        X[:, :len(self.numeric_cols)] = self.scaler.transform(X[:, :len(self.numeric_cols)])
        
        # 6. Predict
        preds = self.model.predict(X)
        # Ensure no negative predictions
        preds = np.maximum(preds, 0.01)
        
        return preds, X
        
    def explain_prediction(self, X_single: np.ndarray) -> list:
        shap_vals = self.explainer.shap_values(X_single)
        if shap_vals.ndim == 1:
            shap_vals = shap_vals.reshape(1, -1)
            
        vals = shap_vals[0]
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
            
        prediction = base_value + vals.sum()
        explanations = []
        order = np.argsort(-np.abs(vals))
        
        for i in order[:5]:
            impact_pct = abs(vals[i] / prediction) * 100 if prediction != 0 else 0
            direction = "increases" if vals[i] > 0 else "decreases"
            feature_label = self.feature_names[i].replace("_", " ")
            text = f"{feature_label} {direction} predicted yield by {impact_pct:.1f}%"
            explanations.append({
                "feature": self.feature_names[i],
                "shap_value": float(vals[i]),
                "impact_pct": round(impact_pct, 1),
                "direction": direction,
                "text": text,
            })
        return explanations
        
    def generate_global_importance(self, X_sample: np.ndarray) -> go.Figure:
        # Limit to 50 samples for performance
        if len(X_sample) > 50:
            X_sample = X_sample[:50]
            
        shap_vals = self.explainer.shap_values(X_sample)
        if shap_vals.ndim == 1:
            shap_vals = shap_vals.reshape(1, -1)
            
        mean_abs = np.abs(shap_vals).mean(axis=0)
        
        order = np.argsort(mean_abs)
        sorted_names = [self.feature_names[i] for i in order]
        sorted_vals = mean_abs[order]
        
        fig = go.Figure(go.Bar(
            x=sorted_vals, y=sorted_names, orientation="h", marker_color="#6366f1"
        ))
        fig.update_layout(
            title="Feature Importance (Mean |SHAP|)",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Feature",
            template="plotly_white",
            height=450,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig
