"""
Synthetic Crop Dataset Generator
Generates ~2000 rows of realistic crop data for model training.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_SAMPLES = 2000

CROP_TYPES = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean"]
SEASONS = ["Kharif", "Rabi", "Zaid"]
IRRIGATION_TYPES = ["Rainfed", "Drip", "Sprinkler", "Flood"]

# Crop-specific parameter ranges  (min, max)
CROP_PROFILES = {
    "Rice":      {"temp": (22, 32), "rain": (800, 2000), "humidity": (60, 90), "base_yield": 4.5},
    "Wheat":     {"temp": (12, 25), "rain": (300, 800),  "humidity": (40, 70), "base_yield": 3.5},
    "Maize":     {"temp": (18, 30), "rain": (500, 1200), "humidity": (50, 80), "base_yield": 5.0},
    "Cotton":    {"temp": (25, 35), "rain": (500, 1000), "humidity": (40, 65), "base_yield": 2.0},
    "Sugarcane": {"temp": (20, 35), "rain": (1000, 2500),"humidity": (60, 85), "base_yield": 70.0},
    "Soybean":   {"temp": (20, 30), "rain": (450, 900),  "humidity": (50, 75), "base_yield": 2.5},
}


def _generate_row(crop: str) -> dict:
    """Generate a single realistic data row for a given crop."""
    profile = CROP_PROFILES[crop]

    temperature = np.random.uniform(*profile["temp"])
    rainfall    = np.random.uniform(*profile["rain"])
    humidity    = np.random.uniform(*profile["humidity"])

    nitrogen   = np.random.uniform(10, 140)
    phosphorus = np.random.uniform(5, 100)
    potassium  = np.random.uniform(5, 110)
    ph         = np.random.uniform(4.5, 8.5)
    area       = np.random.uniform(0.5, 20.0)

    season          = np.random.choice(SEASONS)
    irrigation_type = np.random.choice(IRRIGATION_TYPES)

    # ── Yield formula (realistic, non-linear) ─────────────────────────────
    npk_factor   = 0.005 * (nitrogen + phosphorus + potassium)
    ph_factor    = 1.0 - 0.15 * abs(ph - 6.5)
    rain_factor  = 1.0 - 0.0005 * abs(rainfall - np.mean(profile["rain"]))
    temp_factor  = 1.0 - 0.02 * abs(temperature - np.mean(profile["temp"]))
    irrig_bonus  = {"Drip": 1.08, "Sprinkler": 1.05, "Flood": 1.0, "Rainfed": 0.92}

    base = profile["base_yield"]
    yield_val = (
        base
        * (0.6 + npk_factor)
        * ph_factor
        * rain_factor
        * temp_factor
        * irrig_bonus[irrigation_type]
    )
    # Add Gaussian noise (±8 %)
    yield_val *= np.random.normal(1.0, 0.08)
    yield_val = max(yield_val, 0.1)  # no negative yields

    return {
        "Crop_Type":       crop,
        "Rainfall_mm":     round(rainfall, 1),
        "Temperature_C":   round(temperature, 1),
        "Humidity_pct":     round(humidity, 1),
        "Nitrogen":        round(nitrogen, 1),
        "Phosphorus":      round(phosphorus, 1),
        "Potassium":       round(potassium, 1),
        "pH":              round(ph, 2),
        "Area_ha":         round(area, 2),
        "Season":          season,
        "Irrigation_Type": irrigation_type,
        "Yield_ton_per_ha": round(yield_val, 2),
    }


def generate_dataset(n: int = NUM_SAMPLES) -> pd.DataFrame:
    """Generate full synthetic crop dataset."""
    rows = []
    crops = np.random.choice(CROP_TYPES, size=n)
    for crop in crops:
        rows.append(_generate_row(crop))

    # Introduce ~2 % missing values at random
    df = pd.DataFrame(rows)
    mask = np.random.random(df.shape) < 0.02
    # Only apply missingness to numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df.loc[mask[:, df.columns.get_loc(col)], col] = np.nan

    return df


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "crop_data.csv")

    df = generate_dataset()
    df.to_csv(output_path, index=False)
    print(f"✅  Dataset generated: {output_path}")
    print(f"    Shape : {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Missing values:\n{df.isnull().sum()}")
