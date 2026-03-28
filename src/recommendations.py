"""
Rule-Based Recommendation Engine
Generates actionable farming recommendations based on input conditions.
"""


# ── Thresholds ─────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "Nitrogen":    {"low": 40,  "unit": "kg/ha"},
    "Phosphorus":  {"low": 30,  "unit": "kg/ha"},
    "Potassium":   {"low": 30,  "unit": "kg/ha"},
    "pH_low":      5.5,
    "pH_high":     7.5,
    "Rainfall_mm": {"low": 600, "unit": "mm"},
    "Humidity_pct": {"high": 85, "unit": "%"},
}

# Median yields per crop (approx, for "try alternative crop" logic)
CROP_MEDIAN_YIELD = {
    "Rice": 4.2, "Wheat": 3.2, "Maize": 4.8,
    "Cotton": 1.9, "Sugarcane": 65.0, "Soybean": 2.3,
}

ALTERNATIVE_CROPS = {
    "Rice":      "Maize (higher yield potential in similar conditions)",
    "Wheat":     "Maize or Soybean",
    "Maize":     "Sugarcane (if irrigated) or Soybean",
    "Cotton":    "Soybean or Maize",
    "Sugarcane": "Rice (if waterlogged) or Maize",
    "Soybean":   "Maize or Wheat",
}


def generate_recommendations(input_data: dict, predicted_yield: float) -> list[dict]:
    """
    Generate farming recommendations.

    Parameters
    ----------
    input_data : dict with raw feature values (Nitrogen, Phosphorus, etc.)
    predicted_yield : model's predicted yield (tons/ha)

    Returns
    -------
    list of dicts, each with keys:
        category, severity ("high" | "medium" | "low"), title, action, icon
    """
    recs = []

    # ── Nitrogen ───────────────────────────────────────────────────────────
    if input_data.get("Nitrogen", 999) < THRESHOLDS["Nitrogen"]["low"]:
        recs.append({
            "category": "Soil Nutrient",
            "severity": "high",
            "title": "Low Nitrogen Detected",
            "action": (
                f"Apply nitrogen-rich fertilizer (e.g., Urea @ 50–80 kg/ha). "
                f"Current N level: {input_data['Nitrogen']:.0f} kg/ha "
                f"(threshold: {THRESHOLDS['Nitrogen']['low']} kg/ha)."
            ),
            "icon": "🧪",
        })

    # ── Phosphorus ─────────────────────────────────────────────────────────
    if input_data.get("Phosphorus", 999) < THRESHOLDS["Phosphorus"]["low"]:
        recs.append({
            "category": "Soil Nutrient",
            "severity": "high",
            "title": "Low Phosphorus Detected",
            "action": (
                f"Apply DAP (Diammonium Phosphate) @ 40–60 kg/ha. "
                f"Current P level: {input_data['Phosphorus']:.0f} kg/ha."
            ),
            "icon": "🧪",
        })

    # ── Potassium ──────────────────────────────────────────────────────────
    if input_data.get("Potassium", 999) < THRESHOLDS["Potassium"]["low"]:
        recs.append({
            "category": "Soil Nutrient",
            "severity": "medium",
            "title": "Low Potassium Detected",
            "action": (
                f"Apply MOP (Muriate of Potash) @ 30–50 kg/ha. "
                f"Current K level: {input_data['Potassium']:.0f} kg/ha."
            ),
            "icon": "🧪",
        })

    # ── pH ─────────────────────────────────────────────────────────────────
    ph = input_data.get("pH", 6.5)
    if ph < THRESHOLDS["pH_low"]:
        recs.append({
            "category": "Soil Health",
            "severity": "high",
            "title": "Acidic Soil Detected",
            "action": (
                f"Apply agricultural lime (CaCO₃) @ 2–4 tons/ha to raise pH. "
                f"Current pH: {ph:.2f} (ideal: 5.5–7.5)."
            ),
            "icon": "⚗️",
        })
    elif ph > THRESHOLDS["pH_high"]:
        recs.append({
            "category": "Soil Health",
            "severity": "high",
            "title": "Alkaline Soil Detected",
            "action": (
                f"Apply gypsum (CaSO₄) @ 2–5 tons/ha to lower pH. "
                f"Current pH: {ph:.2f} (ideal: 5.5–7.5)."
            ),
            "icon": "⚗️",
        })

    # ── Rainfall ───────────────────────────────────────────────────────────
    if input_data.get("Rainfall_mm", 9999) < THRESHOLDS["Rainfall_mm"]["low"]:
        recs.append({
            "category": "Irrigation",
            "severity": "high",
            "title": "Low Rainfall — Irrigation Needed",
            "action": (
                f"Set up drip or sprinkler irrigation. "
                f"Current rainfall: {input_data['Rainfall_mm']:.0f} mm "
                f"(threshold: {THRESHOLDS['Rainfall_mm']['low']} mm)."
            ),
            "icon": "💧",
        })

    # ── Humidity ───────────────────────────────────────────────────────────
    if input_data.get("Humidity_pct", 0) > THRESHOLDS["Humidity_pct"]["high"]:
        recs.append({
            "category": "Crop Protection",
            "severity": "medium",
            "title": "High Humidity — Disease Risk",
            "action": (
                f"Monitor for fungal diseases (blast, blight). "
                f"Apply preventive fungicide if needed. "
                f"Current humidity: {input_data['Humidity_pct']:.0f}%."
            ),
            "icon": "🦠",
        })

    # ── Low predicted yield → suggest alternative crop ─────────────────────
    crop = input_data.get("Crop_Type", "")
    median = CROP_MEDIAN_YIELD.get(crop, 0)
    if median > 0 and predicted_yield < median * 0.7:
        alt = ALTERNATIVE_CROPS.get(crop, "a higher-yield crop")
        recs.append({
            "category": "Crop Planning",
            "severity": "medium",
            "title": "Low Yield Predicted — Consider Alternative Crop",
            "action": (
                f"Predicted yield ({predicted_yield:.2f} t/ha) is below 70% of "
                f"the median for {crop} ({median:.1f} t/ha). "
                f"Consider switching to: {alt}."
            ),
            "icon": "🌱",
        })

    # ── If no issues found ─────────────────────────────────────────────────
    if not recs:
        recs.append({
            "category": "General",
            "severity": "low",
            "title": "All Parameters Look Good!",
            "action": "Your soil and weather conditions are within optimal ranges. Continue current practices.",
            "icon": "✅",
        })

    return recs
