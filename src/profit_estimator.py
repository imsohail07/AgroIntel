"""
Profit Estimation Module
Calculates revenue, costs, and net profit for the farmer.
"""


# ── Market Prices (INR per ton) ────────────────────────────────────────────────
MARKET_PRICES = {
    "Rice":      22000,
    "Wheat":     25000,
    "Maize":     18000,
    "Cotton":    55000,
    "Sugarcane":  3500,
    "Soybean":   40000,
}

# ── Cost Components (INR per hectare) ──────────────────────────────────────────
BASE_COST_PER_HA = {
    "Rice":      25000,
    "Wheat":     18000,
    "Maize":     15000,
    "Cotton":    22000,
    "Sugarcane": 50000,
    "Soybean":   14000,
}

FERTILIZER_COST_RATE = 120    # INR per kg of nutrient shortfall
IRRIGATION_COST_PER_HA = 5000  # additional cost if irrigation needed


def estimate_profit(
    input_data: dict,
    predicted_yield: float,
    recommendations: list[dict],
) -> dict:
    """
    Estimate profit/loss for the farmer.

    Parameters
    ----------
    input_data : raw feature dict (must include Crop_Type, Area_ha)
    predicted_yield : tons/ha
    recommendations : output from recommendations module

    Returns
    -------
    dict with revenue, costs, profit, and improvement estimate
    """
    crop = input_data.get("Crop_Type", "Rice")
    area = input_data.get("Area_ha", 1.0)

    market_price = MARKET_PRICES.get(crop, 20000)
    base_cost    = BASE_COST_PER_HA.get(crop, 20000)

    # ── Total yield ────────────────────────────────────────────────────────
    total_yield = predicted_yield * area  # tons

    # ── Revenue ────────────────────────────────────────────────────────────
    revenue = total_yield * market_price   # INR

    # ── Costs ──────────────────────────────────────────────────────────────
    cultivation_cost = base_cost * area

    # Extra fertilizer cost if nutrients are low
    fertilizer_cost = 0
    for rec in recommendations:
        if rec["category"] == "Soil Nutrient":
            fertilizer_cost += FERTILIZER_COST_RATE * 50 * area  # ~50 kg/ha correction

    # Irrigation cost
    irrigation_cost = 0
    for rec in recommendations:
        if rec["category"] == "Irrigation":
            irrigation_cost = IRRIGATION_COST_PER_HA * area
            break

    total_cost = cultivation_cost + fertilizer_cost + irrigation_cost

    # ── Profit ─────────────────────────────────────────────────────────────
    net_profit = revenue - total_cost
    profit_per_ha = net_profit / area if area > 0 else 0

    # ── Potential improvement estimate ─────────────────────────────────────
    # Simple heuristic: each high-severity recommendation, if fixed, could
    # improve yield by ~10 %; medium by 5 %.
    improvement_pct = 0
    for rec in recommendations:
        if rec["severity"] == "high":
            improvement_pct += 10
        elif rec["severity"] == "medium":
            improvement_pct += 5
    improvement_pct = min(improvement_pct, 40)  # cap at 40 %

    improved_yield = predicted_yield * (1 + improvement_pct / 100)
    improved_revenue = improved_yield * area * market_price
    improved_cost = total_cost + fertilizer_cost * 0.5 + irrigation_cost * 0.3  # partial
    improved_profit = improved_revenue - improved_cost

    return {
        "crop": crop,
        "area_ha": area,
        "predicted_yield_per_ha": round(predicted_yield, 2),
        "total_yield_tons": round(total_yield, 2),
        "market_price_per_ton": market_price,
        "revenue": round(revenue, 0),
        "cultivation_cost": round(cultivation_cost, 0),
        "fertilizer_cost": round(fertilizer_cost, 0),
        "irrigation_cost": round(irrigation_cost, 0),
        "total_cost": round(total_cost, 0),
        "net_profit": round(net_profit, 0),
        "profit_per_ha": round(profit_per_ha, 0),
        "improvement_pct": improvement_pct,
        "improved_yield_per_ha": round(improved_yield, 2),
        "improved_profit": round(improved_profit, 0),
        "profit_uplift": round(improved_profit - net_profit, 0),
    }
