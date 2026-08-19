"""
FlowSync — Simulated Traffic Data Generator
Generates a realistic synthetic dataset standing in for CCTV / GPS / IoT feeds.
"""

import numpy as np
import pandas as pd


def _congestion_from_features(vehicle_count, avg_speed, road_capacity,
                               accident, is_peak_hour, previous_congestion):
    """
    Deterministic-ish formula (plus noise) used ONLY to generate believable
    training labels. The Random Forest model in model.py learns from this
    data — it does not see this formula directly.
    """
    volume_ratio = vehicle_count / road_capacity          # >1 = over capacity
    speed_factor = max(0, (40 - avg_speed) / 40)           # slower = worse

    congestion = (
        45 * volume_ratio +
        35 * speed_factor +
        10 * accident +
        5 * is_peak_hour +
        0.15 * previous_congestion
    )
    noise = np.random.normal(0, 4)
    congestion = congestion + noise
    return float(np.clip(congestion, 0, 100))


def generate_dataset(n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    hours = rng.integers(0, 24, n_rows)
    minutes = rng.choice([0, 15, 30, 45], n_rows)

    road_capacity = 500
    is_peak_hour = np.isin(hours, [8, 9, 13, 18, 19]).astype(int)

    # Vehicle counts skew higher during peak hours
    base_vehicles = rng.normal(220, 90, n_rows)
    vehicle_count = base_vehicles + is_peak_hour * rng.normal(150, 40, n_rows)
    vehicle_count = np.clip(vehicle_count, 20, 650).astype(int)

    # Speed drops as vehicle count approaches / exceeds capacity
    volume_ratio = vehicle_count / road_capacity
    avg_speed = 45 - volume_ratio * 28 + rng.normal(0, 3, n_rows)
    avg_speed = np.clip(avg_speed, 4, 50)

    accident = rng.choice([0, 1], n_rows, p=[0.93, 0.07])
    weather = rng.choice(["Clear", "Rain", "Fog"], n_rows, p=[0.75, 0.18, 0.07])

    previous_congestion = rng.uniform(10, 90, n_rows)

    congestion = np.array([
        _congestion_from_features(
            vehicle_count[i], avg_speed[i], road_capacity,
            accident[i], is_peak_hour[i], previous_congestion[i]
        )
        for i in range(n_rows)
    ])

    df = pd.DataFrame({
        "hour": hours,
        "minute": minutes,
        "vehicle_count": vehicle_count,
        "avg_speed": avg_speed.round(1),
        "road_capacity": road_capacity,
        "accident": accident,
        "weather": weather,
        "is_peak_hour": is_peak_hour,
        "previous_congestion": previous_congestion.round(1),
        "congestion": congestion.round(1),
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("traffic_data.csv", index=False)
    print(f"Generated {len(df)} rows -> traffic_data.csv")
    print(df.head())
