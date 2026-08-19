"""
FlowSync — AI Traffic Prediction (Random Forest)
Trains on the simulated dataset and forecasts congestion at +10/+20/+30 min.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from data_generator import generate_dataset

FEATURES = [
    "vehicle_count", "avg_speed", "road_capacity",
    "accident", "is_peak_hour", "previous_congestion", "hour",
]


class TrafficPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42
        )
        self._trained = False

    def train(self, df: pd.DataFrame = None):
        if df is None:
            df = generate_dataset()

        X = df[FEATURES]
        y = df["congestion"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        self._trained = True
        return {"mae": round(mae, 2), "n_train": len(X_train), "n_test": len(X_test)}

    def predict_now(self, vehicle_count, avg_speed, road_capacity,
                     accident, is_peak_hour, previous_congestion, hour):
        row = pd.DataFrame([{
            "vehicle_count": vehicle_count,
            "avg_speed": avg_speed,
            "road_capacity": road_capacity,
            "accident": accident,
            "is_peak_hour": is_peak_hour,
            "previous_congestion": previous_congestion,
            "hour": hour,
        }])
        return float(np.clip(self.model.predict(row)[0], 0, 100))

    def forecast(self, vehicle_count, avg_speed, road_capacity,
                 accident, is_peak_hour, previous_congestion, hour,
                 horizons=(10, 20, 30), growth_rate=0.06):
        """
        Simple rolling forecast: at each horizon step we assume vehicle count
        keeps trending (growth_rate) and speed degrades accordingly, then
        re-run the model. This mimics 'what happens if current trend continues'.
        """
        results = {"now": round(self.predict_now(
            vehicle_count, avg_speed, road_capacity, accident,
            is_peak_hour, previous_congestion, hour), 1)}

        cur_vehicles = vehicle_count
        cur_speed = avg_speed
        cur_prev = results["now"]

        for h in horizons:
            steps = h // 10
            cur_vehicles = vehicle_count * ((1 + growth_rate) ** steps)
            cur_vehicles = min(cur_vehicles, road_capacity * 1.6)
            volume_ratio = cur_vehicles / road_capacity
            cur_speed = max(4, avg_speed - volume_ratio * 6)

            pred = self.predict_now(
                cur_vehicles, cur_speed, road_capacity, accident,
                is_peak_hour, cur_prev, hour
            )
            results[f"+{h}min"] = round(pred, 1)
            cur_prev = pred

        return results


if __name__ == "__main__":
    predictor = TrafficPredictor()
    stats = predictor.train()
    print("Training stats:", stats)

    forecast = predictor.forecast(
        vehicle_count=380, avg_speed=18, road_capacity=500,
        accident=0, is_peak_hour=1, previous_congestion=60, hour=8
    )
    print("Forecast:", forecast)
