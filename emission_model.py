"""
FlowSync — Emission Index Model
NOT a real AQI measurement. A modelled index for comparing scenarios:
    Emission Index = vehicle_count * idle_time_sec * emission_factor
normalized so the "before" scenario reads as 100.
"""

EMISSION_FACTOR = 0.0018  # tunable constant, arbitrary units


def raw_emission_score(vehicle_count: float, idle_time_sec: float,
                        emission_factor: float = EMISSION_FACTOR) -> float:
    return vehicle_count * idle_time_sec * emission_factor


def emission_index(vehicle_count: float, idle_time_sec: float,
                    baseline_score: float = None) -> float:
    """
    Returns an index where `baseline_score` (if given) is treated as 100.
    If no baseline given, returns the raw score directly.
    """
    score = raw_emission_score(vehicle_count, idle_time_sec)
    if baseline_score is None:
        return round(score, 1)
    return round((score / baseline_score) * 100, 1)


if __name__ == "__main__":
    before_raw = raw_emission_score(vehicle_count=380, idle_time_sec=48)
    before_index = emission_index(380, 48, baseline_score=before_raw)  # = 100 by definition
    after_index = emission_index(380, 29, baseline_score=before_raw)

    print(f"Before: idle=48s -> index {before_index}")
    print(f"After:  idle=29s -> index {after_index}")
