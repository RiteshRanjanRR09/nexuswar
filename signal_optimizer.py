"""
FlowSync — Adaptive Traffic Signal Optimizer
Allocates green time per direction proportionally to traffic volume,
subject to a min/max floor/ceiling so no direction is ever starved.
"""

def optimize_signals(direction_traffic: dict, cycle_time: int = 120,
                      min_green: int = 12, max_green: int = 60) -> dict:
    """
    direction_traffic: e.g. {"North": 350, "South": 120, "East": 80, "West": 260}
    Returns green-time (seconds) per direction, summing to ~cycle_time.
    """
    total_traffic = sum(direction_traffic.values()) or 1
    raw_allocation = {
        d: (v / total_traffic) * cycle_time
        for d, v in direction_traffic.items()
    }

    # Clip to [min_green, max_green]
    clipped = {d: max(min_green, min(max_green, t)) for d, t in raw_allocation.items()}

    # Redistribute any remaining time proportionally among non-maxed directions
    diff = cycle_time - sum(clipped.values())
    adjustable = [d for d in clipped if clipped[d] not in (min_green, max_green)]
    if adjustable and abs(diff) > 0.5:
        share = diff / len(adjustable)
        for d in adjustable:
            clipped[d] = max(min_green, min(max_green, clipped[d] + share))

    return {d: round(t) for d, t in clipped.items()}


def baseline_signals(direction_traffic: dict, cycle_time: int = 120) -> dict:
    """Traditional fixed-time signal for comparison (equal split)."""
    n = len(direction_traffic)
    equal_share = round(cycle_time / n)
    return {d: equal_share for d in direction_traffic}


if __name__ == "__main__":
    traffic = {"North": 350, "South": 120, "East": 80, "West": 260}
    print("Baseline (fixed):", baseline_signals(traffic))
    print("FlowSync (adaptive):", optimize_signals(traffic))
