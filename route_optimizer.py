"""
FlowSync — Smart Route Recommendation
Builds a small road-network graph and finds the lowest-COST path using
Dijkstra, where cost = travel_time + traffic_cost + pollution_cost
(not just raw distance).
"""

import networkx as nx


def build_sample_network():
    """
    A small demo road network. Each edge carries:
      distance_km, base_time_min, congestion (0-1), pollution_factor
    """
    G = nx.DiGraph()
    edges = [
        ("A", "B", dict(distance_km=5, base_time_min=10, congestion=0.75, pollution=0.8)),
        ("B", "D", dict(distance_km=4, base_time_min=8,  congestion=0.70, pollution=0.75)),
        ("A", "C", dict(distance_km=6, base_time_min=12, congestion=0.20, pollution=0.25)),
        ("C", "D", dict(distance_km=5, base_time_min=9,  congestion=0.25, pollution=0.30)),
    ]
    for u, v, attrs in edges:
        G.add_edge(u, v, **attrs)
    return G


def compute_edge_cost(attrs, w_time=1.0, w_traffic=8.0, w_pollution=4.0):
    """
    Combined cost = weighted sum of travel time, traffic penalty, pollution penalty.
    Traffic and pollution are 0-1 severity scores scaled up so they meaningfully
    compete with raw travel time in the shortest-path calculation.
    """
    return (
        w_time * attrs["base_time_min"] +
        w_traffic * attrs["congestion"] * attrs["base_time_min"] +
        w_pollution * attrs["pollution"]
    )


def find_best_route(G: nx.DiGraph, source: str, target: str):
    for u, v, attrs in G.edges(data=True):
        attrs["cost"] = compute_edge_cost(attrs)

    path = nx.dijkstra_path(G, source, target, weight="cost")
    total_cost = nx.dijkstra_path_length(G, source, target, weight="cost")

    total_distance = sum(G[u][v]["distance_km"] for u, v in zip(path, path[1:]))
    total_time = sum(G[u][v]["base_time_min"] for u, v in zip(path, path[1:]))
    avg_congestion = sum(G[u][v]["congestion"] for u, v in zip(path, path[1:])) / (len(path) - 1)

    return {
        "path": path,
        "total_cost": round(total_cost, 1),
        "distance_km": total_distance,
        "time_min": total_time,
        "avg_congestion": round(avg_congestion, 2),
    }


def shortest_by_distance_only(G: nx.DiGraph, source: str, target: str):
    """What a naive 'shortest distance' router (e.g. plain Maps) would pick."""
    path = nx.dijkstra_path(G, source, target, weight="distance_km")
    total_distance = sum(G[u][v]["distance_km"] for u, v in zip(path, path[1:]))
    total_time = sum(G[u][v]["base_time_min"] for u, v in zip(path, path[1:]))
    avg_congestion = sum(G[u][v]["congestion"] for u, v in zip(path, path[1:])) / (len(path) - 1)
    return {
        "path": path,
        "distance_km": total_distance,
        "time_min": total_time,
        "avg_congestion": round(avg_congestion, 2),
    }


if __name__ == "__main__":
    G = build_sample_network()
    print("Shortest-distance route:", shortest_by_distance_only(G, "A", "D"))
    print("FlowSync recommended route:", find_best_route(G, "A", "D"))
