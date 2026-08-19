import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FlowSync AI",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .stMetric {
        background-color: #111827;
        border-radius: 10px;
        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.title("🚦 FlowSync AI")

st.caption(
    "Intelligent Mobility Command Center — Predict. Optimize. Move."
)

# Accident input
accident_label = st.sidebar.selectbox(
    "Accident reported?",
    ["No", "Yes"]
)

accident = 1 if accident_label == "Yes" else 0


# ============================================================
# LIVE INCIDENT SIMULATION
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Live Incident Simulation")

incident = st.sidebar.selectbox(
    "Simulate an incident",
    [
        "None",
        "Accident at J3",
        "Road blockage at J5",
        "Event traffic at J2",
        "Transit surge at J6"
    ]
)



# ============================================================
# SIMULATED DATA / MODEL
# ============================================================

@st.cache_data
def create_training_data(n=2400):

    rng = np.random.default_rng(42)

    hour = rng.integers(0, 24, n)
    vehicle_count = rng.integers(20, 650, n)
    avg_speed = rng.uniform(4, 50, n)
    road_capacity = rng.integers(250, 800, n)
    accident = rng.integers(0, 2, n)
    previous_congestion = rng.uniform(5, 95, n)

    peak = (
        ((hour >= 7) & (hour <= 10))
        | ((hour >= 17) & (hour <= 20))
    ).astype(int)

    base = (
        0.35 * (vehicle_count / road_capacity) * 100
        + 0.25 * ((50 - avg_speed) / 46) * 100
        + 0.18 * previous_congestion
        + 10 * peak
        + 12 * accident
    )

    noise = rng.normal(0, 4, n)

    congestion = np.clip(
        base + noise,
        0,
        100
    )

    df = pd.DataFrame({
        "hour": hour,
        "vehicle_count": vehicle_count,
        "avg_speed": avg_speed,
        "road_capacity": road_capacity,
        "accident": accident,
        "previous_congestion": previous_congestion,
        "congestion": congestion
    })

    return df


@st.cache_resource
def train_model():

    df = create_training_data()

    features = [
        "hour",
        "vehicle_count",
        "avg_speed",
        "road_capacity",
        "accident",
        "previous_congestion"
    ]

    X = df[features]
    y = df["congestion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    return model, mae


model, model_mae = train_model()


# ============================================================
# SIDEBAR — SIMULATED TRAFFIC INPUT
# ============================================================

st.sidebar.header("📡 Simulated Traffic Input")

hour = st.sidebar.slider(
    "Hour of day",
    min_value=0,
    max_value=23,
    value=9
)

vehicle_count = st.sidebar.slider(
    "Vehicle count",
    min_value=20,
    max_value=650,
    value=380
)

avg_speed = st.sidebar.slider(
    "Average speed (km/h)",
    min_value=4,
    max_value=50,
    value=18
)

road_capacity = st.sidebar.number_input(
    "Road capacity",
    min_value=100,
    max_value=1000,
    value=500,
    step=50
)

previous_congestion = st.sidebar.slider(
    "Previous congestion (%)",
    min_value=0,
    max_value=100,
    value=60
)

# ============================================================
# DIRECTIONAL TRAFFIC
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🚦 Junction Traffic")

north = st.sidebar.slider(
    "North",
    0,
    500,
    350
)

south = st.sidebar.slider(
    "South",
    0,
    500,
    120
)

east = st.sidebar.slider(
    "East",
    0,
    500,
    80
)

west = st.sidebar.slider(
    "West",
    0,
    500,
    260
)

direction_traffic = {
    "North": north,
    "South": south,
    "East": east,
    "West": west
}


# ============================================================
# CONGESTION PREDICTION
# ============================================================

features = pd.DataFrame([{
    "hour": hour,
    "vehicle_count": vehicle_count,
    "avg_speed": avg_speed,
    "road_capacity": road_capacity,
    "accident": accident,
    "previous_congestion": previous_congestion
}])

current_prediction = float(
    model.predict(features)[0]
)

# Forecast assumptions
future_10 = min(
    current_prediction + max(0, (vehicle_count - 300) / 150),
    100
)

future_20 = min(
    current_prediction + max(0, (vehicle_count - 250) / 90),
    100
)

future_30 = min(
    current_prediction + max(0, (vehicle_count - 200) / 55),
    100
)

# The ML model predicts the base corridor congestion. The live incident
# layer then adjusts the operational forecast for the selected disruption.
incident_forecast_boost = {
    "None": 0.0,
    "Accident at J3": 8.0,
    "Road blockage at J5": 12.0,
    "Event traffic at J2": 7.0,
    "Transit surge at J6": 5.0,
}.get(incident, 0.0)

forecast = {
    "now": round(min(100, current_prediction + incident_forecast_boost), 1),
    "+10min": round(min(100, future_10 + incident_forecast_boost), 1),
    "+20min": round(min(100, future_20 + incident_forecast_boost), 1),
    "+30min": round(min(100, future_30 + incident_forecast_boost), 1)
}

# ============================================================
# SYSTEM STATUS
# ============================================================

if incident != "None":
    st.error(
        f"🔴 FLOWSYNC ACTIVE RESPONSE MODE  |  {incident}"
    )
elif forecast["+30min"] >= 80:
    st.warning(
        "🟠 FLOWSYNC ALERT MODE  |  Congestion expected to rise"
    )
else:
    st.success(
        "🟢 FLOWSYNC MONITORING MODE  |  Network operating normally"
    )

# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3 = st.columns(3)

c1.metric(
    "Current Congestion",
    f"{forecast['now']:.1f}%"
)

c2.metric(
    "Predicted +20 min",
    f"{forecast['+20min']:.1f}%",
    delta=f"{forecast['+20min'] - forecast['now']:.1f} pts"
)

c3.metric(
    "Predicted +30 min",
    f"{forecast['+30min']:.1f}%",
    delta=f"{forecast['+30min'] - forecast['now']:.1f} pts"
)


# ============================================================
# ALERT
# ============================================================

if forecast["+30min"] >= 80:

    st.warning(
        f"⚠️ Congestion predicted to reach "
        f"{forecast['+30min']:.1f}% within 30 minutes. "
        "FlowSync recommends proactive intervention."
    )

elif forecast["+30min"] >= 65:

    st.info(
        f"⚠️ Congestion is trending upward and may reach "
        f"{forecast['+30min']:.1f}% within 30 minutes."
    )


# ============================================================
# TRAFFIC PREDICTION CHART
# ============================================================

st.subheader("📈 AI Traffic Prediction")

fig_forecast = go.Figure()

fig_forecast.add_trace(
    go.Scatter(
        x=["Now", "+10 min", "+20 min", "+30 min"],
        y=[
            forecast["now"],
            forecast["+10min"],
            forecast["+20min"],
            forecast["+30min"]
        ],
        mode="lines+markers",
        line=dict(width=4),
        marker=dict(size=9),
        name="Predicted congestion"
    )
)

fig_forecast.update_layout(
    height=350,
    yaxis_title="Congestion %",
    xaxis_title="Time",
    margin=dict(t=20, b=20)
)

st.plotly_chart(
    fig_forecast,
    use_container_width=True
)


# ============================================================
# 7-JUNCTION NETWORK
# ============================================================

def get_network(vehicle_count, avg_speed, accident):

    G = nx.DiGraph()

    # Base road configuration
    roads = [

        # Main / high-demand corridor
        ("J1", "J2", 2.0, 0.55),
        ("J2", "J3", 2.5, 0.72),
        ("J3", "J5", 2.2, 0.68),

        # Alternative corridor
        ("J1", "J4", 1.8, 0.30),
        ("J4", "J5", 2.0, 0.42),

        # Lower corridor
        ("J4", "J6", 2.2, 0.25),
        ("J5", "J6", 2.0, 0.52),

        # Exit corridor
        ("J6", "J7", 2.5, 0.35)
    ]

    # Scale congestion based on current traffic
    traffic_factor = vehicle_count / 380

    speed_factor = max(
        0.7,
        min(1.4, 25 / max(avg_speed, 4))
    )

    for u, v, distance, base_congestion in roads:

        congestion = (
            base_congestion
            * (0.75 + 0.25 * traffic_factor)
            * (0.85 + 0.15 * speed_factor)
        )

        if accident and (u, v) in [
            ("J2", "J3"),
            ("J3", "J5")
        ]:
            congestion += 0.15

        congestion = min(
            max(congestion, 0.05),
            0.98
        )

        G.add_edge(
            u,
            v,
            distance=distance,
            congestion=congestion
        )

    return G


G = get_network(
    vehicle_count,
    avg_speed,
    accident
)

# ============================================================
# LIVE INCIDENT EFFECT
# ============================================================

if incident == "Accident at J3":

    if G.has_edge("J2", "J3"):
        G["J2"]["J3"]["congestion"] = min(
            G["J2"]["J3"]["congestion"] + 0.35,
            0.98
        )

    if G.has_edge("J3", "J5"):
        G["J3"]["J5"]["congestion"] = min(
            G["J3"]["J5"]["congestion"] + 0.35,
            0.98
        )

elif incident == "Road blockage at J5":

    if G.has_edge("J3", "J5"):
        G["J3"]["J5"]["congestion"] = 0.98

    if G.has_edge("J5", "J6"):
        G["J5"]["J6"]["congestion"] = 0.98

elif incident == "Event traffic at J2":

    if G.has_edge("J1", "J2"):
        G["J1"]["J2"]["congestion"] = min(
            G["J1"]["J2"]["congestion"] + 0.30,
            0.98
        )

    if G.has_edge("J2", "J3"):
        G["J2"]["J3"]["congestion"] = min(
            G["J2"]["J3"]["congestion"] + 0.30,
            0.98
        )

elif incident == "Transit surge at J6":

    if G.has_edge("J4", "J6"):
        G["J4"]["J6"]["congestion"] = min(
            G["J4"]["J6"]["congestion"] + 0.25,
            0.98
        )

    if G.has_edge("J5", "J6"):
        G["J5"]["J6"]["congestion"] = min(
            G["J5"]["J6"]["congestion"] + 0.25,
            0.98
        )

# ============================================================
# ROUTE FUNCTIONS
# ============================================================

def shortest_by_distance_only(
    graph,
    source="J1",
    target="J7"
):

    path = nx.shortest_path(
        graph,
        source=source,
        target=target,
        weight="distance"
    )

    return path


def route_score(graph, path):

    total_distance = 0
    total_congestion = 0

    for i in range(len(path) - 1):

        u = path[i]
        v = path[i + 1]

        edge = graph[u][v]

        total_distance += edge["distance"]
        total_congestion += edge["congestion"]

    return (
        total_distance,
        total_congestion
    )


def find_best_route(
    graph,
    source="J1",
    target="J7"
):

    all_paths = list(
        nx.all_simple_paths(
            graph,
            source=source,
            target=target
        )
    )

    best_path = None
    best_score = float("inf")

    for path in all_paths:

        distance, congestion = route_score(
            graph,
            path
        )

        # Balanced mobility score
        score = (
            0.40 * distance
            + 8.0 * congestion
        )

        if score < best_score:

            best_score = score
            best_path = path

    return {
        "path": best_path,
        "score": best_score
    }


naive_path = shortest_by_distance_only(
    G,
    "J1",
    "J7"
)

smart = find_best_route(
    G,
    "J1",
    "J7"
)

smart_path = smart["path"]

# ============================================================
# INCIDENT ALERT
# ============================================================

if incident != "None":

    st.error(
        f"🚨 ACTIVE INCIDENT: {incident}"
    )

    st.warning(
        "FlowSync has detected a disruption and is recalculating "
        "signal timing, route selection and network congestion."
    )

# ============================================================
# NETWORK STATUS
# ============================================================

st.markdown("### 🌐 Network Status")

critical_roads = []

for u, v, data in G.edges(data=True):

    congestion = data["congestion"] * 100

    if congestion >= 70:

        critical_roads.append(
            f"{u} → {v}"
        )

s1, s2, s3, s4 = st.columns(4)

s1.metric(
    "🚦 Junctions",
    "7"
)

s2.metric(
    "🛣️ Active Roads",
    str(len(G.edges))
)

s3.metric(
    "🔴 Critical Roads",
    str(len(critical_roads))
)

if incident == "None":

    network_state = "NORMAL"

elif len(critical_roads) >= 2:

    network_state = "CRITICAL"

else:

    network_state = "DISRUPTED"

s4.metric(
    "Network State",
    network_state
)

if critical_roads:

    st.warning(
        "Critical corridors: "
        + ", ".join(critical_roads)
    )

else:

    st.success(
        "No critical road congestion detected."
    )

# ============================================================
# AI ACTION PLAN
# ============================================================

st.markdown("### 🎯 FlowSync Action Plan")

if incident != "None":

    actions = [
        f"🚨 Respond to {incident}",
        "🚦 Rebalance signal timing around affected junction",
        f"🗺️ Recalculate route from J1 to J7",
        "🚌 Encourage lower-emission alternatives",
        "📈 Monitor predicted congestion for the next 30 minutes"
    ]

elif forecast["+30min"] >= 80:

    actions = [
        "⚠️ Prepare for critical congestion",
        "🚦 Apply adaptive signal timing",
        "🗺️ Redirect traffic toward lower-congestion roads",
        "🚌 Promote public/shared transport",
        "📈 Continue 30-minute congestion monitoring"
    ]

elif forecast["+30min"] >= 65:

    actions = [
        "⚠️ Monitor rising congestion",
        "🚦 Adjust signal timing",
        "🗺️ Recommend alternate route where beneficial",
        "🚌 Promote shared transport",
        "📈 Re-evaluate network conditions continuously"
    ]

else:

    actions = [
        "✅ Maintain current traffic strategy",
        "🚦 Continue adaptive signal monitoring",
        "🗺️ Keep optimal route active",
        "🚌 Monitor public transport demand",
        "📈 Continue congestion forecasting"
    ]

for action in actions:

    st.markdown(
        f"- {action}"
    )

# ============================================================
# ROUTE INFORMATION
# ============================================================

def route_information(graph, path):

    distance = 0
    congestion = []

    for i in range(len(path) - 1):

        edge = graph[path[i]][path[i + 1]]

        distance += edge["distance"]
        congestion.append(
            edge["congestion"]
        )

    avg_congestion = (
        np.mean(congestion) * 100
        if congestion
        else 0
    )

    # Simple simulated travel-time model
    travel_time = (
        distance / max(avg_speed, 5)
    ) * 60

    return (
        distance,
        travel_time,
        avg_congestion
    )


naive_distance, naive_time, naive_congestion = (
    route_information(
        G,
        naive_path
    )
)

smart_distance, smart_time, smart_congestion = (
    route_information(
        G,
        smart_path
    )
)


# ============================================================
# SIGNAL OPTIMIZER
# ============================================================

st.markdown("---")

sig_col, route_col = st.columns(2)


with sig_col:

    st.subheader(
        "🚦 Adaptive Signal Optimizer"
    )

    total_directional = sum(
        direction_traffic.values()
    )

    fixed_time = 30

    adaptive_times = {}

    for direction, traffic in direction_traffic.items():

        share = (
            traffic / total_directional
            if total_directional > 0
            else 0.25
        )

        adaptive_times[direction] = round(
            max(10, min(55, 10 + share * 90)),
            1
        )

    fig_signal = go.Figure()

    fig_signal.add_trace(
        go.Bar(
            name="Fixed-time (traditional)",
            x=list(direction_traffic.keys()),
            y=[fixed_time] * 4
        )
    )

    fig_signal.add_trace(
        go.Bar(
            name="FlowSync (adaptive)",
            x=list(direction_traffic.keys()),
            y=list(adaptive_times.values())
        )
    )

    fig_signal.update_layout(
        barmode="group",
        height=350,
        yaxis_title="Green time (sec)",
        margin=dict(t=20)
    )

    st.plotly_chart(
        fig_signal,
        use_container_width=True
    )


with route_col:

    st.subheader(
        "🗺️ Smart Route Recommendation"
    )

    st.markdown(
        f"""
        **Shortest-distance route:**  
        `{" → ".join(naive_path)}`

        • {naive_distance:.1f} km  
        • {naive_time:.0f} min  
        • avg congestion {naive_congestion:.0f}%
        """
    )

    st.markdown(
        f"""
        **FlowSync recommended route:**  
        `{" → ".join(smart_path)}`

        • {smart_distance:.1f} km  
        • {smart_time:.0f} min  
        • avg congestion {smart_congestion:.0f}%
        """
    )

    if smart_path != naive_path:

        st.success(
            "FlowSync avoided the higher-congestion corridor "
            "even though it may not be the shortest by distance."
        )

    else:

        st.info(
            "Current conditions do not justify a major route diversion."
        )


# ============================================================
# LIVE MOBILITY NETWORK
# ============================================================

st.markdown("---")

st.subheader(
    "🗺️ Live Mobility Network"
)

st.caption(
    "Simulated urban corridor showing seven signalized junctions, "
    "congestion levels and the FlowSync recommended route."
)


# Fixed visual layout
network_pos = {

    "J1": (0, 0),

    "J2": (2, 2),

    "J3": (4, 2),

    "J4": (2, 0),

    "J5": (4, 0),

    "J6": (3, -2),

    "J7": (3, -4)
}


fig_network = go.Figure()


# ============================================================
# DRAW ROADS
# ============================================================

for u, v, data in G.edges(data=True):

    congestion = float(
        data["congestion"]
    )

    if congestion >= 0.70:

        road_color = "#ef4444"
        road_width = 8

    elif congestion >= 0.40:

        road_color = "#f59e0b"
        road_width = 7

    else:

        road_color = "#22c55e"
        road_width = 6

    x1, y1 = network_pos[u]
    x2, y2 = network_pos[v]

    fig_network.add_trace(
        go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode="lines",
            line=dict(
                color=road_color,
                width=road_width
            ),
            hovertemplate=(
                f"<b>{u} → {v}</b><br>"
                f"Congestion: {congestion * 100:.0f}%"
                "<extra></extra>"
            ),
            showlegend=False
        )
    )


# ============================================================
# HIGHLIGHT FLOWSYNC ROUTE
# ============================================================

for i in range(
    len(smart_path) - 1
):

    u = smart_path[i]
    v = smart_path[i + 1]

    x1, y1 = network_pos[u]
    x2, y2 = network_pos[v]

    fig_network.add_trace(
        go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode="lines",
            line=dict(
                color="#38bdf8",
                width=12
            ),
            hovertemplate=(
                f"⭐ <b>FlowSync Route</b><br>"
                f"{u} → {v}"
                "<extra></extra>"
            ),
            showlegend=False
        )
    )


# ============================================================
# VEHICLES
# ============================================================

np.random.seed(42)

display_vehicle_count = min(
    max(25, int(vehicle_count * 0.15)),
    90
)

edges = list(
    G.edges(data=True)
)

vehicle_x = []
vehicle_y = []
vehicle_hover = []

weights = [
    max(edge[2]["congestion"], 0.05)
    for edge in edges
]

for _ in range(
    display_vehicle_count
):

    edge_index = np.random.choice(
        len(edges),
        p=np.array(weights) / sum(weights)
    )

    u, v, data = edges[
        edge_index
    ]

    x1, y1 = network_pos[u]
    x2, y2 = network_pos[v]

    position = np.random.uniform(
        0.08,
        0.92
    )

    x = (
        x1
        + (x2 - x1) * position
    )

    y = (
        y1
        + (y2 - y1) * position
    )

    vehicle_x.append(x)
    vehicle_y.append(y)

    vehicle_hover.append(
        f"🚗 Vehicle<br>"
        f"Road: {u} → {v}<br>"
        f"Road congestion: "
        f"{data['congestion'] * 100:.0f}%"
    )


fig_network.add_trace(
    go.Scatter(
        x=vehicle_x,
        y=vehicle_y,
        mode="markers",
        marker=dict(
            size=7,
            symbol="circle"
        ),
        hovertext=vehicle_hover,
        hoverinfo="text",
        name="🚗 Vehicles"
    )
)


# ============================================================
# JUNCTIONS
# ============================================================

node_x = []
node_y = []
node_labels = []

for node, (x, y) in network_pos.items():

    node_x.append(x)
    node_y.append(y)

    node_labels.append(
        f"🚦 {node}"
    )


fig_network.add_trace(
    go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        marker=dict(
            size=38,
            color="#111827",
            line=dict(
                color="white",
                width=2
            )
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Signalized junction"
            "<extra></extra>"
        ),
        showlegend=False
    )
)


# ============================================================
# NETWORK LEGEND
# ============================================================

fig_network.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(
            color="#ef4444",
            width=7
        ),
        name="🔴 High congestion"
    )
)

fig_network.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(
            color="#f59e0b",
            width=7
        ),
        name="🟡 Medium congestion"
    )
)

fig_network.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(
            color="#22c55e",
            width=7
        ),
        name="🟢 Low congestion"
    )
)

fig_network.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(
            color="#38bdf8",
            width=10
        ),
        name="⭐ FlowSync route"
    )
)


# ============================================================
# NETWORK LAYOUT
# ============================================================

fig_network.update_layout(

    height=520,

    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20
    ),

    xaxis=dict(
        visible=False,
        range=[-1, 5]
    ),

    yaxis=dict(
        visible=False,
        range=[-5, 3],
        scaleanchor="x",
        scaleratio=1
    ),

    plot_bgcolor="rgba(0,0,0,0)",

    paper_bgcolor="rgba(0,0,0,0)",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)


st.plotly_chart(
    fig_network,
    use_container_width=True
)


# ============================================================
# NETWORK SUMMARY
# ============================================================

n1, n2, n3, n4 = st.columns(4)

n1.metric(
    "🚦 Active Junctions",
    "7"
)

n2.metric(
    "🚗 Vehicles in Network",
    f"{vehicle_count}"
)

n3.metric(
    "🔴 Critical Roads",
    str(
        sum(
            1
            for _, _, data in G.edges(data=True)
            if data["congestion"] >= 0.70
        )
    )
)

n4.metric(
    "⭐ FlowSync Route",
    " → ".join(smart_path)
)

st.caption(
    f"🚗 {vehicle_count} vehicles are represented by "
    f"{display_vehicle_count} visual markers for readability."
)

# ============================================================
# AI DECISION CENTER
# ============================================================

st.markdown("---")

st.subheader("🤖 FlowSync AI Decision Center")

# Find most congested road
most_congested_edge = max(
    G.edges(data=True),
    key=lambda x: x[2]["congestion"]
)

critical_from = most_congested_edge[0]
critical_to = most_congested_edge[1]
critical_congestion = (
    most_congested_edge[2]["congestion"] * 100
)

# Find least congested alternative
least_congested_edge = min(
    G.edges(data=True),
    key=lambda x: x[2]["congestion"]
)

alternative_from = least_congested_edge[0]
alternative_to = least_congested_edge[1]

# Decide intervention
if incident != "None":

    signal_action = (
        f"Emergency adaptive signal response for {incident}"
    )

    route_action = (
        "Redirect traffic away from the affected corridor"
    )

    decision_priority = "INCIDENT"

elif forecast["+30min"] >= 80:

    signal_action = (
        "Increase green time at critical junction"
    )

    route_action = (
        "Redirect traffic to lower-congestion corridor"
    )

    decision_priority = "CRITICAL"

elif forecast["+30min"] >= 65:

    signal_action = (
        "Apply adaptive signal timing"
    )

    route_action = (
        "Recommend alternate route"
    )

    decision_priority = "HIGH"

else:

    signal_action = (
        "Maintain adaptive monitoring"
    )

    route_action = (
        "Keep current optimal route"
    )

    decision_priority = "NORMAL"


# ------------------------------------------------------------
# Decision cards
# ------------------------------------------------------------

d1, d2, d3 = st.columns(3)

with d1:

    st.markdown("### 🚨 Network Risk")

    st.metric(
        "Critical road",
        f"{critical_from} → {critical_to}"
    )

    st.metric(
        "Congestion",
        f"{critical_congestion:.0f}%"
    )


with d2:

    st.markdown("### 🚦 Signal Action")

    st.info(signal_action)

    st.caption(
        f"Priority: **{decision_priority}**"
    )


with d3:

    st.markdown("### 🗺️ Route Action")

    st.success(route_action)

    st.caption(
        f"Alternative corridor detected near "
        f"{alternative_from} → {alternative_to}"
    )


# ------------------------------------------------------------
# Explainability
# ------------------------------------------------------------

st.markdown("### 🧠 Why did FlowSync make this decision?")

reasons = [
    f"Junction {critical_from} → {critical_to} currently has "
    f"the highest simulated congestion ({critical_congestion:.0f}%).",

    f"The model predicts network congestion will reach "
    f"{forecast['+30min']:.1f}% within 30 minutes.",

    f"FlowSync compared multiple network paths instead of "
    f"selecting a route only by distance.",

    f"The selected intervention prioritizes travel time, "
    f"congestion reduction and estimated emissions."
]

for i, reason in enumerate(reasons, 1):

    st.markdown(
        f"**{i}️⃣ {reason}**"
    )


# ------------------------------------------------------------
# Expected impact
# ------------------------------------------------------------

st.markdown("### 🎯 Expected Impact")

e1, e2, e3 = st.columns(3)

e1.metric(
    "Congestion",
    "↓ 23%",
    "Expected"
)

e2.metric(
    "Waiting Time",
    "↓ 35%",
    "Expected"
)

e3.metric(
    "Estimated Emissions",
    "↓ 40%",
    "Expected"
)

st.caption(
    "Impact values are simulated estimates for the hackathon "
    "prototype and are not measured real-world reductions."
)

# ============================================================
# SMART COMMUTER
# ============================================================

st.markdown("---")

st.subheader(
    "🧑‍💼 Smart Commuter"
)

st.caption(
    "FlowSync compares private transport, public transport and carpool "
    "using simulated travel time, congestion exposure and estimated emissions."
)


cc1, cc2, cc3 = st.columns(3)


with cc1:

    st.markdown(
        "**📍 Current trip**"
    )

    st.info(
        "Junction J1 → City Exit J7"
    )


with cc2:

    st.markdown(
        "**🧭 Network condition**"
    )

    st.info(
        "Simulated real-time traffic conditions"
    )


with cc3:

    commuter_priority = st.selectbox(
        "Your priority",
        [
            "Balanced",
            "Fastest journey",
            "Lowest emissions",
            "Lowest congestion"
        ],
        key="commuter_priority"
    )


# ============================================================
# MODE SCORING
# ============================================================

if commuter_priority == "Fastest journey":

    weights = {
        "time": 0.60,
        "congestion": 0.25,
        "emissions": 0.15
    }

elif commuter_priority == "Lowest emissions":

    weights = {
        "time": 0.20,
        "congestion": 0.20,
        "emissions": 0.60
    }

elif commuter_priority == "Lowest congestion":

    weights = {
        "time": 0.20,
        "congestion": 0.60,
        "emissions": 0.20
    }

else:

    weights = {
        "time": 0.40,
        "congestion": 0.35,
        "emissions": 0.25
    }


modes = {

    "🚗 Private car": {
        "time": 20,
        "congestion": forecast["now"],
        "emissions": 90
    },

    "🚌 Public transport": {
        "time": 25,
        "congestion": forecast["now"] * 0.45,
        "emissions": 35
    },

    "🚗 Carpool": {
        "time": 22,
        "congestion": forecast["now"] * 0.65,
        "emissions": 50
    }
}


for mode in modes:

    data = modes[mode]

    data["score"] = (
        weights["time"] * data["time"]
        + weights["congestion"] * data["congestion"]
        + weights["emissions"] * data["emissions"]
    )


best_mode = min(
    modes,
    key=lambda x: modes[x]["score"]
)


# ============================================================
# MODE CARDS
# ============================================================

m1, m2, m3 = st.columns(3)


with m1:

    st.markdown("### 🚗 Private Car")

    st.metric(
        "Travel time",
        f"{modes['🚗 Private car']['time']} min"
    )

    st.metric(
        "Congestion exposure",
        f"{modes['🚗 Private car']['congestion']:.0f}%"
    )

    st.metric(
        "Emission level",
        "High"
    )


with m2:

    st.markdown("### 🚌 Public Transport")

    st.metric(
        "Travel time",
        f"{modes['🚌 Public transport']['time']} min"
    )

    st.metric(
        "Congestion exposure",
        f"{modes['🚌 Public transport']['congestion']:.0f}%"
    )

    st.metric(
        "Emission level",
        "Low"
    )


with m3:

    st.markdown("### 🚗 Carpool")

    st.metric(
        "Travel time",
        f"{modes['🚗 Carpool']['time']} min"
    )

    st.metric(
        "Congestion exposure",
        f"{modes['🚗 Carpool']['congestion']:.0f}%"
    )

    st.metric(
        "Emission level",
        "Medium"
    )


st.caption(
    "🚌 Public transport and carpool values are currently modelled "
    "for the prototype. In deployment, these can be connected to "
    "live transit, GPS and occupancy feeds."
)


# ============================================================
# RECOMMENDATION
# ============================================================

st.markdown(
    "### ⭐ FlowSync Recommendation"
)

st.caption(
    "Recommendation score: travel time 40% + congestion exposure 35% "
    "+ estimated emissions 25%. Weights change according to commuter priority."
)

st.success(
    f"Recommended mode: **{best_mode}** "
    f"for the selected priority: **{commuter_priority}**."
)


# ============================================================
# EMISSION FUNCTIONS
# ============================================================

def raw_emission_score(
    vehicles,
    waiting
):

    return (
        vehicles * 0.18
        + waiting * 0.5
    )


def emission_index(
    vehicles,
    waiting,
    baseline_score
):

    current_score = raw_emission_score(
        vehicles,
        waiting
    )

    if baseline_score <= 0:
        return 100.0

    return round(
        (current_score / baseline_score) * 100,
        1
    )

# ============================================================
# FLOWSYNC IMPACT SCORE
# ============================================================

st.markdown("### 📊 FlowSync Impact Score")

impact_score = 0

if forecast["+30min"] >= 80:
    impact_score += 35
elif forecast["+30min"] >= 65:
    impact_score += 20
else:
    impact_score += 10

if incident != "None":
    impact_score += 30

if len(critical_roads) >= 2:
    impact_score += 20
elif len(critical_roads) == 1:
    impact_score += 10

impact_score = min(impact_score, 100)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Network Risk",
    f"{impact_score}/100"
)

c2.metric(
    "AI Response",
    "ACTIVE" if incident != "None" else "MONITORING"
)

c3.metric(
    "Prediction Horizon",
    "30 min"
)

# ============================================================
# WHAT-IF SIMULATOR
# ============================================================

st.markdown("---")

st.subheader(
    "🚀 What-If Simulator"
)

before_congestion = forecast["now"]

before_speed = avg_speed

before_waiting = round(
    north / 8,
    1
)

before_emission_raw = raw_emission_score(
    vehicle_count,
    before_waiting
)


run = st.button(
    "▶ RUN AI INTERVENTION",
    type="primary"
)


if run:

    # --------------------------------------------------------
    # AI intervention decision
    # --------------------------------------------------------

    st.markdown(
        "### 🤖 FlowSync AI Decision"
    )

    if forecast["+30min"] >= 80:

        st.warning(
            "Critical congestion predicted."
        )

        st.success(
            """
            ✓ Adaptive traffic signal timing  
            ✓ Traffic redistribution  
            ✓ Promote public transport / carpool  
            """
        )

    elif forecast["+30min"] >= 65:

        st.info(
            "Moderate-to-high congestion predicted."
        )

        st.success(
            """
            ✓ Adjust signal timing  
            ✓ Recommend lower-congestion route  
            ✓ Encourage shared/public transport  
            """
        )

    else:

        st.success(
            """
            ✓ Maintain adaptive signals  
            ✓ Monitor network conditions  
            ✓ Recommend current best route  
            """
        )


    # --------------------------------------------------------
    # Simulated intervention impact
    # --------------------------------------------------------

    after_congestion = round(
        before_congestion * 0.66,
        1
    )

    after_speed = round(
        before_speed * 1.5,
        1
    )

    after_waiting = round(
        before_waiting * 0.6,
        1
    )

    after_emission_index = emission_index(
        vehicle_count,
        after_waiting,
        before_emission_raw
    )


    # --------------------------------------------------------
    # Before / After metrics
    # --------------------------------------------------------

    b1, b2, b3, b4 = st.columns(4)


    b1.metric(
        "Congestion",
        f"{after_congestion}%",
        delta=(
            f"{after_congestion - before_congestion:.1f} pts"
        ),
        delta_color="inverse"
    )


    b2.metric(
        "Avg Speed",
        f"{after_speed} km/h",
        delta=(
            f"{after_speed - before_speed:.1f} km/h"
        )
    )


    b3.metric(
        "Waiting Time",
        f"{after_waiting}s",
        delta=(
            f"{after_waiting - before_waiting:.1f}s"
        ),
        delta_color="inverse"
    )


    b4.metric(
        "Emission Index",
        f"{after_emission_index:.1f}",
        delta=(
            f"{after_emission_index - 100:.1f}"
        ),
        delta_color="inverse"
    )


    # --------------------------------------------------------
    # Comparison chart
    # --------------------------------------------------------

    fig_cmp = go.Figure()

    metrics = [
        "Congestion %",
        "Avg Speed (km/h)",
        "Waiting (s)",
        "Emission Index"
    ]

    before_vals = [
        before_congestion,
        before_speed,
        before_waiting,
        100
    ]

    after_vals = [
        after_congestion,
        after_speed,
        after_waiting,
        after_emission_index
    ]


    fig_cmp.add_trace(
        go.Bar(
            name="Before",
            x=metrics,
            y=before_vals
        )
    )


    fig_cmp.add_trace(
        go.Bar(
            name="After",
            x=metrics,
            y=after_vals
        )
    )


    fig_cmp.update_layout(
        barmode="group",
        height=350,
        margin=dict(t=10)
    )


    st.plotly_chart(
        fig_cmp,
        use_container_width=True
    )


    st.success(
        "Simulated intervention reduces congestion, waiting time "
        "and estimated emissions while increasing average speed."
    )


else:

    st.info(
        "Press **RUN AI INTERVENTION** to simulate adaptive signals "
        "+ smart rerouting and compare Before vs After."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.caption(
    f"Model MAE on held-out test set: {model_mae:.2f} points "
    "(trained on 2400 simulated rows)."
)


# ============================================================
# DISCLAIMER
# ============================================================

st.caption(
    "Data shown is simulated for this prototype. Architecture is "
    "designed so simulated inputs can be replaced by CCTV-derived "
    "vehicle counts, GPS, IoT traffic sensors and public transport "
    "feeds. Emission Index is a modelled comparison metric, not "
    "a measured AQI or laboratory CO₂ measurement."
)