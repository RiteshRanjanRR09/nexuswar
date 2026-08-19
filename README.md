# 🚦 FlowSync — Smart Traffic Management System

FlowSync is a smart traffic management and route recommendation system built with Python and Streamlit.

It simulates an urban road network and uses traffic conditions, congestion predictions, route optimization, and adaptive signal timing to recommend better traffic-management decisions.

## 🚀 Features

### 📊 Traffic Prediction
FlowSync predicts future traffic congestion based on the current traffic conditions and user-provided inputs.

It provides congestion predictions for:
- Current conditions
- +15–20 minutes
- +30 minutes

### 🗺️ Smart Route Recommendation
FlowSync compares:
- The shortest-distance route
- The FlowSync optimized route

Instead of considering only distance, the optimized route considers traffic congestion and other mobility factors.

The route optimization uses graph-based pathfinding to select a lower-cost route. :contentReference[oaicite:1]{index=1}

### 🚦 Adaptive Signal Optimizer
The system calculates adaptive green-light timings based on traffic volume.

It compares traditional fixed-time signals with FlowSync's adaptive signal timings. :contentReference[oaicite:2]{index=2}

### 🚨 Live Incident Simulation
Users can simulate traffic incidents such as:
- Accident at J3
- Road blockage at J5
- Event traffic at J2
- Transit surge at J6

When an incident occurs, FlowSync modifies the affected road congestion and recalculates the recommended route.

### 🗺️ Live Mobility Network
The application provides a visual representation of a seven-junction urban road network.

Roads are displayed according to their congestion levels, and the recommended FlowSync route is highlighted. :contentReference[oaicite:3]{index=3}

### 🤖 AI Decision Center
The AI Decision Center analyzes the current network and determines:
- The most congested road
- A lower-congestion alternative
- Recommended signal action
- Recommended route action
- Priority level

When an incident is active, FlowSync switches to an emergency response strategy. :contentReference[oaicite:4]{index=4}

### 🧠 Explainable Decisions
FlowSync explains why a particular intervention was selected by considering:
- Current network congestion
- Predicted future congestion
- Alternative routes
- Travel time
- Congestion reduction
- Estimated emissions

## 🛠️ Technologies Used

- Python
- Streamlit
- NetworkX
- NumPy
- Plotly
- Scikit-learn

## 📁 Project Structure

```text
FlowSync/
│
├── app.py
├── route_optimizer.py
├── README.md
└── requirements.txt