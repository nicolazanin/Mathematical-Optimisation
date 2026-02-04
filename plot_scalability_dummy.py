import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "exact_method": "#1f77b4",  # Blue
    "kn_1": "#ff7f0e",  # Orange
    "kn_3": "#2ca02c",  # Green
    "kn_3_l": "#d62728"  # Red
}

results = pickle.load(open('analysis_scalability_v2.pkl', 'rb'))

x_axis = results["N"]
fig = make_subplots(
    rows=1, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.15,
)

for method in ["exact_method", "kn_1", "kn_3", "kn_3_l"]:
    if method not in results: continue

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=results[method]["t"],
            mode='lines+markers',
            name=f"{method} (Time)",
            line=dict(color=COLORS.get(method, "gray"), width=2),
            marker=dict(size=6),
            legendgroup=method
        ),
        row=1, col=1
    )

fig.update_layout(
    title_text="<b>EACN-REG Scalability Analysis (Execution Time Comparison)</b>",
    title_x=0.5,
    template="plotly_white",
    hovermode="x unified",
)

fig.update_xaxes(title_text="Number of Airports (N)", row=2, col=1)
fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
fig.update_yaxes(title_text="Gap (%)", row=2, col=1)


fig.show()