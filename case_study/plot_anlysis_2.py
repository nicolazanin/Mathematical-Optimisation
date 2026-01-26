import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

results = pickle.load(open('analysis_2.pkl', 'rb'))

ttt_3_400 = []
for cell in sorted(results[3][400]["active"][3], key=lambda d: min(d['total_times'])):
    ttt_3_400.append([time * 60 for time in cell["total_times"]])
ttt_3_600 = []
for cell in sorted(results[3][600]["active"][3], key=lambda d: min(d['total_times'])):
    ttt_3_600.append([time * 60 for time in cell["total_times"]])
ttt_4_400 = []
for cell in sorted(results[4][400]["active"][3], key=lambda d: min(d['total_times'])):
    ttt_4_400.append([time * 60 for time in cell["total_times"]])
ttt_4_600 = []
for cell in sorted(results[4][600]["active"][3], key=lambda d: min(d['total_times'])):
    ttt_4_600.append([time * 60 for time in cell["total_times"]])

titles = [
    "(a) Curr. Ntw. (r=400 km, TTT=3 hrs)",
    "(b) Curr. Ntw. (r=600 km, TTT=3 hrs)",
    "(c) Curr. Ntw. (r=400 km, TTT=4 hrs)",
    "(d) Curr. Ntw. (r=600 km, TTT=4 hrs)",
    # "(d) Full Ntw. (r=400 km, TTT=3 hrs)",
    # "(e) Full Ntw. (r=400 km, TTT=4 hrs)",
    # "(f) Full Ntw. (r=600 km, TTT=4 hrs)",
]

fig = make_subplots(
    rows=1,
    cols=4,
    subplot_titles=titles,
    horizontal_spacing=0.05,
    vertical_spacing=0.12
)

for i, data in enumerate([ttt_3_400, ttt_3_600, ttt_4_400, ttt_4_600]):
    row = 1
    col = i + 1
    max_len = len(data) - 1  # last x index
    max_time = max(max(times) for times in data)
    for k, times in enumerate(data):
        fig.add_trace(
            go.Scatter(
                x=[k] * len(times),  # same x, multiple y values
                y=times,
                mode="markers",
                marker=dict(size=3, color="black", opacity=0.5),
                showlegend=False
            ),
            row=row,
            col=col
        )
    fig.add_shape(
        type="line",
        x0=max_len,
        x1=max_len,
        y0=0,
        y1=max_time,
        line=dict(color="gray", width=1, dash="dash"),
        row=row,
        col=col
    )

    fig.add_shape(
        type="line",
        x0=0,
        x1=max_len,
        y0=max_time,
        y1=max_time,
        line=dict(color="gray", width=1, dash="dash"),
        row=row,
        col=col
    )
    fig.add_annotation(
        x=max_len,
        y=max_time * 0.5,
        text=f"Max index = {max_len}",
        xshift=20,
        textangle=90,
        row=row,
        col=col
    )

    fig.add_annotation(
        x=max_len * 0.5,
        y=max_time,
        text=f"Max TTT = {max_time:.1f} min",
        showarrow=False,
        yshift=10,
        row=row,
        col=col
    )

fig.update_yaxes(
    title_text="Total Travel Time (TTT) in minutes",
    range=[0, 280],
    showgrid=True
)

fig.update_xaxes(
    title_text="Pop. area ids (sorted by the minimum TTT)",
    showgrid=True
)

fig.update_layout(
    template="plotly_white",
    margin=dict(l=60, r=20, t=80, b=60),
    font=dict(size=12)
)

fig.show()
