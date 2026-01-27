import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

results = pickle.load(open(BASE_DIR /'analysis_1.pkl', 'rb'))
charging_bases = results["max_cells"]["active"][0]

max_cells_curr = results["max_cells"]["active"][1]
max_cells_pop_curr = results["max_cells"]["active"][2]

max_pop_curr = results["max_pop"]["active"][1]
max_pop_pop_curr = results["max_pop"]["active"][2]

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        "(a) Maximize number of cells covered",
        "(b) Maximize number of cells covered (population calculated ex-post)",
        "(c) Maximize population covered",
        "(d) Maximize population covered (population calculated ex-post)"
    ]
)


def add_line(fig, x, y, name, row, col, color, showlegend=False):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=name,
            line=dict(color=color),
            marker=dict(size=6),
            showlegend=showlegend
        ),
        row=row,
        col=col
    )


add_line(fig, charging_bases, max_cells_curr, "Curr Ntw", 1, 1, "firebrick", showlegend=True)
# add_line(fig, charging_bases, cells_full_1, "Full Ntw", 1, 1, "royalblue", showlegend=False)

add_line(fig, charging_bases, max_cells_pop_curr, "Curr Ntw", 1, 2, "firebrick")
# add_line(fig, charging_bases, pop_full_1, "Full Ntw", 1, 2, "royalblue")

add_line(fig, charging_bases, max_pop_curr, "Curr Ntw", 2, 1, "firebrick")
# add_line(fig, charging_bases, cells_full_2, "Full Ntw", 2, 1, "royalblue")

add_line(fig, charging_bases, max_pop_pop_curr, "Curr Ntw", 2, 2, "firebrick")
# add_line(fig, charging_bases, pop_full_2, "Full Ntw", 2, 2, "royalblue")

fig.update_xaxes(title_text="Nr. charging bases")
fig.update_yaxes(title_text="Nr. cells", row=1, col=1)
fig.update_yaxes(title_text="Population (Mln)", row=1, col=2)
fig.update_yaxes(title_text="Nr. cells", row=2, col=1)
fig.update_yaxes(title_text="Population (Mln)", row=2, col=2)

fig.update_layout(
    template="plotly_white",
    legend_title_text="Legend",
    hovermode="x unified"
)

fig.show()
