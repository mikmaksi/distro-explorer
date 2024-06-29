from dash import Dash, html, dcc, callback, Output, Input, State, ALL
from dash import callback_context as ctx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc

from scipy.stats._continuous_distns import _distn_names as cont_distns
from scipy.stats._discrete_distns import _distn_names as disc_distns
from scipy import stats

# random number generator
rng = np.random.default_rng()


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

PARAM_DEFAULTS = {
    "n": 5,
    "a": 0.25,
    "b": 0.25,
    "s": 0.1,
    "c": 0.1,
    "mu": 1,
}

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(dbc.InputGroupText("distribution"), width={"size": 2}),
                dbc.Col(html.Div(dcc.Dropdown(options=cont_distns + disc_distns, value="norm", id="dropdown-selection"))),
            ], className="g-0",
        ),
        html.Hr(),
        dbc.Stack(
            children=[
                dbc.InputGroup([dbc.InputGroupText("size"), dbc.Input(id="distn-size", value=1000, type="number")]),
                dbc.InputGroup([dbc.InputGroupText("loc"), dbc.Input(id="distn-loc", value=0, type="number")]),
                dbc.InputGroup([dbc.InputGroupText("scale"), dbc.Input(id="distn-scale", value=1, type="number")]),
                dbc.InputGroup([dbc.InputGroupText("nbins"), dbc.Input(id="distn-nbins", value=None, type="number")]),
            ],
            id="general-distn-params",
        ),
        html.Hr(),
        dbc.Stack(children=None, id="distn-params"),
        dcc.Graph(id="graph-content"),
        dbc.Alert(children=None, id="alert-message", color="primary"),
    ],
    fluid=True,
)


@callback(Output("distn-params", "children"), Input("dropdown-selection", "value"))
def add_distn_params_to_layout(distn_name):
    func_gen = getattr(stats, distn_name)
    if func_gen.shapes is not None:
        shapes = func_gen.shapes.split(", ")
        children = [
            dbc.InputGroup(
                [
                    dbc.InputGroupText(shape),
                    dbc.Input(id={"type": "distn-param", "index": shape}, value=PARAM_DEFAULTS[shape], type="number"),
                ]
            )
            for shape in shapes
        ]
        return children


@callback(
    Output("graph-content", "figure"),
    Output("alert-message", "children"),
    Input("distn-size", "value"),
    Input("distn-loc", "value"),
    Input("distn-scale", "value"),
    Input("distn-nbins", "value"),
    Input({"type": "distn-param", "index": ALL}, "value"),
    Input({"type": "distn-param", "index": ALL}, "id"),
    State("dropdown-selection", "value"),
)
def update_graph(distn_size, distn_loc, distn_scale, distn_nbins, distn_param_values, distn_param_ids, distn_name):
    func_kwargs = {}
    if distn_param_values != []:
        for param_id, param_value in zip(distn_param_ids, distn_param_values):
            if param_value is None:
                return go.Figure(), "Not all params selected"
            func_kwargs.setdefault(param_id["index"], param_value)
    func_gen = getattr(stats, distn_name)
    try:
        func = func_gen(loc=distn_loc, scale=distn_scale, **func_kwargs)
        data_sample = func.rvs(size=distn_size, random_state=rng)
    except (ValueError, TypeError) as e:
        return go.Figure(), str(e)
    if not isinstance(data_sample, np.ndarray) or len(data_sample) == 0:
        return go.Figure(), "Increase sample size"
    return px.histogram(data_sample, nbins=distn_nbins), None


if __name__ == "__main__":
    app.run(debug=True)
