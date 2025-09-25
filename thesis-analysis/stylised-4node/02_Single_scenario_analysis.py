"""
Experiment: Stylised 4-node network – scenario-based comparison of flow-based domains
and operational outcomes under different capacity calculation and validation setups.

Cases compared:
    1. Deterministic domain (no minRAM, no validation).
    2. Deterministic domain with minRAM.
    3. Deterministic domain with validation (IVA).
    4. Validation with forecast uncertainty in solar (non-robust IVA).
    5. Validation with uncertainty (robust chance-constrained validation).

Outputs:
    - Flow-based domains for each case.
    - Line 2 flow comparison (basecase, market coupling, redispatch).
    - Custom legend for publication-quality figures.
"""

import pandas as pd
import munacco as mc
import plotly.graph_objects as go

# -------------------------------------------------------------------------
# Load precalculated meshpoints (deterministic and uncertain cases)
# -------------------------------------------------------------------------
meshpoints_det = pd.read_pickle("thesis-data/stylised-4node/02_meshpoints_det.pkl")
meshpoints_unc = pd.read_pickle("thesis-data/stylised-4node/02_meshpoints_unc.pkl")

#%% Step 1: Load base network
loader = mc.InputLoader()
network = loader.load_from_csv("thesis-data/stylised-4node/base_model")
network.split_res_generators({'wind': 0.7, 'solar': 0.5})
network.initialize()

#%% Step 2: Prepare reusable model
model = mc.CACMModel(options_path="munacco/model/options_default.json")
model.options['capacity_calculation']['basecase'] = 'opf'
model.options['capacity_calculation']['frm'] = 0
model.options['capacity_calculation']['include_minram'] = False
model.options['validation']['include'] = False

# -------------------------------------------------------------------------
# Define scenarios for each case
# -------------------------------------------------------------------------

# Case 1: deterministic domain (no minRAM, no validation)
scenario_case1 = mc.ScenarioGenerator().generate(
    network, 1, forecast_timing=['d0'], forecast_sigma={'wind': 0, 'solar': 0}
)[0]
model.options['capacity_calculation']['include_minram'] = False
model.options['validation']['include'] = False
model.run(scenario_case1)

# Case 2: deterministic domain with minRAM
scenario_case2 = mc.ScenarioGenerator().generate(
    network, 1, forecast_timing=['d0'], forecast_sigma={'wind': 0, 'solar': 0}
)[0]
model.options['capacity_calculation']['include_minram'] = True
model.options['capacity_calculation']['minram'] = 0.5
model.options['validation']['include'] = False
model.run(scenario_case2)

# Case 3: deterministic domain + validation (IVA)
scenario_case3 = mc.ScenarioGenerator().generate(
    network, 1, forecast_timing=['d0'], forecast_sigma={'wind': 0, 'solar': 0}
)[0]
model.options['validation']['include'] = True
model.options['validation']['vertex_selection'] = True
model.options['validation']['max_vertex_angle'] = 40
model.options['validation']['robust'] = False
model.run(scenario_case3)

# Case 4: validation with solar uncertainty (non-robust)
scenario_case4 = mc.ScenarioGenerator().generate(
    network, 1, forecast_timing=['d0'], forecast_sigma={'wind': 0, 'solar': 0.2}
)[0]
# tweak RES forecast to match case study
scenario_case4.res_forecast.loc['pv3', 'p_d0'] = 9.5
scenario_case4.res_forecast.loc['pv4', 'p_d0'] = 30.7
model.options['validation']['include'] = True
model.options['validation']['robust'] = False
model.run(scenario_case4)

# Case 5: validation with uncertainty (robust chance-constrained)
scenario_case5 = mc.ScenarioGenerator().generate(
    network, 1, forecast_timing=['d0'], forecast_sigma={'wind': 0, 'solar': 0.2}
)[0]
scenario_case5.res_forecast.loc['pv3', 'p_d0'] = 9.5
scenario_case5.res_forecast.loc['pv4', 'p_d0'] = 30.7
model.options['validation']['include'] = True
model.options['validation']['robust'] = True
model.options['validation']['robust_method'] = 'chance_constrained'
model.run(scenario_case5)

#%% Step 3: Plot domains for all cases
def plot_domain(scenario, meshpoints, highlight=None, show_legend=None):
    inspector = mc.ScenarioInspector(scenario)
    p = inspector.create_domain_plot(('Z1','Z2'), ('Z1','Z3'), 'market_coupling', show=False)
    p.xlims = [-250, 350]
    p.ylims = [-350, 250]
    p.mesh_points = meshpoints
    p.figure_pub(
        domains=['initial'] if highlight == ['initial'] else ['initial', 'minram', 'iva'],
        highlight=highlight,
        show_legend=show_legend
    )

plot_plan = [
    (scenario_case1, meshpoints_det, ['initial'], False),
    (scenario_case2, meshpoints_det, ['minram'], False),
    (scenario_case3, meshpoints_det, ['iva'], False),
    (scenario_case4, meshpoints_unc, ['iva'], False),
    (scenario_case5, meshpoints_unc, ['iva'], False),
]
for scen, mesh, highlight, show_legend in plot_plan:
    plot_domain(scen, mesh, highlight=highlight, show_legend=show_legend)

#%% Step 4: Compare flows on Line 2 (grouped bar chart)
def collect_l2_flows(scenario, case_name):
    """Extract flow on Line 2 for basecase, MC, redispatch."""
    rows = []
    for result_name in ["basecase", "market_coupling", "redispatch"]:
        if result_name not in scenario.results:
            continue
        F = scenario.results[result_name].F
        rows.append({"case": case_name, "result": result_name, "flow_l2": F[1]})
    return rows

cases = [
    ("Case 1", scenario_case1),
    ("Case 2", scenario_case2),
    ("Case 3", scenario_case3),
    ("Case 4", scenario_case4),
    ("Case 5", scenario_case5),
]

# Build tidy DataFrame
flows_l2 = pd.DataFrame([rec for case, scen in cases for rec in collect_l2_flows(scen, case)])

# Colors + pretty legend labels
color_map = {
    "basecase": ("rgba(65,105,225,0.5)", "royalblue", "Basecase"),
    "market_coupling": ("rgba(178,34,34,0.5)", "firebrick", "Market Coupling"),
    "redispatch": ("rgba(147,112,219,0.5)", "mediumpurple", "Redispatch"),
}

fig = go.Figure()
for result, (fill, edge, label) in color_map.items():
    subset = flows_l2[flows_l2["result"] == result]
    fig.add_trace(go.Bar(
        x=subset["case"], y=subset["flow_l2"], name=label,
        marker=dict(color=fill, line=dict(color=edge, width=2)),
        offsetgroup=result,
    ))

# Capacity line (dashed)
fig.add_hline(y=30, line=dict(color="black", width=3, dash="dash"), layer="above")
fig.add_annotation(
    xref="paper", x=1.01, yref="y", y=30,
    text="  Maximum<br>Capacity", showarrow=True,
    arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor="black",
    ax=65, ay=0, xanchor="left",
    font=dict(size=40, family="serif")
)

fig.update_layout(
    template="plotly_white",
    font=dict(family="serif", size=40),
    yaxis=dict(title="Flow on Line 2", title_font=dict(size=46),
               showline=True, linewidth=1.5, linecolor="black",
               ticks="outside", showgrid=True, gridcolor="lightgrey", gridwidth=1, griddash="dash"),
    barmode="group", bargap=0.10, bargroupgap=0.05,
    legend=dict(title="", font=dict(size=40), x=1.02, y=1, xanchor="left", yanchor="top",
                bordercolor="black", borderwidth=1),
    margin=dict(l=80, r=220, t=80, b=80),
    height=750, width=1280,
)
fig.show()
fig.write_html("flows.html")

#%% Step 5: Legend figure (standalone for publication)
fig = go.Figure()

# Dummy legend entries
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                         line=dict(color="black", width=3), name="Market Domain"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                         line=dict(color="red", width=2),
                         fill="toself", fillcolor="rgba(255,0,0,0.4)",
                         name="Operationally insecure positions"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                         line=dict(color="rgba(241,234,164,0.8)", width=2),
                         fill="toself", fillcolor="rgba(245,239,177,0.8)",
                         name="Not reachable (gen/load constraints)"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                         line=dict(color="rgba(83,171,53,0.8)", width=2),
                         fill="toself", fillcolor="rgba(129,203,103,0.8)",
                         name="Operationally secure positions"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                         marker=dict(symbol='circle-x', size=24, color='white',
                                     line=dict(color='red', width=3)),
                         name="Exchanges @ MCP"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                         marker=dict(symbol='square-x', size=24, color='white',
                                     line=dict(color='black', width=3)),
                         name="Exchanges @ NPF"))

fig.update_layout(
    showlegend=True,
    legend=dict(font=dict(family="Times New Roman, serif", size=25),
                bgcolor="white", orientation="v", yanchor="top", y=1,
                xanchor="left", x=0),
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(visible=False), yaxis=dict(visible=False),
    plot_bgcolor="white", paper_bgcolor="white"
)
fig.show()
