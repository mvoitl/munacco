"""
Multi-Scenario Experiment: 4-Node Case
--------------------------------------
This experiment evaluates the impact of validation and robustness settings on
market outcomes under forecast uncertainty. We compare:

    Case 1: Capacity Calculation + minRAM, No Validation
    Case 2: Capacity Calculation + minRAM, Deterministic Validation
    Case 3: Capacity Calculation + minRAM, Probabilistic (Chance-Constrained) Validation

Outputs:
    - KPI distributions (feasible vs. infeasible redispatch solutions).
    - Boxplots of system costs across cases.
    - Sensitivity statistics (mean values, infeasibility shares).
"""

import munacco as mc
from tqdm import tqdm
import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#%% Step 1: Load network
network = mc.InputLoader().load_from_csv("thesis-data/stylised-4node/base_model")
network.split_res_generators({'wind': 0.7, 'solar': 0.5})
network.initialize()

#% Step 2: Generate scenarios
scenarios = mc.ScenarioGenerator(random_seed=5).generate(
    network, 600, forecast_timing=['d0', 'd1']
)

#% Step 3: Define experiment cases
cases = {
    'Case 1': {'minram': True, 'validation': False, 'robust': False},
    'Case 2': {'minram': True, 'validation': True,  'robust': False},
    'Case 3': {'minram': True, 'validation': True,  'robust': True},
}

kpis_all = {}
err_flows_all = {}

#%% Step 4: Run experiments
for c in cases:
    model = mc.CACMModel(options_path="munacco/model/options_default.json")
    model.options['capacity_calculation'].update({
        'basecase': 'opf',
        'include_minram': cases[c]['minram'],
        'minram': 0.5,
        'frm': 0.1,
    })
    model.options['validation'].update({
        'include': cases[c]['validation'],
        'vertex_selection': True,
        'robust': cases[c]['robust'],
        'robust_method': 'chance_constrained',
    })

    # Base scenario for CC and validation
    scenario_base = copy.deepcopy(scenarios[0])
    model.run_capacity_calculation(scenario_base)
    model.run_validation(scenario_base)

    # Run MC + redispatch for all scenarios
    for scenario in tqdm(scenarios, desc=f"Running {c}"):
        scenario.GENMargin = scenario_base.GENMargin
        scenario.alpha = scenario_base.alpha
        scenario.npf = scenario_base.npf
        scenario.fb_parameters = scenario_base.fb_parameters
        scenario.results = copy.deepcopy(scenario_base.results)

        model.options['model']['print'] = False
        model.run_market_coupling(scenario)
        model.run_redispatch(scenario)
        model.collect_kpis(scenario)

    analyzer = mc.Analyzer(scenarios)
    kpis_all[c] = analyzer.df
    err_flows_all[c] = analyzer.err_flows

# Quick summary printout
for res in kpis_all:
    infeasible_count = sum(kpis_all[res].remaining_overload)
    print(f"{res}: {infeasible_count} infeasible redispatch solutions")

#%% Step 5: KPI histograms (feasible vs infeasible)
kpi = 'total_system_cost'
case_titles = {
    "Case 1": "<b>No Validation</b>",
    "Case 2": "<b>Det. Validation</b>",
    "Case 3": "<b>Prob. Validation</b>",
}
case_colors = {
    "Case 1": ("rgba(65,105,225,0.5)", "royalblue"),
    "Case 2": ("rgba(178,34,34,0.5)", "firebrick"),
    "Case 3": ("rgba(147,112,219,0.5)", "mediumpurple"),
}
case_hatches = {c: "/" for c in case_colors}
infeasible_fill_color = "lightgrey"

# Global binning
all_costs = pd.concat([df[kpi] for df in kpis_all.values()])
nbins = 70
bins = np.linspace(all_costs.min(), all_costs.max(), nbins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Subplot layout
fig = make_subplots(
    rows=len(cases), cols=1,
    shared_xaxes=True, shared_yaxes=True,
    subplot_titles=[case_titles[c] for c in cases],
    vertical_spacing=0.08,
)

# Add traces per case
for i, (case_name, df) in enumerate(kpis_all.items(), start=1):
    feasible = df[df['remaining_overload'] == False]
    infeasible = df[df['remaining_overload'] == True]
    fill, edge = case_colors[case_name]

    # Infeasible
    infeasible_counts, _ = np.histogram(infeasible[kpi], bins=bins)
    fig.add_trace(go.Bar(
        x=bin_centers, y=infeasible_counts,
        marker=dict(
            color=infeasible_fill_color,
            line=dict(color=edge, width=2),
            pattern=dict(shape=case_hatches[case_name],
                         fgcolor=edge, bgcolor=infeasible_fill_color),
        ),
        opacity=0.9, showlegend=False,
    ), row=i, col=1)

    # Feasible
    feasible_counts, _ = np.histogram(feasible[kpi], bins=bins)
    fig.add_trace(go.Bar(
        x=bin_centers, y=feasible_counts,
        marker=dict(color=fill, line=dict(color=edge, width=2)),
        opacity=0.9, showlegend=False,
    ), row=i, col=1)

# Manual legend
fig.add_trace(go.Bar(
    x=[None], y=[None],
    name="Feasible RD Solution",
    marker=dict(color="grey", line=dict(color="black", width=2)),
    showlegend=True,
))
fig.add_trace(go.Bar(
    x=[None], y=[None],
    name="Infeasible RD Solution",
    marker=dict(color="lightgrey", line=dict(color="black", width=2),
                pattern=dict(shape="/", fgcolor="black", bgcolor="lightgrey")),
    showlegend=True,
))

# Layout
fig.update_layout(
    template="plotly_white",
    font=dict(family="serif", size=40, color="black"),
    barmode="stack", bargap=0.05,
    height=300 * len(cases), width=1400,
    margin=dict(l=120, r=200, t=80, b=100),
    legend=dict(
        title="", font=dict(size=30, color="black"),
        orientation="v", x=0.7, y=1, xanchor="left", yanchor="top",
        bordercolor="black", borderwidth=1,
    ),
)
for ann in fig['layout']['annotations']:
    ann['font'] = dict(size=30, family="serif", color="black")
    ann['yshift'] = -1

fig.show()

#%% Step 6: KPI Boxplots (market, RD, total cost)
kpis = ['market_cost', 'rd_cost', 'total_system_cost']
kpi_colors = {
    'market_cost': ("rgba(65,105,225,0.5)", "royalblue", "Day-Ahead Market Cost"),
    'rd_cost': ("rgba(178,34,34,0.5)", "firebrick", "Redispatch Cost"),
    'total_system_cost': ("rgba(147,112,219,0.5)", "mediumpurple", "Total System Cost"),
}
case_titles = {"Case 1": "No Validation", "Case 2": "Det. Validation", "Case 3": "Prob. Validation"}

all_data = []
for case_name, df in kpis_all.items():
    for kpi in kpis:
        all_data.append(pd.DataFrame({
            "case": case_titles[case_name],
            "kpi": kpi,
            "value": df[kpi],
        }))
df_long = pd.concat(all_data)

fig = go.Figure()
for kpi in kpis:
    kpi_df = df_long[df_long["kpi"] == kpi]
    fill, edge, label = kpi_colors[kpi]
    fig.add_trace(go.Box(
        x=kpi_df["case"], y=kpi_df["value"],
        name=label, marker_color=fill, line=dict(color=edge, width=2),
        boxmean=False, legendgroup=kpi, showlegend=True, fillcolor=fill,
    ))

fig.update_layout(
    template="plotly_white",
    font=dict(family="serif", size=30, color="black"),
    boxmode='group', boxgap=0.5, boxgroupgap=0.2,
    height=800, width=1200,
    margin=dict(l=120, r=200, t=80, b=100),
    legend=dict(
        title="", font=dict(size=30, color="black"),
        orientation="v", x=0.02, y=1, xanchor="left", yanchor="top",
        bordercolor="black", borderwidth=1,
    ),
    xaxis=dict(
        tickfont=dict(size=36, color="black"),
        categoryorder="array", categoryarray=list(case_titles.values()),
        showline=True, linewidth=2, linecolor="black", ticks="outside",
    ),
    yaxis=dict(
        title="Cost [€]", tickfont=dict(size=30, color="black"),
        title_font=dict(size=36, color="black"),
        showline=True, linewidth=2, linecolor="black", ticks="outside",
        showgrid=True, gridcolor="lightgrey", griddash="dash",
    ),
)
fig.show()

#%% Step 7: Statistics

# Combine results into one long DataFrame
df_all = pd.concat([df.assign(case=c) for c, df in kpis_all.items()])

# --- Aggregate KPIs ---
kpi_summary = df_all.groupby("case").agg(
    mean_market_cost=("market_cost", "mean"),
    mean_rd_cost=("rd_cost", "mean"),
    mean_total_system_cost=("total_system_cost", "mean"),
    mean_rd_volume=("rd_volume", "mean"),
    infeasible_share=("remaining_overload", lambda x: x.sum() / len(x))
).reset_index()

# --- Slack statistics (only scenarios with nonzero slack) ---
df_slack = df_all.loc[df_all.total_slack > 1e-5]
slack_summary = df_slack.groupby("case").agg(
    mean_total_slack=("total_slack", "mean"),
    max_total_slack=("total_slack", "max")
).reset_index()

# --- Merge both summaries for convenience ---
summary = pd.merge(kpi_summary, slack_summary, on="case", how="left")


