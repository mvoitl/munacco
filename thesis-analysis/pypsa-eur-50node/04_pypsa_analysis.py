"""
Postprocessing for 50-node CACM experiments
-------------------------------------------

This script aggregates and visualizes the results of the thesis experiments
on the 50-node PyPSA-Eur case. It loads the saved pickle files from the main run
script and produces:

1. Summary statistics on feasibility (share of runs with remaining overloads)
2. Mean cost comparisons across cases
3. A grouped boxplot figure comparing cost distributions
   (Day-ahead, Redispatch, Total System Cost)

Inputs (in folder `thesis-data/pypsa-eur-50node/result_data`):
    - snapshot_overview_det_high_res.pkl
    - snapshot_overview_robust_high_res.pkl
    - snapshot_overview_det_low_res.pkl
    - snapshot_overview_robust_low_res.pkl
    - all_data_det_high_res.pkl
    - all_data_robust_high_res.pkl
    - all_data_det_low_res.pkl
    - all_data_robust_low_res.pkl

Outputs:
    - Printed tables for summary and cost statistics
    - Plotly figure for cost distributions (interactive, publication-ready)
    - DataFrames `df_summary` and `df_costs` (can be saved as CSV if needed)
"""

import pandas as pd
import plotly.graph_objects as go

#%% Step 1: Load data

folder = "thesis-data/pypsa-eur-50node/result_data"

def load_pickle(path: str):
    """Print which file is being loaded and return the DataFrame."""
    print(f"Loading {path} ...")
    df = pd.read_pickle(path)
    print(f"Finished loading {path}")
    return df

overview_det_high   = load_pickle(f"{folder}/snapshot_overview_det_high_res.pkl")
overview_robust_high = load_pickle(f"{folder}/snapshot_overview_robust_high_res.pkl")
overview_det_low    = load_pickle(f"{folder}/snapshot_overview_det_low_res.pkl")
overview_robust_low  = load_pickle(f"{folder}/snapshot_overview_robust_low_res.pkl")

all_det_high        = load_pickle(f"{folder}/all_data_det_high_res.pkl")
all_robust_high     = load_pickle(f"{folder}/all_data_robust_high_res.pkl")
all_det_low         = load_pickle(f"{folder}/all_data_det_low_res.pkl")
all_robust_low      = load_pickle(f"{folder}/all_data_robust_low_res.pkl")

print("All pickles loaded successfully.\n")


#%% Step 2: Feasibility statistics
def analyze_case(overview, all_data, label):
    """Compute feasibility statistics for one case."""
    failed_runs = overview.loc[overview.status != 'ok']
    share_failed = len(failed_runs) / len(overview) if len(overview) > 0 else float("nan")

    df_ro = all_data.loc[all_data.remaining_overload]

    share_ro_0 = sum(all_data.groupby("snapshot")["remaining_overload"].sum() > 0) / (
        len(overview) - len(failed_runs)
    )
    share_ro_5 = sum(all_data.groupby("snapshot")["remaining_overload"].sum() > 5) / (
        len(overview) - len(failed_runs)
    )

    return {
        "case": label,
        "snapshots": len(overview),
        "failed_runs": len(failed_runs),
        "share_failed_%": round(share_failed * 100, 2),
        "n_scenarios_with_RO": len(df_ro),
        "snapshots_with_RO": len(df_ro["snapshot"].unique()),
        "share_RO>0%": round(share_ro_0 * 100, 2),
        "share_RO>5%": round(share_ro_5 * 100, 2),
    }


results = [
    analyze_case(overview_det_high, all_det_high, "Deterministic (High RES)"),
    analyze_case(overview_robust_high, all_robust_high, "Robust (High RES)"),
    analyze_case(overview_det_low, all_det_low, "Deterministic (Low RES)"),
    analyze_case(overview_robust_low, all_robust_low, "Robust (Low RES)"),
]

df_summary = pd.DataFrame(results)
print("\n=== Feasibility Summary ===")
print(df_summary.to_string(index=False))


#%% Step 3: Cost statistics
def analyze_costs(all_data, label):
    """Compute mean costs for one case (in k€)."""
    return {
        "case": label,
        "mean_market_cost_k€": all_data.market_cost.mean() / 1000,
        "mean_rd_cost_k€": all_data.rd_cost.mean() / 1000,
        "mean_total_system_cost_k€": all_data.total_system_cost.mean() / 1000,
    }


costs = [
    analyze_costs(all_det_high, "Deterministic (High RES)"),
    analyze_costs(all_robust_high, "Robust (High RES)"),
    analyze_costs(all_det_low, "Deterministic (Low RES)"),
    analyze_costs(all_robust_low, "Robust (Low RES)"),
]

df_costs = pd.DataFrame(costs)
print("\n=== Cost Overview (mean values, k€) ===")
print(df_costs.to_string(index=False, float_format="%.2f"))


#%% Step 4: Boxplot visualization
kpis = ["market_cost", "rd_cost", "total_system_cost"]
kpi_colors = {
    "market_cost": ("rgba(65,105,225,0.5)", "royalblue", "Day-Ahead Market Cost"),
    "rd_cost": ("rgba(178,34,34,0.5)", "firebrick", "Redispatch Cost"),
    "total_system_cost": ("rgba(147,112,219,0.5)", "mediumpurple", "Total System Cost"),
}

case_titles = {
    ("det", "low"): "Det.",
    ("robust", "low"): "Prob.",
    ("det", "high"): "Det.",
    ("robust", "high"): "Prob.",
}

dfs = {
    ("det", "low"): all_det_low,
    ("robust", "low"): all_robust_low,
    ("det", "high"): all_det_high,
    ("robust", "high"): all_robust_high,
}

df_long = pd.concat(
    [
        pd.DataFrame(
            {"case": case_titles[(case, res)], "kpi": kpi, "value": df[kpi]}
        )
        for (case, res), df in dfs.items()
        for kpi in kpis
    ]
)

fig = go.Figure()
for kpi in kpis:
    kpi_df = df_long[df_long["kpi"] == kpi]
    fill, edge, label = kpi_colors[kpi]
    fig.add_trace(
        go.Box(
            x=kpi_df["case"],
            y=kpi_df["value"],
            name=label,
            marker_color=fill,
            line=dict(color=edge, width=2),
            boxmean=False,
            legendgroup=kpi,
            showlegend=True,
            fillcolor=fill,
        )
    )

# Layout & styling
fig.update_layout(
    template="plotly_white",
    font=dict(family="serif", size=36, color="black"),
    boxmode="group",
    boxgap=0.4,
    boxgroupgap=0.15,
    height=900,
    width=1400,
    margin=dict(l=120, r=250, t=100, b=180),
    legend=dict(
        title="",
        font=dict(size=32, color="black"),
        orientation="h",
        x=0.5,
        y=1.1,
        xanchor="center",
        yanchor="top",
        bordercolor="black",
        borderwidth=0,
    ),
    xaxis=dict(
        title="",
        tickfont=dict(size=32, color="black"),
        categoryorder="array",
        categoryarray=list(case_titles.values()),
        showline=True,
        linewidth=2,
        linecolor="black",
        ticks="outside",
        showgrid=False,
    ),
    yaxis=dict(
        title="Cost [€]",
        tickfont=dict(size=32, color="black"),
        title_font=dict(size=38, color="black"),
        showline=True,
        linewidth=2,
        linecolor="black",
        ticks="outside",
        showgrid=True,
        gridcolor="lightgrey",
        gridwidth=1,
        griddash="dash",
    ),
)

# Annotate RES groups
fig.add_annotation(x=0.84, y=-0.18, xref="paper", yref="paper",
                   text="High RES", showarrow=False,
                   font=dict(size=34, family="serif"))
fig.add_annotation(x=0.16, y=-0.18, xref="paper", yref="paper",
                   text="Low RES", showarrow=False,
                   font=dict(size=34, family="serif"))

# Group lines
fig.add_shape(type="line", x0=0.05, x1=0.45, y0=-0.10, y1=-0.10,
              xref="paper", yref="paper", line=dict(color="black", width=2))
fig.add_shape(type="line", x0=0.55, x1=0.95, y0=-0.10, y1=-0.10,
              xref="paper", yref="paper", line=dict(color="black", width=2))

fig.show()
