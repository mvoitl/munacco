"""
4-Node Experiment: Plotting Sensitivity to Validation Risk Parameter (ε)
------------------------------------------------------------------------

This script visualizes the effect of the validation risk parameter ε
on system performance. It shows:

- Total system cost (mean + interquartile range) as ε varies
- Average size of remaining overloads (slack) [% of line capacity]
- Number of scenarios with remaining overloads (infeasible cases)
- Deterministic validation as a benchmark (vertical line + markers)

The plot uses dual y-axes:
    Left  axis: Total system cost [€]
    Right axis: Feasibility metrics [% and counts]

"""

#%
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ----------------------------
# Step 1: Load precomputed results
# ----------------------------
df_all = pd.read_pickle('thesis-data/stylised-4node/04_results_sensitivity_100.pkl')
det_val_df = pd.read_pickle('thesis-data/stylised-4node/04_results_sensitivity_det.pkl')

# ----------------------------
# Step 2: Aggregate statistics per ε
# ----------------------------
summary = []
for eps, group in df_all.groupby('epsilon'):
    total_costs = group['total_system_cost']
    infeasible = group[group['remaining_overload'] == True]
    slack_values = infeasible['total_slack']  # only for infeasible runs

    summary.append({
        'epsilon': eps,
        'cost_mean': total_costs.mean(),
        'cost_q25': np.percentile(total_costs, 25),
        'cost_q75': np.percentile(total_costs, 75),
        'slack_mean': slack_values.mean() * 100 / 30 if not slack_values.empty else 0,
        'slack_q25': np.percentile(slack_values, 25) * 100 / 30 if not slack_values.empty else 0,
        'slack_q75': np.percentile(slack_values, 75) * 100 / 30 if not slack_values.empty else 0,
        'num_infeasible': len(infeasible)
    })

summary_df = pd.DataFrame(summary)

# ----------------------------
# Step 3: Create Plotly figure
# ----------------------------
fig = go.Figure()

# --- System cost (left y-axis) ---
# Interquartile shaded band
fig.add_trace(go.Scatter(
    x=summary_df['epsilon'], y=summary_df['cost_q75'],
    mode='lines', line=dict(width=0), hoverinfo='skip',
    showlegend=False, yaxis='y1'
))
fig.add_trace(go.Scatter(
    x=summary_df['epsilon'], y=summary_df['cost_q25'],
    mode='lines', fill='tonexty', fillcolor='rgba(65,105,225,0.2)',
    line=dict(width=0), hoverinfo='skip',
    name='System cost (25–75%)', showlegend=False, yaxis='y1'
))
# Mean line
fig.add_trace(go.Scatter(
    x=summary_df['epsilon'], y=summary_df['cost_mean'],
    mode='lines+markers',
    line=dict(color='royalblue', width=5),
    marker=dict(size=6),
    name='Mean total system cost', yaxis='y1'
))

# --- Slack size (right y-axis) ---
fig.add_trace(go.Scatter(
    x=summary_df['epsilon'], y=summary_df['slack_q75'],
    mode='lines', line=dict(width=0), hoverinfo='skip',
    showlegend=False, yaxis='y2'
))
fig.add_trace(go.Scatter(
    x=summary_df['epsilon'], y=summary_df['slack_q25'],
    mode='lines', fill='tonexty', fillcolor='rgba(147,112,219,0.2)',
    line=dict(width=0), hoverinfo='skip',
    name='Total slack (25–75%)', showlegend=False, yaxis='y2'
))
fig.add_trace(go.Scatter(
    x=summary_df['epsilon'], y=summary_df['slack_mean'],
    mode='lines+markers',
    line=dict(color='mediumpurple', dash='dot', width=5),
    marker=dict(symbol='square', size=6),
    name='Mean size of Remaining Overloads', yaxis='y2'
))

# --- Infeasible counts (right y-axis) ---
fig.add_trace(go.Scatter(
    x=summary_df['epsilon'], y=summary_df['num_infeasible'],
    mode='lines',
    line=dict(color='firebrick', dash='dash', width=5),
    name='Scenarios with Remaining Overloads', yaxis='y2'
))

# ----------------------------
# Step 4: Deterministic benchmark (ε = 0.5)
# ----------------------------
det_epsilon = 0.5
det_cost = det_val_df['total_system_cost'].mean()
det_slack = det_val_df.loc[det_val_df['remaining_overload'], 'total_slack'].mean() * 100 / 30
det_infeasible = det_val_df['remaining_overload'].sum()

# Vertical line
fig.add_vline(x=det_epsilon, line=dict(color='black', dash='dot'), layer='below')

# Annotation
fig.add_annotation(
    x=det_epsilon, y=summary_df['cost_q75'].max(),
    text="Deterministic<br>Validation",
    showarrow=False,
    font=dict(size=25, family="serif", color="black"),
    align="left", xanchor="left", yanchor="middle"
)

# Markers for deterministic values
fig.add_trace(go.Scatter(x=[det_epsilon], y=[det_cost],
    mode='markers', marker=dict(symbol='square-dot', size=14, color='royalblue'),
    showlegend=False, yaxis='y1'))
fig.add_trace(go.Scatter(x=[det_epsilon], y=[det_slack],
    mode='markers', marker=dict(symbol='square-dot', size=14, color='mediumpurple'),
    showlegend=False, yaxis='y2'))
fig.add_trace(go.Scatter(x=[det_epsilon], y=[det_infeasible],
    mode='markers', marker=dict(symbol='square-dot', size=14, color='firebrick'),
    showlegend=False, yaxis='y2'))

# ----------------------------
# Step 5: Layout & Styling
# ----------------------------
fig.update_layout(
    template="plotly_white",
    font=dict(family="serif", size=40, color='black'),
    xaxis=dict(
        title="ε (Risk Tolerance)", title_font=dict(size=46),
        tickfont=dict(size=40),
        showline=True, linewidth=1.5, linecolor="black",
        ticks="outside", showgrid=False,
        autorange='reversed'  # decreasing ε from left to right
    ),
    yaxis=dict(
        title="Total System Cost [€]", title_font=dict(size=46),
        tickfont=dict(size=40),
        showline=True, linewidth=1.5, linecolor="black",
        ticks="outside",
        showgrid=True, gridcolor="lightgrey", gridwidth=1, griddash="dash"
    ),
    yaxis2=dict(
        title="Feasibility Metrics [%]",
        title_font=dict(size=46), tickfont=dict(size=40),
        overlaying="y", side="right",
        showline=True, linewidth=1.5, linecolor="black",
        ticks="outside", showgrid=False
    ),
    legend=dict(
        title="", orientation="h",
        x=0.5, y=1.10, xanchor="center", yanchor="bottom",
        font=dict(size=30),
        bordercolor="black", borderwidth=0,
        tracegroupgap=80
    ),
    margin=dict(l=80, r=220, t=80, b=80),
    height=750, width=1500,
)

fig.show()
