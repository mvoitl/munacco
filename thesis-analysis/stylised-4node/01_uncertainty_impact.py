"""
4-Node Case Study: Forecast Uncertainty in Capacity Calculation

This script analyzes the impact of RES forecast uncertainty on 
the Flow-Based Market Coupling (FBMC) process. 

We compare:
- Pure Capacity Calculation (CC) runs with different minRAM levels.
- Capacity Calculation + Validation runs with different minRAM levels.

Outputs:
1. Distribution of MCP distances across scenarios (boxplots).
2. Flow-based domain visualizations (2×2 subplot grid).
"""

import munacco as mc
from tqdm import tqdm
import numpy as np

# ---------------------------------------------------------------
# Helper: Euclidean distance between vectors
# Used to compute MCP distance between scenarios and the reference.
# ---------------------------------------------------------------
def euclidean_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.linalg.norm(vec1 - vec2)

# ---------------------------------------------------------------
# 1. Load network topology (stylized 4-node test case)
# ---------------------------------------------------------------
loader = mc.InputLoader()
network = loader.load_from_csv("thesis-data/stylised-4node/base_model")
network.split_res_generators({'wind': 0.7, 'solar': 0.5})
network.initialize()

# ---------------------------------------------------------------
# 2. Generate scenarios
#    - 300 stochastic cases with forecast error (σ=0.2)
#    - 1 reference case (perfect forecast, no error)
# ---------------------------------------------------------------
scenarios = mc.ScenarioGenerator().generate(network, 300, forecast_timing=['d2'], forecast_sigma={'wind': 0.2, 'solar': 0.2})
scenario = mc.ScenarioGenerator().generate(network, 1, forecast_timing=['d0', 'd1', 'd2'], forecast_sigma={'wind': 0, 'solar': 0})[0]
scenario.id = 'reference'
scenarios += [scenario]

#%%
# ===============================================================
# EXPERIMENT 1: Capacity Calculation without validation
# ===============================================================
ls_analyzer_cc = []

for minram in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    scs = scenarios.copy()

    # Configure model
    model = mc.CACMModel(options_path="munacco/model/options_default.json")
    model.options['capacity_calculation']['include_minram'] = True
    model.options['capacity_calculation']['minram'] = minram
    model.options['validation']['include'] = False

    # Run all scenarios
    for scenario in tqdm(scs):
        model.options['model']['print'] = False
        model.run(scenario)

    # Compute MCP distance to reference
    ref_sc = next((s for s in scenarios if s.id == 'reference'), None)
    for sc in scs:
        dist = euclidean_distance(sc.results['market_coupling'].NP, ref_sc.results['market_coupling'].NP)
        sc.kpis['mcp_dist'] = dist

    ls_analyzer_cc.append(mc.Analyzer(scs, {'case': f'minram_{minram}'}))

# ===============================================================
# EXPERIMENT 2: Capacity Calculation with validation
# ===============================================================
ls_analyzer_val = []

for minram in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    scs = scenarios.copy()

    # Configure model
    model = mc.CACMModel(options_path="munacco/model/options_default.json")
    model.options['capacity_calculation']['include_minram'] = True
    model.options['capacity_calculation']['minram'] = minram
    model.options['validation']['include'] = True
    model.options['validation']['vertex_selection'] = True
    model.options['validation']['max_vertex_angle'] = 40

    # Run all scenarios
    for scenario in tqdm(scs):
        model.options['model']['print'] = False
        model.run(scenario)

    # Compute MCP distance to reference
    ref_sc = next((s for s in scenarios if s.id == 'reference'), None)
    for sc in scs:
        dist = euclidean_distance(sc.results['market_coupling'].NP, ref_sc.results['market_coupling'].NP)
        sc.kpis['mcp_dist'] = dist

    ls_analyzer_val.append(mc.Analyzer(scs, {'case': f'minram_{minram}'}))

#%%
# ===============================================================
# PLOT 1: Boxplots of MCP distance vs minRAM
# ===============================================================
import plotly.graph_objects as go

fig = go.Figure()

# Colors for CC and CC+Validation
color_cc = "royalblue"
color_val = "firebrick"

# Only show legend entry once per group
showlegend_cc = True
showlegend_val = True

# Desired x-axis order
minram_order = ['0%', '10%', '20%', '30%', '40%', '50%']

# ---- CC analyzers ----
for i, analyzer in enumerate(ls_analyzer_cc):
    label = minram_order[i]
    y = analyzer.df['mcp_dist']

    fig.add_trace(go.Box(
        y=y,
        x=[label] * len(y),
        name="Capacity Calculation",
        legendgroup="Capacity Calculation",
        showlegend=showlegend_cc,
        offsetgroup="Capacity Calculation",
        alignmentgroup="mcp",
        marker_color=color_cc,
        boxmean=True
    ))
    showlegend_cc = False

# ---- CC + Validation analyzers ----
for i, analyzer in enumerate(ls_analyzer_val):
    label = minram_order[i]
    y = analyzer.df['mcp_dist']

    fig.add_trace(go.Box(
        y=y,
        x=[label] * len(y),
        name="CC + Validation",
        legendgroup="CC + Validation",
        showlegend=showlegend_val,
        offsetgroup="CC + Validation",
        alignmentgroup="mcp",
        marker_color=color_val,
        boxmean=True
    ))
    showlegend_val = False

# Layout
fig.update_layout(
    yaxis_title="MCP Distance",
    template="simple_white",
    font=dict(family="serif", size=40),
    xaxis=dict(
        title="minRAM",
        tickfont=dict(size=40),
        title_font=dict(size=40),
        categoryorder='array',
        categoryarray=minram_order
    ),
    yaxis=dict(
        tickfont=dict(size=40),
        title_font=dict(size=40)
    ),
    boxmode='group',
    boxgroupgap=0.1,
    boxgap=0.15,
    legend=dict(
        title='',
        font=dict(size=40),
        x=0.4, y=0.98,
        xanchor='right', yanchor='top',
        bgcolor="rgba(255,255,255,0.6)",
        bordercolor="black",
        borderwidth=1
    )
)

fig.show()
fig.write_html("minram_box.html")

#%%
# ===============================================================
# PLOT 2: Flow-Based Domains (2×2 subplot)
# ===============================================================
from scipy import spatial
from plotly.subplots import make_subplots
from munacco.analysis.visualization import FBdomainPlot
from munacco.tools import compute_polytope_vertices

# Helper function: add FB domain traces for one analyzer
def add_fb_domain_traces_to_subplot(analyzer, x_axis, y_axis, result_name, domain,
                                    fig, row, col,
                                    color_reference='red', color_other='grey',
                                    alpha_other=0.05):
    for sc in analyzer.scenarios:
        plot = FBdomainPlot(sc, x_axis, y_axis, result_name=result_name)
        A = np.array(sc.fb_parameters[domain][sc.network.Z])
        b = np.array(sc.fb_parameters[domain]['RAM'])
        A_2d, b_shift = plot.create_domain_representation(plot.x_axis, plot.y_axis, A, b, plot.np_shift)

        color = color_reference if sc.id == 'reference' else color_other
        op = 1.0 if sc.id == 'reference' else alpha_other

        feas = np.array(compute_polytope_vertices(A_2d, b))
        if feas.shape[0] >= 3:
            hull = spatial.ConvexHull(feas)
            idx = list(hull.vertices) + [hull.vertices[0]]
            x_poly, y_poly = feas[idx, 0], feas[idx, 1]

            if sc.id == 'reference':
                # Reference: filled underlay + crisp outline
                fig.add_trace(go.Scatter(
                    x=x_poly, y=y_poly, mode='lines',
                    fill='toself', fillcolor=color, opacity=0.30,
                    line=dict(color='rgba(0,0,0,0)'),
                    hoverinfo='skip', showlegend=False
                ), row=row, col=col)
                fig.add_trace(go.Scatter(
                    x=x_poly, y=y_poly, mode='lines',
                    line=dict(color=color, width=2), opacity=1.0,
                    showlegend=False
                ), row=row, col=col)
            else:
                # Other scenarios: light outline only
                fig.add_trace(go.Scatter(
                    x=x_poly, y=y_poly, mode='lines',
                    opacity=op, line=dict(color=color, width=1),
                    showlegend=False
                ), row=row, col=col)

        # Add MCP marker
        ex_pt = (
            sc.results['market_coupling'].EX[x_axis] - sc.results['market_coupling'].EX[x_axis[::-1]],
            sc.results['market_coupling'].EX[y_axis] - sc.results['market_coupling'].EX[y_axis[::-1]]
        )
        fig.add_trace(go.Scatter(
            x=[ex_pt[0]], y=[ex_pt[1]],
            mode='markers',
            marker=dict(color=color, size=12, symbol='circle',
                        line=dict(color='black', width=1)),
            showlegend=False
        ), row=row, col=col)

# ---- Build subplot grid (2×2) ----
subplot_titles = (
    "Capacity Calculation minRAM 0%",
    "Capacity Calculation minRAM 40%",
    "Validation minRAM 0%",
    "Validation minRAM 40%",
)

an_cc_0   = next((a for a in ls_analyzer_cc  if a.metadata['case'] == 'minram_0'),   None)
an_cc_40  = next((a for a in ls_analyzer_cc  if a.metadata['case'] == 'minram_0.4'), None)
an_val_0  = next((a for a in ls_analyzer_val if a.metadata['case'] == 'minram_0'),   None)
an_val_40 = next((a for a in ls_analyzer_val if a.metadata['case'] == 'minram_0.4'), None)

fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                    horizontal_spacing=0.12, vertical_spacing=0.12,
                    subplot_titles=subplot_titles)

# Style subplot titles
def style_subplot_titles(fig, size=24, family="serif", yshift=0.06, color=None):
    anns = []
    for ann in (fig.layout.annotations or []):
        is_subplot_title = (
            isinstance(ann.xref, str) and ann.xref.endswith(" domain") and
            isinstance(ann.yref, str) and ann.yref.endswith(" domain")
        )
        if is_subplot_title:
            a = ann.to_plotly_json()
            a["font"] = dict(family=family, size=size, color=(color or "black"))
            a["y"] = a.get("y", 1.0) + yshift
            a["yanchor"], a["xanchor"] = "bottom", "center"
            anns.append(a)
        else:
            anns.append(ann)
    fig.update_layout(annotations=anns)

style_subplot_titles(fig, size=24, family="serif", yshift=0.06)

# Axes and domains
x_axis = ('Z1', 'Z2')
y_axis = ('Z1', 'Z3')
result_name = 'market_coupling'

add_fb_domain_traces_to_subplot(an_cc_0,  x_axis, y_axis, result_name, 'minram', fig, 1, 1, color_reference='royalblue')
add_fb_domain_traces_to_subplot(an_cc_40, x_axis, y_axis, result_name, 'minram', fig, 1, 2, color_reference='royalblue')
add_fb_domain_traces_to_subplot(an_val_0, x_axis, y_axis, result_name, 'iva',    fig, 2, 1, color_reference='firebrick')
add_fb_domain_traces_to_subplot(an_val_40,x_axis, y_axis, result_name, 'iva',    fig, 2, 2, color_reference='firebrick')

# Global layout
fig.update_layout(template="simple_white", font=dict(family="serif", size=22),
                  showlegend=False, margin=dict(l=100, r=100, t=100, b=100),
                  height=900, width=900)

# Shared axis styling
fig.update_xaxes(range=[-160, 260], ticks="outside", showline=True, mirror=True,
                 linewidth=1.5, linecolor="black",
                 zeroline=True, zerolinewidth=1, zerolinecolor="black",
                 showgrid=True, gridcolor="lightgrey", gridwidth=1, griddash="dash")
fig.update_yaxes(range=[-200, 140], ticks="outside", showline=True, mirror=True,
                 linewidth=1.5, linecolor="black",
                 zeroline=True, zerolinewidth=1, zerolinecolor="black",
                 showgrid=True, gridcolor="lightgrey", gridwidth=1, griddash="dash")

# Square aspect ratio per subplot
fig.update_xaxes(scaleanchor="y",  scaleratio=1, row=1, col=1)
fig.update_xaxes(scaleanchor="y2", scaleratio=1, row=1, col=2)
fig.update_xaxes(scaleanchor="y3", scaleratio=1, row=2, col=1)
fig.update_xaxes(scaleanchor="y4", scaleratio=1, row=2, col=2)

# Global axis labels
x_title = f"Exchange Zone {x_axis[0]} → Zone {x_axis[1]}"
y_title = f"Exchange Zone {y_axis[0]} → Zone {y_axis[1]}"

existing_ann = list(fig.layout.annotations) if fig.layout.annotations else []
axis_label_annotations = [
    dict(text=x_title, x=0.5, y=-0.06, xref="paper", yref="paper",
         showarrow=False, xanchor="center", yanchor="top", font=dict(size=24)),
    dict(text=y_title, x=-0.08, y=0.5, xref="paper", yref="paper",
         showarrow=False, textangle=-90, xanchor="right", yanchor="middle", font=dict(size=24)),
]
fig.update_layout(annotations=existing_ann + axis_label_annotations)
fig.update_annotations(font_size=24)

fig.show()
