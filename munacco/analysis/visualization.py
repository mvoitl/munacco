"""
munacco.analysis.visualization

Provides the `FBdomainPlot` class and helper functions for creating
flow-based domain (FB) visualizations.

Main features:
- Build and validate feasible domains in 2D projections (e.g. x_axis vs y_axis exchanges)
- Use convex/concave hulls to represent feasible regions
- Include feasibility checks via cvxpy optimization
- Plot with Plotly (interactive) or generate publication-style plots

Key components:
- feasibility_validation(): solve small feasibility LPs for given NPs
- zero_sum_points() and generate_balanced_nps_from_axes(): generate NP grids
- FBdomainPlot: central class for building & visualizing domains
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import scipy.spatial as spatial
import cvxpy as cp
import pandas as pd
from itertools import product
from tqdm import tqdm

# External helper for concave hulls (used in publication figures)
from concave_hull import concave_hull_indexes

# --- Color palettes and hatch patterns (Okabe-Ito safe colors) ---
OKABE_ITO = {
    'initial': '#999999',  # grey
    'minram':  '#56B4E9',  # sky blue
    'iva':     '#D55E00',  # vermillion
    'amax':    '#009E73',  # bluish green
    'a':       '#F0E442',  # yellow
}

HATCHES = {
    'initial': '//',
    'minram':  '\\\\',
    'iva':     'xx',
    'amax':    '..',
    'a':       '++',
}

from munacco.model.result import Result
from munacco.tools import compute_polytope_vertices, compute_sec


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def feasibility_validation(self, vertex, modus='amax', forecast_day='d0'):
    """
    Check feasibility of a given net position vector using cvxpy.

    Parameters
    ----------
    vertex : array
        Candidate net positions for zones.
    modus : str
        'amax' for relaxed validation, 'a' for strict.
    forecast_day : str
        RES forecast day (e.g. 'd0', 'd2').

    Returns
    -------
    feasible : bool
        True if problem is feasible, False otherwise.
    """
    GEN = cp.Variable(len(self.scenario.network.P), pos=True)
    INJ = cp.Variable(len(self.scenario.network.N))
    CU = cp.Variable(len(self.scenario.network.RES), pos=True)
    A_f = cp.Variable(pos=True)
    D = self.scenario.network.nodes.Pd.values
    
    obj = cp.Minimize(A_f)

    constraints = [
        # Nodal Energy Balance
        self.scenario.network.map_np @ GEN
        + self.scenario.network.map_nres @ (self.scenario.res_forecast[f'p_{forecast_day}'].values - CU)
        - D == INJ,

        # Zonal Energy Balance
        self.scenario.network.map_nz.T @ INJ == vertex,

        # Curtailment ≤ forecast
        CU <= self.scenario.res_forecast['p_d2'].values,

        # Generator capacity constraints
        GEN <= self.scenario.network.plants.g_max.values,

        # Power flow balance
        sum(INJ) == 0,
    ]

    if modus == 'a':
        constraints.append(self.scenario.network.ptdf_nes @ INJ <= self.scenario.network.nes.f_max)

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        print("Clarabel failed, trying SCS as fallback...")
        prob.solve(solver=cp.SCS)
    
    return prob.status != 'optimal'


def zero_sum_points(val_max, step, n_dim):
    """
    Generate points in n-dim space with zero sum across coordinates.

    Used to construct candidate NP points on a grid.

    Returns
    -------
    np.ndarray of shape (n_points, n_dim)
    """
    assert n_dim >= 2, "Need at least 2 dimensions"

    grid_vals = np.arange(-val_max, val_max + step, step)
    base_combos = product(grid_vals, repeat=n_dim - 1)

    points = []
    for combo in base_combos:
        last_val = -sum(combo)
        if -val_max <= last_val <= val_max:
            points.append(tuple(combo) + (last_val,))
    return np.array(points)


def generate_balanced_nps_from_axes(
    base_np: pd.Series,
    x_axis: tuple[str, str],
    y_axis: tuple[str, str],
    val_max: float,
    step: float,
    center_np_shift: bool,
) -> list[pd.Series]:
    """
    Generate net position vectors by varying zones along x_axis/y_axis,
    ensuring zero-sum across zones.

    Returns
    -------
    list of pd.Series
    """
    varying_zones = sorted(set(x_axis + y_axis))
    fixed_zones = [z for z in base_np.index if z not in varying_zones]

    grid_vals = np.arange(-val_max, val_max + step, step)
    base_combos = product(grid_vals, repeat=len(varying_zones) - 1)

    result = []
    for combo in base_combos:
        combo = np.array(combo)
        last_val = -(sum(combo) + sum(base_np[fixed_zones]))
        all_vals = list(combo) + [last_val]

        new_np = base_np.copy()
        for z, v in zip(varying_zones, all_vals):
            new_np[z] = v
        result.append(new_np[base_np.index])

    return np.array(result)


# -------------------------------------------------------------------
# FBdomainPlot class
# -------------------------------------------------------------------

class FBdomainPlot:
    """
    Flow-Based Domain plotting class.

    Provides methods for:
    - Creating 2D domain representations (from FB constraints A, b)
    - Generating feasible NP points via mesh validation
    - Plotting interactive (Plotly) or publication-quality figures
    """

    def __init__(self, scenario, x_axis, y_axis, result_name=None, title='', np_shift=None):
        """
        Parameters
        ----------
        scenario : Scenario
            Scenario object containing network, results, fb_parameters, etc.
        x_axis, y_axis : tuple(str, str)
            Zone pair defining the x and y exchanges.
        result_name : str, optional
            Key in scenario.results for the result to overlay.
        title : str
            Plot title.
        np_shift : pd.Series, optional
            Net position shift vector. If None, taken from result.NP.
        """
        self.scenario = scenario
        self.title = title
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.np_shift = pd.Series(np_shift, index=scenario.network.Z, name="NP_shift")
        self.domains = scenario.fb_parameters

        # Plot state
        self.xlims = None
        self.ylims = None
        self.mcp = None
        self.npf = scenario.npf
        self.mesh_points = None

        # Attach result if available
        try:
            self.result = scenario.results[result_name]
        except KeyError:
            self.result = None
    
        # Store 2D domain reductions
        self.domains_2d = {k: None for k in ['initial', 'minRAM', 'iva', 'a', 'amax']}
        
        # Default to MCP NP if no np_shift provided
        if isinstance(self.result, Result):
            self.mcp = self.result.NP
            if np_shift is None:
                self.np_shift = self.result.NP

    def create_mesh_points(self, np_max=200, step=10, forecast_day='d0', center_np_shift=False):
        """
        Generate candidate NP mesh points, validate feasibility, and store results.

        Parameters
        ----------
        np_max : int
            Maximum absolute NP value in grid.
        step : int
            Step size for NP grid.
        forecast_day : str
            RES forecast label (e.g. 'd0').
        center_np_shift : bool
            If True, center mesh at np_shift.

        Stores
        ------
        self.mesh_points : pd.DataFrame
            Columns: NPs, EX, amax_inf, a_inf, ex_x, ex_y, valid
        """
        # Candidate grid of NP points (restricted to x/y axes)
        mesh_points = generate_balanced_nps_from_axes(
            self.np_shift, self.x_axis, self.y_axis, np_max, step, center_np_shift
        )
        print(f'{len(mesh_points)} points created.')
        
        # Keep exchanges fixed for zones not in x/y
        constraints = {}
        domain_exchange = [self.x_axis, self.x_axis[::-1], self.y_axis, self.y_axis[::-1]]
        ex = compute_sec(self.scenario.network.zone_border, self.np_shift.values)
        for e in self.result.EX.index:
            if e not in domain_exchange:
                constraints[e] = ('==', self.result.EX[e])   
                
        # Validate feasibility of each NP candidate
        validated_points = []
        for point in tqdm(mesh_points, desc='Validating'):
            try:
                ex = compute_sec(self.scenario.network.zone_border, point, exchange_constraints=constraints)
            except Exception:
                continue

            # Project to x/y coordinates
            ex_x = ex[self.x_axis] - ex[self.x_axis[::-1]]
            ex_y = ex[self.y_axis] - ex[self.y_axis[::-1]]

            # Check feasibility (with/without maxA constraints)
            amax_val = feasibility_validation(self, np.round(point, 3), modus='amax', forecast_day=forecast_day)
            a_val = feasibility_validation(self, np.round(point, 3), modus='a', forecast_day=forecast_day)

            validated_points.append([point, ex, amax_val, a_val, ex_x, ex_y]) 
            
        # Collect into dataframe
        points = pd.DataFrame(validated_points, columns=['NPs', 'EX', 'amax_inf','a_inf', 'ex_x', 'ex_y'])
        points['valid'] = ~points['a_inf']
        
        print(f'{len(points)} points validated.')
        self.mesh_points = points

    def create_domain_representation(self, x_axis, y_axis, A, b, np_shift):
        """
        Convert high-dim FB constraints to 2D representation.

        Parameters
        ----------
        x_axis, y_axis : tuple(str, str)
            Zone exchange axes.
        A, b : np.ndarray
            FB domain constraint matrices.
        np_shift : pd.Series
            NP shift vector.

        Returns
        -------
        A_2d : np.ndarray
            Reduced constraint matrix (2D).
        b_shift : np.ndarray
            Shifted RHS.
        """
        # Project A onto selected axis pairs
        domain_idx = [[self.scenario.network.Z.index(z0), self.scenario.network.Z.index(z1)]
                      for z0, z1 in [x_axis, y_axis]]
        A_2d = np.vstack([A[:, i] - A[:, j] for i, j in domain_idx]).T
        
        # Apply NP shift (fix exchanges for other zones)
        b_shift = b.copy()
        if np_shift is not None:
            exchange = compute_sec(self.scenario.network.zone_border, np_shift.values)
            domain_exchange = [x_axis, x_axis[::-1], y_axis, y_axis[::-1]]
            for e in exchange.index:
                if e not in domain_exchange:
                    b_shift -= exchange[e] * (A[:, self.scenario.network.Z.index(e[0])]
                                              - A[:, self.scenario.network.Z.index(e[1])])
                    b_shift = np.maximum(b_shift, 0)
        return A_2d, b_shift
    
    def add_domain_to_fig(self, fig, A_2d, b, domain, color, name):
        """
        Add feasible region and constraints as Plotly traces.

        Returns
        -------
        xlims, ylims : list
            Extent of feasible polygon.
        """
        # Compute feasible vertices
        feasible_vertices = np.array(compute_polytope_vertices(A_2d, b))
        if feasible_vertices.shape[0] >= 3:
            hull = spatial.ConvexHull(feasible_vertices)
            fig.add_trace(go.Scatter(
                x=feasible_vertices[hull.vertices, 0],
                y=feasible_vertices[hull.vertices, 1],
                fill='toself', fillcolor=color, opacity=0.5,
                legendgroup=name, legendgrouptitle={'text': name},
                name=f"Feasible Region {name}", mode='none'
            ))
        
        # Add constraint lines
        x_vals = np.linspace(-4*max(b), 4*max(b), 100)
        for i in range(len(A_2d)):
            text = f"{self.domains[domain].iloc[i]['CB']}_{self.domains[domain].iloc[i]['DIR']}"
            if A_2d[i, 1] != 0:  # Non-vertical line
                y_vals = (b[i] - A_2d[i, 0] * x_vals) / A_2d[i, 1]
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode='lines',
                    line=dict(color=color), opacity=0.3,
                    text=text, name=text,
                    legendgroup=name
                ))
            elif A_2d[i, 0] != 0:  # Vertical line
                x_fixed = b[i] / A_2d[i, 0]
                fig.add_trace(go.Scatter(
                    x=[x_fixed, x_fixed], y=[-1000, 1000],
                    mode='lines', line=dict(color=color), opacity=0.3,
                    text=text, name=text,
                    legendgroup=name
                ))
        
        return [feasible_vertices[:, 0].min(), feasible_vertices[:, 0].max()], \
               [feasible_vertices[:, 1].min(), feasible_vertices[:, 1].max()]

    def figure(self, title=None, domains=['initial', 'minram', 'iva'], show_figure=True):
        """
        Build and display an interactive Plotly FB domain figure.

        Parameters
        ----------
        title : str
            Plot title.
        domains : list
            Which domains to overlay (subset of ['initial','minram','iva','amax','a']).
        show_figure : bool
            If True, call fig.show().
        """
        colors = {'initial': 'grey', 'minram': 'lightblue',
                  'iva': 'red', 'amax':'green', 'a': 'yellow'}
        
        if title is None:
            title = self.title
        
        fig = go.Figure()  
        x_vals, y_vals = [], []
        
        # Plot each domain
        for domain in domains:
            if self.domains[domain] is None:
                continue
            A = np.array(self.domains[domain][self.scenario.network.Z])
            b = np.array(self.domains[domain]['RAM'])
            A_2d, b_shift = self.create_domain_representation(self.x_axis, self.y_axis, A, b, self.np_shift)
            xs, ys = self.add_domain_to_fig(fig, A_2d, b_shift, domain, color=colors[domain], name=domain)
            x_vals.extend(xs); y_vals.extend(ys)
            
        # Mark NPF
        ex = compute_sec(self.scenario.network.zone_border, self.npf.values)
        exchange_point = (ex[self.x_axis] - ex[self.x_axis[::-1]],
                          ex[self.y_axis] - ex[self.y_axis[::-1]])
        fig.add_trace(go.Scatter(
            x=[exchange_point[0]], y=[exchange_point[1]],
            mode='markers', marker=dict(color='black', size=15, symbol='x'),
            name='Exchanges @ NPF'
        ))
        
        # Mark MCP
        if isinstance(self.result, Result):
            exchange_point = (self.result.EX[self.x_axis] - self.result.EX[self.x_axis[::-1]],
                              self.result.EX[self.y_axis] - self.result.EX[self.y_axis[::-1]])
            fig.add_trace(go.Scatter(
                x=[exchange_point[0]], y=[exchange_point[1]],
                mode='markers', marker=dict(color='red', size=15, symbol='x'),
                name='Exchanges @ MCP'
            ))

        # Add feasibility mesh if available
        if self.mesh_points is not None:
            for column, color in zip(['a_inf','amax_inf','valid'], ['orange', 'yellow', 'green']):
                fig.add_trace(go.Scatter(
                    x=self.mesh_points.loc[self.mesh_points[column], 'ex_x'],
                    y=self.mesh_points.loc[self.mesh_points[column], 'ex_y'],
                    mode='markers', marker=dict(color=color, size=8)
                ))
        
        # Set axis limits if unset
        if self.xlims is None:
            xmin, xmax = min(x_vals), max(x_vals)
            self.xlims = [xmin - 0.5*abs(xmin), xmax + 0.5*abs(xmax)]
        if self.ylims is None:
            ymin, ymax = min(y_vals), max(y_vals)
            self.ylims = [ymin - 0.5*abs(ymin), ymax + 0.5*abs(ymax)]
        
        fig.update_layout(
            xaxis=dict(range=self.xlims),
            yaxis=dict(range=self.ylims),
            title=f"Flow-Based Domain: {title}",
            xaxis_title=f"Exchange Zone {self.x_axis[0]} → Zone {self.x_axis[1]}",
            yaxis_title=f"Exchange Zone {self.y_axis[0]} → Zone {self.y_axis[1]}",
            showlegend=True,
        )    
        fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')

        if show_figure:
            fig.show()
        else:
            return fig

    def add_domain_to_fig_pub(self, fig, A_2d, b, domain, color, dash, width, name):
        """
        Add domain constraints & feasible polygon (publication style).

        Parameters
        ----------
        fig : go.Figure
            Plotly figure to add traces to.
        A_2d, b : np.ndarray
            Reduced constraints for 2D projection.
        domain : str
            Domain key (e.g., 'initial', 'minram', 'iva').
        color : str
            Line color.
        dash : str
            Line dash style (e.g., 'solid', 'dash').
        width : int
            Line width.
        name : str
            Label for traces.

        Returns
        -------
        xlims, ylims : list
            Extents of feasible polygon.
        """
        feasible_vertices = np.array(compute_polytope_vertices(A_2d, b))
        if feasible_vertices.shape[0] >= 3:
            hull = spatial.ConvexHull(feasible_vertices)
            # Ensure closed polygon by repeating first point
            x = np.append(feasible_vertices[hull.vertices, 0],
                          feasible_vertices[hull.vertices[0], 0])
            y = np.append(feasible_vertices[hull.vertices, 1],
                          feasible_vertices[hull.vertices[0], 1])
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='lines',
                line=dict(color='black', width=width, dash=dash),
                legendgroup=name, legendgrouptitle={'text': name},
                name=f"Feasible Region {name}"
            ))
        
        # Add individual constraint lines (faded)
        x_vals = np.linspace(-4*max(b), 4*max(b), 100)
        for i in range(len(A_2d)):
            text = f"{self.domains[domain].iloc[i]['CB']}_{self.domains[domain].iloc[i]['DIR']}"
            if A_2d[i, 1] != 0:  # Standard line
                y_vals = (b[i] - A_2d[i, 0] * x_vals) / A_2d[i, 1]
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode='lines',
                    line=dict(color=color, dash=dash, width=2), opacity=0.3,
                    text=text, name=text, legendgroup=name
                ))
            elif A_2d[i, 0] != 0:  # Vertical line
                x_fixed = b[i] / A_2d[i, 0]
                fig.add_trace(go.Scatter(
                    x=[x_fixed, x_fixed], y=[-1000, 1000],
                    mode='lines', line=dict(color=color, dash=dash, width=2), opacity=0.3,
                    text=text, name=text, legendgroup=name
                ))
        
        return [feasible_vertices[:, 0].min(), feasible_vertices[:, 0].max()], \
               [feasible_vertices[:, 1].min(), feasible_vertices[:, 1].max()]


    def figure_pub(self, title=None, domains=['initial', 'minram','iva'], highlight=['iva'],
                    show_figure=True, show_legend=False):
        """
        Alternative publication-quality FB domain figure.

        Adds concave hull feasibility regions, domain outlines,
        and markers for NPF and MCP.

        Parameters
        ----------
        title : str
            Plot title.
        domains : list
            Domains to include.
        highlight : list
            Domains to emphasize.
        show_figure : bool
            Whether to display.
        show_legend : bool
            Whether to display legend.

        Returns
        -------
        go.Figure or None
        """
        # Default styles
        colors = {d: 'lightgrey' for d in domains}
        dash = {d: 'dot' if d != 'initial' else 'dash' for d in domains}
        width = {d: 4 for d in domains}

        # Highlighted domain(s)
        for domain in highlight:
            colors[domain] = 'black'
            width[domain] = 6
            dash[domain] = 'solid'
        
        if title is None:
            title = self.title
        
        fig = go.Figure()  

        # Feasibility regions (concave hulls)
        if self.mesh_points is not None:
            self.mesh_points.loc[self.mesh_points.amax_inf==True, 'a_inf'] = False
            for column, color in zip(['amax_inf','a_inf','valid'], ['#f7f6ba', 'red', 'green']):
                x = self.mesh_points.loc[self.mesh_points[column], 'ex_x']
                y = self.mesh_points.loc[self.mesh_points[column], 'ex_y']
                points = np.array([x, y]).T
                if len(points) >= 3:
                    hull = concave_hull_indexes(points, concavity=4)
                    hull = np.r_[hull, hull[0]]  # close loop
                    # Filled region
                    fig.add_trace(go.Scatter(
                        x=points[hull, 0], y=points[hull, 1],
                        fill='toself', fillcolor=color, opacity=0.3,
                        mode='lines', line=dict(color='rgba(0,0,0,0)'),
                        hoverinfo='skip', showlegend=show_legend
                    ))
                    # Outline
                    fig.add_trace(go.Scatter(
                        x=points[hull, 0], y=points[hull, 1],
                        mode='lines', line=dict(color=color, width=2),
                        showlegend=False
                    ))
        
        # Domain overlays
        x_vals, y_vals = [], []
        for domain in domains:
            if self.domains[domain] is None:
                continue
            A = np.array(self.domains[domain][self.scenario.network.Z])
            b = np.array(self.domains[domain]['RAM'])
            A_2d, b_shift = self.create_domain_representation(self.x_axis, self.y_axis, A, b, self.np_shift)
            xs, ys = self.add_domain_to_fig_pub(fig, A_2d, b_shift, domain,
                                                color=colors[domain], dash=dash[domain],
                                                width=width[domain], name=domain)
            x_vals.extend(xs); y_vals.extend(ys)
        
        # NPF marker
        ex = compute_sec(self.scenario.network.zone_border, self.npf.values)
        exchange_point = (ex[self.x_axis] - ex[self.x_axis[::-1]],
                          ex[self.y_axis] - ex[self.y_axis[::-1]])
        fig.add_trace(go.Scatter(
            x=[exchange_point[0]], y=[exchange_point[1]], mode='markers',
            marker=dict(symbol='square-x', size=24, color='white',
                        line=dict(color='black', width=3)),
            name='Exchanges @ NPF'
        ))
        
        # MCP marker
        if isinstance(self.result, Result):
            exchange_point = (self.result.EX[self.x_axis] - self.result.EX[self.x_axis[::-1]],
                              self.result.EX[self.y_axis] - self.result.EX[self.y_axis[::-1]])
            fig.add_trace(go.Scatter(
                x=[exchange_point[0]], y=[exchange_point[1]], mode='markers',
                marker=dict(symbol='circle-x', size=24, color='white',
                            line=dict(color='red', width=3)),
                name='Exchanges @ MCP'
            ))
        
        # Axis ranges
        if self.xlims is None:
            xmin, xmax = min(x_vals), max(x_vals)
            self.xlims = [xmin - 0.5*abs(xmin), xmax + 0.5*abs(xmax)]
        if self.ylims is None:
            ymin, ymax = min(y_vals), max(y_vals)
            self.ylims = [ymin - 0.5*abs(ymin), ymax + 0.5*abs(ymax)]
        
        # Layout
        fig.update_layout(
            template="simple_white", width=1000, height=1000,
            margin=dict(l=100, r=100, t=100, b=100),
            font=dict(family="serif", size=45),
            showlegend=show_legend,
            xaxis=dict(range=self.xlims),
            yaxis=dict(range=self.ylims),
            xaxis_title=f"Exchange Zone {self.x_axis[0]} → Zone {self.x_axis[1]}",
            yaxis_title=f"Exchange Zone {self.y_axis[0]} → Zone {self.y_axis[1]}"
        )
        fig.update_xaxes(scaleanchor="y", scaleratio=1,
                         ticks="outside", showline=True, mirror=True,
                         linewidth=2, linecolor="black",
                         zeroline=True, zerolinewidth=2, zerolinecolor="black",
                         showgrid=True, gridcolor="lightgrey", gridwidth=1, griddash="dash")
        fig.update_yaxes(scaleanchor="x", scaleratio=1,
                         ticks="outside", showline=True, mirror=True,
                         linewidth=2, linecolor="black",
                         zeroline=True, zerolinewidth=2, zerolinecolor="black",
                         showgrid=True, gridcolor="lightgrey", gridwidth=1, griddash="dash")

        if show_figure:
            fig.show()
        else:
            return fig

