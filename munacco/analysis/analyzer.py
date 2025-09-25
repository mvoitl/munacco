"""
munacco.analysis.analyzer

Provides the `Analyzer` class for exploratory and comparative
analysis of CACM scenarios. Includes utilities to:
- Inspect KPI distributions (histograms, boxplots, violins)
- Compare KPIs (scatter matrix, correlations)
- Visualize flow-based domains and MCP exchange points

Plots use Plotly and are thesis-ready (serif fonts, large sizes).
"""

from __future__ import annotations

import copy
import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.spatial as spatial

from munacco.analysis.visualization import FBdomainPlot
from munacco.tools import compute_polytope_vertices

logger = logging.getLogger(__name__)


class Analyzer:
    """
    Analyze and compare results across multiple scenarios using Plotly.

    Parameters
    ----------
    scenarios : list
        List of scenario objects (with KPIs and results).
    metadata : dict, optional
        Optional metadata for labeling, config, etc.
    """

    def __init__(self, scenarios, metadata: Optional[dict] = None):
        self.scenarios = copy.deepcopy(scenarios)
        self.df = pd.DataFrame([s.kpis for s in scenarios])
        self.err_flows = self.compute_err_flows(scenarios)
        self.metadata = metadata or {}

    # ------------------------------------------------------------------
    # Error flow computation
    # ------------------------------------------------------------------

    def compute_err_flows(self, scenarios) -> pd.DataFrame:
        """
        Compute post-error flows for all scenarios.

        Returns
        -------
        pd.DataFrame
            Error flows (lines × scenarios).
        """
        post_error_flows = []
        for s in scenarios:
            err = s.res_forecast["p_d0"] - s.res_forecast["p_d2"]
            I_err = (
                s.network.map_nres @ err
                - s.network.map_np @ s.network.map_palpha @ s.alpha * sum(err)
            )
            post_error_flows.append(s.network.ptdf @ I_err)

        return pd.DataFrame(columns=s.network.L, data=post_error_flows)

    # ------------------------------------------------------------------
    # KPI plots
    # ------------------------------------------------------------------

    def histogram(self, kpi: str, save_path: Optional[str] = None):
        """Plot histogram of a KPI across scenarios."""
        if kpi not in self.df:
            logger.warning(f"KPI {kpi} not found in dataframe.")
            return

        fig = go.Figure(go.Histogram(x=self.df[kpi], nbinsx=50))
        fig.update_layout(
            title=f"Distribution of {kpi}",
            xaxis_title=kpi,
            yaxis_title="Frequency",
            template="simple_white",
            font=dict(family="Serif", size=30),
        )
        self._show_or_save(fig, save_path)

    def boxplot(self, kpis: Sequence[str], save_path: Optional[str] = None):
        """Boxplot of selected KPIs."""
        fig = go.Figure()
        for kpi in kpis:
            if kpi not in self.df:
                logger.warning(f"KPI {kpi} not found.")
                continue
            fig.add_trace(go.Box(y=self.df[kpi], name=kpi, boxmean=True))

        fig.update_layout(
            title="Distribution of Key KPIs",
            template="simple_white",
            font=dict(family="Serif", size=30),
        )
        self._show_or_save(fig, save_path)

    def violin(self, kpi: str, save_path: Optional[str] = None):
        """Violin plot for a single KPI."""
        if kpi not in self.df:
            logger.warning(f"KPI {kpi} not found.")
            return

        fig = go.Figure(
            go.Violin(
                y=self.df[kpi],
                name=kpi,
                box_visible=True,
                meanline_visible=True,
                points="all",
                pointpos=0,
            )
        )
        fig.update_layout(
            title=f"Distribution of {kpi}",
            template="simple_white",
            font=dict(family="Serif", size=30),
        )
        self._show_or_save(fig, save_path)

    def pairplot(self, kpis: Sequence[str], save_path: Optional[str] = None):
        """Scatter matrix for selected KPIs."""
        fig = px.scatter_matrix(
            self.df[kpis],
            dimensions=kpis,
            title="Pairwise KPI Plot",
            template="simple_white",
        )
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(font=dict(family="Serif", size=25))
        self._show_or_save(fig, save_path)

    def correlation_heatmap(self, save_path: Optional[str] = None):
        """Heatmap of KPI correlation matrix."""
        corr = self.df.corr()
        fig = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation"),
            )
        )
        fig.update_layout(
            title="KPI Correlation Matrix",
            template="simple_white",
            font=dict(family="Serif", size=25),
        )
        self._show_or_save(fig, save_path)

    # ------------------------------------------------------------------
    # Flow-based domain plots
    # ------------------------------------------------------------------

    def plot_fb_domains(
        self,
        x_axis: tuple[str, str],
        y_axis: tuple[str, str],
        result_name: str,
        domain: str = "minram",
        save_path: Optional[str] = None,
    ):
        """
        Plot flow-based domains and MCP exchange points for all scenarios.

        Parameters
        ----------
        x_axis, y_axis : tuple
            Zone pairs for exchange axes (e.g. ("DE", "FR")).
        result_name : str
            Name of result to highlight (e.g. "market_coupling").
        domain : str, default="minram"
            Domain type to visualize.
        save_path : str, optional
            Path to save figure instead of interactive display.
        """
        fig = go.Figure()

        for sc in self.scenarios:
            plot = FBdomainPlot(sc, x_axis, y_axis, result_name=result_name)
            A = np.array(sc.fb_parameters[domain][sc.network.Z])
            b = np.array(sc.fb_parameters[domain]["RAM"])
            A_2d, _ = plot.create_domain_representation(
                plot.x_axis, plot.y_axis, A, b, plot.np_shift
            )

            color = "red" if sc.id == "reference" else "grey"
            op = 1 if sc.id == "reference" else 0.3

            # Convex hull of feasible region
            feasible_vertices = np.array(compute_polytope_vertices(A_2d, b))
            if feasible_vertices.shape[0] >= 3:
                hull = spatial.ConvexHull(feasible_vertices)
                hull_idx = list(hull.vertices) + [hull.vertices[0]]
                fig.add_trace(
                    go.Scatter(
                        x=feasible_vertices[hull_idx, 0],
                        y=feasible_vertices[hull_idx, 1],
                        mode="lines",
                        opacity=op,
                        line=dict(color=color),
                        name=f"Feasible {sc.id}",
                    )
                )

            # Market coupling point
            ex_x = sc.results["market_coupling"].EX[x_axis[0]] - sc.results[
                "market_coupling"
            ].EX[x_axis[1]]
            ex_y = sc.results["market_coupling"].EX[y_axis[0]] - sc.results[
                "market_coupling"
            ].EX[y_axis[1]]
            fig.add_trace(
                go.Scatter(
                    x=[ex_x],
                    y=[ex_y],
                    mode="markers",
                    marker=dict(color=color, size=15, symbol="circle-dot", opacity=op),
                    name=f"MCP {sc.id}",
                )
            )

        fig.update_layout(
            xaxis_title=f"Exchange {x_axis[0]} → {x_axis[1]}",
            yaxis_title=f"Exchange {y_axis[0]} → {y_axis[1]}",
            showlegend=True,
            font=dict(family="Serif", size=25),
        )
        fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black")
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black")

        self._show_or_save(fig, save_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _show_or_save(self, fig, save_path: Optional[str] = None):
        """Show interactive plot or save to file if save_path is given."""
        if save_path:
            fig.write_image(save_path)
            logger.info(f"Saved figure to {save_path}")
        else:
            fig.show()
