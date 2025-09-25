"""
munacco.analysis.inspector

Provides the `ScenarioInspector` class for deep-dive analysis of
a single scenario:
- Print KPIs
- Visualize network (matplotlib or Plotly)
- Inspect flow-based domains
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np
import plotly.graph_objects as go

from munacco.analysis.visualization import FBdomainPlot

logger = logging.getLogger(__name__)


class ScenarioInspector:
    """
    Inspect a single scenario: KPIs, network state, flow-based domains.
    """

    def __init__(self, scenario):
        self.scenario = scenario

    # ------------------------------------------------------------------
    # KPI inspection
    # ------------------------------------------------------------------

    def print_kpis(self):
        """Print all KPIs for the scenario."""
        print("\n--- Scenario KPIs ---")
        for k, v in self.scenario.kpis.items():
            print(f"{k}: {v}")

    # ------------------------------------------------------------------
    # Network plots
    # ------------------------------------------------------------------

    def plot_network(self, result_name: str, save_path: Optional[str] = None):
        """
        Plot a static schematic of the network using matplotlib.

        Parameters
        ----------
        result_name : str
            Name of result in scenario (e.g. "market_coupling").
        save_path : str, optional
            If given, saves the figure instead of showing interactively.
        """
        result = self.scenario.results[result_name]

        # Create directed graph
        G = nx.DiGraph()
        flows = result.F[result.F >= 0]
        for line in flows.index:
            G.add_edge(
                result.network.nes.loc[line, "node_i"],
                result.network.nes.loc[line, "node_j"],
                flow=result.F[line],
                fmax=result.network.nes.loc[line, "f_max"],
                name=result.network.nes.loc[line, "name"],
            )

        pos = {
            node: (result.network.nodes.loc[node, "x"], result.network.nodes.loc[node, "y"])
            for node in result.network.nodes.index
        }

        # Zone coloring
        cmap = plt.get_cmap("Pastel1")
        zone_colors = {
            zone: cmap(i / len(result.network.Z)) for i, zone in enumerate(result.network.Z)
        }

        # Figure
        span_x = max(result.network.nodes.x) - min(result.network.nodes.x) + 1
        span_y = max(result.network.nodes.y) - min(result.network.nodes.y) + 1
        fig, ax = plt.subplots(figsize=(span_x * 3, span_y * 2))

        # Zone rectangles
        for zone in result.network.Z:
            coords = np.array(
                [
                    pos[n]
                    for n in result.network.nodes.index
                    if result.network.nodes.loc[n, "zone"] == zone
                ]
            )
            x_min, y_min = coords.min(axis=0) - 0.3
            x_max, y_max = coords.max(axis=0) + 0.3
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                color=zone_colors[zone],
                alpha=0.6,
                linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(
                x_min,
                y_max,
                f"{zone} Net: {result.NP.loc[zone]:.2f}",
                fontsize=10,
                ha="left",
                color="black",
            )

        # Nodes
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=250,
            node_color="tab:grey",
            font_size=8,
            ax=ax,
        )

        # Edge labels
        edge_labels = {
            (n1, n2): f"{d['name']}\n{d['flow']:.1f} ({d['fmax']})"
            for n1, n2, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

        ax.set_title(f"Electricity Network ({result.config['name']})", fontsize=14)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            logger.info(f"Saved static network plot to {save_path}")
        else:
            plt.show()

    def plot_network_plotly(self, result_name: str, save_path: Optional[str] = None):
        """
        Plot interactive network schematic with Plotly.

        Parameters
        ----------
        result_name : str
            Name of result (e.g. "market_coupling").
        save_path : str, optional
            Path to save figure instead of showing.
        """
        result = self.scenario.results[result_name]

        G = nx.DiGraph()
        flows = result.F[result.F >= 0]
        for line in flows.index:
            G.add_edge(
                result.network.nes.loc[line, "node_i"],
                result.network.nes.loc[line, "node_j"],
                flow=result.F[line],
                fmax=result.network.nes.loc[line, "f_max"],
                name=result.network.nes.loc[line, "name"],
            )

        pos = {
            node: (result.network.nodes.loc[node, "x"], result.network.nodes.loc[node, "y"])
            for node in result.network.nodes.index
        }

        # Zone colors (RGBA strings for Plotly)
        cmap = plt.get_cmap("Pastel1")
        zone_colors = {}
        for i, zone in enumerate(result.network.Z):
            r, g, b, _ = cmap(i / len(result.network.Z))
            zone_colors[zone] = f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.6)"

        fig = go.Figure()

        # Zone rectangles
        for zone in result.network.Z:
            coords = np.array(
                [
                    pos[n]
                    for n in result.network.nodes.index
                    if result.network.nodes.loc[n, "zone"] == zone
                ]
            )
            x_min, y_min = coords.min(axis=0) - 0.3
            x_max, y_max = coords.max(axis=0) + 0.3
            fig.add_shape(
                type="rect",
                x0=x_min,
                y0=y_min,
                x1=x_max,
                y1=y_max,
                line=dict(color="black", width=1),
                fillcolor=zone_colors[zone],
            )

            fig.add_trace(
                go.Scatter(
                    x=[x_min],
                    y=[y_max],
                    text=[f"{zone} Net: {result.NP.loc[zone]:.2f}"],
                    mode="text",
                    textfont=dict(size=12, color="black"),
                    showlegend=False,
                )
            )

        # Edges
        for n1, n2, d in G.edges(data=True):
            x0, y0 = pos[n1]
            x1, y1 = pos[n2]
            color = "red" if d["flow"] > d["fmax"] else "black"
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=3, color=color),
                    hovertext=f"Line {d['name']}<br>Flow: {d['flow']:.1f}/{d['fmax']}",
                    hoverinfo="text",
                    showlegend=False,
                )
            )

        # Nodes
        fig.add_trace(
            go.Scatter(
                x=[x for x, _ in pos.values()],
                y=[y for _, y in pos.values()],
                mode="markers+text",
                marker=dict(size=12, color="grey"),
                text=list(pos.keys()),
                textposition="top center",
                hoverinfo="skip",
            )
        )

        fig.update_layout(
            title=f"Electricity Network ({result.config['name']})",
            template="simple_white",
            font=dict(family="Serif", size=20),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        )

        if save_path:
            fig.write_image(save_path)
            logger.info(f"Saved interactive network plot to {save_path}")
        else:
            fig.show()

    # ------------------------------------------------------------------
    # Domain visualization
    # ------------------------------------------------------------------

    def create_domain_plot(
        self,
        x_axis: tuple[str, str],
        y_axis: tuple[str, str],
        result_name: str,
        title: str = "Domain Plot",
        np_shift=None,
        show: bool = False,
    ):
        """
        Create a flow-based domain plot wrapper for this scenario.
        """
        plot = FBdomainPlot(
            self.scenario, x_axis, y_axis, result_name=result_name, title=title, np_shift=np_shift
        )
        if show:
            plot.figure(domains=["initial", "minram", "iva"], show_figure=True)
        return plot

    # ------------------------------------------------------------------
    # Dispatch plots
    # ------------------------------------------------------------------

    def plot_dispatch(self):
        """Placeholder for dispatch plotting."""
        raise NotImplementedError("plot_dispatch is not yet implemented.")
