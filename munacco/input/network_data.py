"""
munacco.input.network_data

Defines the `NetworkData` class, which stores all network-related
data (nodes, lines, plants, RES, zones) and constructs derived
structures such as PTDF matrices, mapping matrices, and zonal
aggregates.

This is the central data container for the CACM model.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NetworkData:
    """
    Container for network data (nodes, lines, plants, RES, zones).

    On initialization, computes:
    - Index sets (nodes, lines, zones, plants, RES, alpha-plants)
    - Mapping matrices (node-plant, node-RES, node-zone, plant-alpha)
    - Extended network elements (NES, with direct/opposite directions)
    - PTDF matrices
    - Zonal aggregates (demand, g_max, etc.)
    - RES forecasts (d2, d1, d0)
    - Zone border matrix

    Parameters
    ----------
    nodes : pd.DataFrame
        Node-level data with at least `Pd`, `zone`, `slack`.
    lines : pd.DataFrame
        Line data with `node_i`, `node_j`, `b`, `f_max`.
    plants : pd.DataFrame
        Plant data with `g_max`, `mc`, `bus`, `alpha`.
    res : pd.DataFrame
        Renewable generators with `g_max`, `p_max_pu`, `mc`, `bus`, `sigma`, `RD`.
    zones : pd.DataFrame
        Zone-level data (will be augmented with n_nodes, demand, g_max, ...).
    """

    def __init__(
        self,
        nodes: pd.DataFrame,
        lines: pd.DataFrame,
        plants: pd.DataFrame,
        res: pd.DataFrame,
        zones: pd.DataFrame,
    ):
        self.nodes = nodes
        self.lines = lines
        self.plants = plants
        self.res = res
        self.zones = zones

        # Derived structures
        self.nes: pd.DataFrame | None = None
        self.zone_border: pd.DataFrame | None = None

        # Index sets
        self.Z: List[str] | None = None
        self.N: List[str] | None = None
        self.L: List[str] | None = None
        self.P: List[str] | None = None
        self.RES: List[str] | None = None
        self.A: List[str] | None = None

        # Mapping matrices
        self.map_np: np.ndarray | None = None
        self.map_nres: np.ndarray | None = None
        self.map_nz: np.ndarray | None = None
        self.map_palpha: np.ndarray | None = None

        # PTDF
        self.ptdf: np.ndarray | None = None
        self.ptdf_nes: np.ndarray | None = None

        try:
            self.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize NetworkData: {e}")
            raise

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize sets, mappings, PTDF, forecasts, and zonal info."""
        self.create_sets()
        self.create_mapping()
        self.seperate_network_elements()
        self.create_zonal_info()
        self.create_res_forecast()
        self.ptdf = self.create_ptdf_matrix()
        self.ptdf_nes = np.vstack([self.ptdf, -self.ptdf])
        self.zone_border = self.create_zone_border()

    def deepcopy(self) -> "NetworkData":
        """Return a deep copy of the network data."""
        return NetworkData(
            nodes=self.nodes.copy(),
            lines=self.lines.copy(),
            plants=self.plants.copy(),
            res=self.res.copy(),
            zones=self.zones.copy(),
        )

    # ------------------------------------------------------------------
    # Sets and mappings
    # ------------------------------------------------------------------

    def create_sets(self) -> None:
        """Create index sets for nodes, lines, zones, plants, RES, and alpha plants."""
        self.P = list(self.plants.index)
        self.N = list(self.nodes.index)
        self.L = list(self.lines.index)
        self.Z = list(self.zones.index)
        self.RES = list(self.res.index)
        self.A = list(self.plants[self.plants.alpha].index)

    def create_mapping(self) -> None:
        """Create mapping matrices between nodes, plants, zones, and RES."""
        self.map_np = np.zeros((len(self.N), len(self.P)))
        for p in self.plants.index:
            self.map_np[self.N.index(self.plants.loc[p, "bus"]), self.P.index(p)] = 1

        self.map_nres = np.zeros((len(self.N), len(self.RES)))
        for r in self.res.index:
            self.map_nres[self.N.index(self.res.loc[r, "bus"]), self.RES.index(r)] = 1

        self.map_nz = np.zeros((len(self.N), len(self.Z)))
        for n in self.nodes.index:
            self.map_nz[self.N.index(n), self.Z.index(self.nodes.loc[n, "zone"])] = 1

        self.map_palpha = np.zeros((len(self.plants), sum(self.plants.alpha)))
        for p in self.plants[self.plants.alpha].index:
            self.map_palpha[self.P.index(p), self.A.index(p)] = 1

    def seperate_network_elements(self) -> None:
        """Expand lines into directed network elements (NES)."""
        nes1 = self.lines.copy()
        nes1["dir"] = "DIRECT"
        nes2 = nes1.copy()
        nes2["dir"] = "OPPOSITE"
        nes2["node_i"] = nes1["node_j"]
        nes2["node_j"] = nes1["node_i"]
        self.nes = pd.concat([nes1, nes2], ignore_index=True)

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def create_zonal_info(self) -> None:
        """Compute zonal aggregates (nodes, g_max, demand, etc.)."""
        self.zones["n_nodes"] = [sum(n) for n in self.map_nz.T]
        self.zones["g_max"] = self.map_np @ self.plants.g_max @ self.map_nz
        self.zones["g_max_w_res"] = (
            self.map_nres @ (self.res.g_max * self.res.p_max_pu) @ self.map_nz
            + self.zones["g_max"]
        )
        self.zones["demand"] = self.nodes.Pd @ self.map_nz
        self.nodes["g_max"] = self.map_np @ self.plants.g_max

    # ------------------------------------------------------------------
    # PTDF
    # ------------------------------------------------------------------

    def create_incidence_matrix(self) -> np.ndarray:
        """Create incidence matrix (lines × nodes)."""
        incidence = np.zeros((len(self.lines), len(self.nodes)))
        for i, elem in enumerate(self.lines.index):
            incidence[i, self.nodes.index.get_loc(self.lines.loc[elem, "node_i"])] = 1
            incidence[i, self.nodes.index.get_loc(self.lines.loc[elem, "node_j"])] = -1
        return incidence

    def create_susceptance_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Create line and node susceptance matrices."""
        susceptance_vector = self.lines.b
        incidence = self.create_incidence_matrix()
        susceptance_diag = np.diag(susceptance_vector)
        line_susceptance = susceptance_diag @ incidence
        node_susceptance = incidence.T @ susceptance_diag @ incidence
        return line_susceptance, node_susceptance

    def create_ptdf_matrix(self) -> np.ndarray:
        """Compute PTDF matrix (lines × nodes)."""
        slack = list(self.nodes.index[self.nodes.slack])
        slack_idx = [self.nodes.index.get_loc(s) for s in slack]
        line_susceptance, node_susceptance = self.create_susceptance_matrices()

        list_wo_slack = [i for i in range(len(self.nodes)) if i not in slack_idx]
        node_susc_wo_slack = node_susceptance[np.ix_(list_wo_slack, list_wo_slack)]
        inv = np.linalg.inv(node_susc_wo_slack)

        node_susceptance_inv = np.zeros((len(self.nodes), len(self.nodes)))
        node_susceptance_inv[np.ix_(list_wo_slack, list_wo_slack)] = inv

        return line_susceptance @ node_susceptance_inv

    # ------------------------------------------------------------------
    # RES
    # ------------------------------------------------------------------

    def create_res_forecast(self) -> None:
        """Initialize deterministic RES forecasts for d2, d1, d0."""
        self.res["p_d2"] = self.res["p_max_pu"] * self.res["g_max"]
        self.res["p_d1"] = self.res["p_max_pu"] * self.res["g_max"]
        self.res["p_d0"] = self.res["p_max_pu"] * self.res["g_max"]

    def split_res_generators(self, curtailable_share: Dict[str, float]) -> None:
        """
        Split RES into uncertain and controllable parts.

        Parameters
        ----------
        curtailable_share : dict
            Mapping from carrier (e.g. "wind", "solar") to curtailable share (0–1).
        """
        if "sigma" not in self.res:
            self.res["sigma"] = 0

        res = []
        for i, row in self.res.iterrows():
            tech = row["carrier"]
            alpha = curtailable_share.get(tech, 0)

            # Controllable part
            res.append(
                {
                    "name": f"{i}_contr",
                    "bus": row["bus"],
                    "carrier": tech,
                    "mc": row["mc"],
                    "g_max": alpha * row["g_max"],
                    "p_max_pu": row["p_max_pu"],
                    "sigma": 0,
                    "RD": True,
                }
            )
            # Uncertain part
            res.append(
                {
                    "name": i,
                    "bus": row["bus"],
                    "carrier": tech,
                    "mc": row["mc"],
                    "g_max": (1 - alpha) * row["g_max"],
                    "p_max_pu": row["p_max_pu"],
                    "sigma": row["sigma"],
                    "RD": False,
                }
            )

        self.res = pd.DataFrame(res).set_index("name")

    # ------------------------------------------------------------------
    # Zone borders
    # ------------------------------------------------------------------

    def create_zone_border(self) -> pd.DataFrame:
        """
        Compute zone border adjacency matrix.

        Returns
        -------
        pd.DataFrame
            Square matrix (zones × zones), with 1 if zones are connected.
        """
        zones = sorted(self.nodes["zone"].unique())
        zone_to_idx = {z: i for i, z in enumerate(zones)}
        zone_border = np.zeros((len(zones), len(zones)), dtype=int)

        incidence = self.create_incidence_matrix()
        node_zone = self.nodes["zone"].values

        for line in incidence:
            from_node = np.where(line == 1)[0][0]
            to_node = np.where(line == -1)[0][0]
            z_from, z_to = node_zone[from_node], node_zone[to_node]
            if z_from != z_to:
                i, j = zone_to_idx[z_from], zone_to_idx[z_to]
                zone_border[i, j] = zone_border[j, i] = 1

        return pd.DataFrame(zone_border, index=zones, columns=zones)
