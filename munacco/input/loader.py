"""
munacco.input.loader

Provides utilities to load network data into a `NetworkData` object
from either:
- Simplified CSV input files
- A PyPSA network object

Author: [Your Name]
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from munacco.input.network_data import NetworkData

try:
    import pypsa  # type: ignore
except ImportError:
    pypsa = None

logger = logging.getLogger(__name__)


class InputLoader:
    """
    Loader for network input data.

    Supports:
    - Loading from CSV folder (`nodes.csv`, `lines.csv`, `plants.csv`, `res.csv`)
    - Loading from a PyPSA network object
    """

    # ------------------------------------------------------------------
    # CSV loader
    # ------------------------------------------------------------------

    def load_from_csv(self, folder: str | Path, initialize: bool = True) -> NetworkData:
        """
        Load simplified network data from CSV files.

        Expected files inside `folder`:
        - nodes.csv (with zone, Pd, slack)
        - lines.csv (with node_i, node_j, f_max, b)
        - plants.csv (with g_max, mc, bus, alpha)
        - res.csv (with g_max, p_max_pu, mc, bus, sigma, RD)

        Parameters
        ----------
        folder : str or Path
            Path to folder containing CSV input files.
        initialize : bool, default=True
            If True, run `NetworkData.initialize()`.

        Returns
        -------
        NetworkData
        """
        folder = Path(folder)
        required = ["plants.csv", "lines.csv", "nodes.csv", "res.csv"]
        for f in required:
            if not (folder / f).exists():
                raise FileNotFoundError(f"Missing required file: {folder/f}")

        plants = pd.read_csv(folder / "plants.csv", index_col=0, sep=";")
        lines = pd.read_csv(folder / "lines.csv", sep=";")
        nodes = pd.read_csv(folder / "nodes.csv", index_col=0, sep=";")
        res = pd.read_csv(folder / "res.csv", index_col=0, sep=";")

        zones = pd.DataFrame(index=nodes["zone"].unique())

        network = NetworkData(nodes=nodes, lines=lines, plants=plants, res=res, zones=zones)
        if initialize:
            network.initialize()
        return network

    # ------------------------------------------------------------------
    # PyPSA loader
    # ------------------------------------------------------------------

    def load_from_pypsa(
        self,
        network,
        snapshot: int,
        p_nom_opt: bool = False,
        initialize: bool = True,
    ) -> NetworkData:
        """
        Load network data from a PyPSA network object.

        Parameters
        ----------
        network : pypsa.Network
            PyPSA network object.
        snapshot : int
            Time snapshot index.
        p_nom_opt : bool, default=False
            If True, use `p_nom_opt` instead of `p_nom` for plant capacities.
        initialize : bool, default=True
            If True, run `NetworkData.initialize()`.

        Returns
        -------
        NetworkData
        """
        if pypsa is None:
            raise ImportError("PyPSA is not installed. Cannot load PyPSA networks.")

        n = network

        # ---------------- Nodes ----------------
        nodes = n.buses.rename(columns={"Bus": "P", "country": "zone"})
        loads = n.loads_t.p_set.iloc[snapshot]
        nodes["Pd"] = loads
        nodes["slack"] = False

        # ⚠ Hard-coded slack bus (DE0 0)
        if "DE0 0" in nodes.index:
            nodes.loc["DE0 0", "slack"] = True
        else:
            logger.warning("No slack bus defined, defaulting to first node")
            nodes.iloc[0, nodes.columns.get_loc("slack")] = True

        # ---------------- Lines ----------------
        lines = n.lines.rename(
            columns={"bus0": "node_i", "bus1": "node_j", "s_nom": "f_max"}
        )
        lines["name"] = lines.index
        lines["interconnector"] = lines.node_i.str[:2] != lines.node_j.str[:2]

        # ---------------- Generators ----------------
        gens = n.generators.copy()
        if p_nom_opt:
            gens = gens.rename(columns={"p_nom_opt": "g_max", "marginal_cost": "mc"})
        else:
            gens = gens.rename(columns={"p_nom": "g_max", "marginal_cost": "mc"})

        # Add snapshot p_max_pu
        p_max_pu_t = n.generators_t.p_max_pu.iloc[snapshot]
        gens.loc[p_max_pu_t.index.intersection(gens.index), "p_max_pu"] = p_max_pu_t

        # Split RES vs. conventional plants
        res_mask = gens["carrier"].isin(["onwind", "offwind-dc", "offwind-ac", "solar"])
        res = gens[res_mask].copy()
        plants = gens[~res_mask].copy()
        plants["alpha"] = True

        # ---------------- Zones ----------------
        zones = pd.DataFrame(index=nodes["zone"].unique())

        network = NetworkData(nodes=nodes, lines=lines, plants=plants, res=res, zones=zones)
        if initialize:
            network.initialize()
        return network
