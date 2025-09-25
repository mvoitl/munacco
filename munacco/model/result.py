"""
model/result.py

Defines the Result class, a structured container for optimization outcomes
from market coupling and redispatch problems.

Each result holds decision variable values as pandas Series indexed by
network components (nodes, zones, plants, etc.), plus derived quantities
such as flows and exchanges. Metadata and objective values are stored in
the config attribute.
"""

import pandas as pd
from typing import Optional, Dict, Any


class Result:
    """
    Container for results of optimization problems in CACM.

    Attributes
    ----------
    network : NetworkData
        The network object used for the optimization.
    GEN : pd.Series
        Generation by conventional plants.
    INJ : pd.Series
        Nodal injections.
    NP : pd.Series
        Zonal net positions.
    CU : pd.Series
        RES curtailment.
    RD_POS : pd.Series
        Positive redispatch of generators.
    RD_NEG : pd.Series
        Negative redispatch of generators.
    XBRD_POS : pd.Series
        Cross-border redispatch (positive).
    XBRD_NEG : pd.Series
        Cross-border redispatch (negative).
    RD_RES : pd.Series
        RES redispatch.
    SLACK : pd.Series
        Line constraint slack variables (per network element).
    SLACK_XB : pd.Series
        Slack variables for cross-border redispatch.
    F : pd.Series
        Physical line flows (computed via PTDF).
    res_max : pd.Series
        Maximum RES availability for the forecast horizon.
    RES_GEN : Optional[pd.Series]
        RES generation (set externally after solving).
    EX : Optional[pd.Series]
        Exchanges across borders (set externally after solving).
    config : dict
        Metadata with fields: "type", "name", "objective_value".
    """

    def __init__(
        self,
        *,
        GEN=None,
        INJ=None,
        NP=None,
        CU=None,
        RD_POS=None,
        RD_NEG=None,
        XBRD_POS=None,
        XBRD_NEG=None,
        RD_res=None,
        SLACK=None,
        SLACK_XB=None,
        network,
        res_max,
    ):
        self.network = network

        # Initialize decision variables as Series with appropriate indices
        self.INJ = pd.Series(INJ, index=network.N, name="INJ") if INJ is not None else pd.Series(0, index=network.N, name="INJ")
        self.NP = pd.Series(NP, index=network.Z, name="NP") if NP is not None else pd.Series(0, index=network.Z, name="NP")
        self.CU = pd.Series(CU, index=network.RES, name="CU") if CU is not None else pd.Series(0, index=network.RES, name="CU")
        self.RD_POS = pd.Series(RD_POS, index=network.P, name="RD_POS") if RD_POS is not None else pd.Series(0, index=network.P, name="RD_POS")
        self.RD_NEG = pd.Series(RD_NEG, index=network.P, name="RD_NEG") if RD_NEG is not None else pd.Series(0, index=network.P, name="RD_NEG")
        self.XBRD_POS = pd.Series(XBRD_POS, index=network.Z, name="XBRD_POS") if XBRD_POS is not None else pd.Series(0, index=network.Z, name="XBRD_POS")
        self.XBRD_NEG = pd.Series(XBRD_NEG, index=network.Z, name="XBRD_NEG") if XBRD_NEG is not None else pd.Series(0, index=network.Z, name="XBRD_NEG")
        self.RD_RES = pd.Series(RD_res, index=network.RES, name="RD_RES") if RD_res is not None else pd.Series(0, index=network.RES, name="RD_RES")
        self.SLACK = pd.Series(SLACK, index=network.nes.index, name="SLACK") if SLACK is not None else pd.Series(0, index=network.nes.index, name="SLACK")
        self.SLACK_XB = pd.Series(SLACK_XB, index=network.Z, name="SLACK_XB") if SLACK_XB is not None else pd.Series(0, index=network.Z, name="SLACK_XB")
        self.GEN = pd.Series(GEN, index=network.P, name="GEN") if GEN is not None else pd.Series(0, index=network.P, name="GEN")

        # Derived values
        self.F = pd.Series(self.calculate_flows(), index=network.nes.index, name="F")

        self.res_max = pd.Series(res_max, index=network.RES, name="g_max")
        self.RES_GEN: Optional[pd.Series] = None
        self.EX: Optional[pd.Series] = None

        self.config: Dict[str, Any] = {
            "type": None,
            "name": None,
            "objective_value": None,
        }

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def calculate_flows(self) -> pd.Series:
        """
        Compute physical flows based on nodal injections and PTDF.

        Returns
        -------
        pd.Series
            Flow values per network element.
        """
        return self.network.ptdf_nes @ self.INJ

    def summary(self) -> Dict[str, Any]:
        """
        Return a compact summary of key metrics.

        Returns
        -------
        dict
            Objective value, redispatch volumes, curtailment, slack usage.
        """
        return {
            "Objective Value": self.config["objective_value"],
            "Total Redispatch (MW)": float((self.RD_POS + self.RD_NEG).sum()),
            "Total RES Curtailment (MW)": float(self.CU.sum()),
            "Total XB Redispatch (MW)": float((self.XBRD_POS + self.XBRD_NEG).sum()),
            "RD Slack Violations (MW)": float(self.SLACK.sum()),
            "XB RD Slack Violations (MW)": float(self.SLACK_XB.sum()),
        }
