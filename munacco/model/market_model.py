"""
model/market_model.py

Defines the MarketModel class, which builds and solves optimization problems
for different stages of the CACM process:

    - Economic Dispatch Problem (EDP)
    - Market Coupling (FBMC, OPF, no-exchange, etc.)
    - Redispatch

All problems are modeled in cvxpy and return Result objects with consistent
structure for downstream analysis.
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, Dict, Any

from munacco.tools import compute_sec
from munacco.model.result import Result


class MarketModel:
    """
    Optimization models for market stages of CACM.

    Provides class and static methods to construct and solve cvxpy-based
    formulations of economic dispatch, flow-based market coupling, and
    redispatch.
    """

    # ------------------------------------------------------------------
    # Base EDP (Economic Dispatch Problem)
    # ------------------------------------------------------------------

    @staticmethod
    def build_EDP(model, forecast_day: str) -> Tuple[cp.Minimize, list, Dict[str, cp.Variable]]:
        """
        Build a basic Economic Dispatch Problem (EDP) formulation.

        Decision variables
        ------------------
        GEN : generation of conventional plants
        INJ : nodal injections
        NP  : net positions (zonal)
        CU  : curtailment of RES

        Parameters
        ----------
        model : CACMModel
            The CACM model instance with attached scenario.
        forecast_day : {"d2", "d1", "d0"}
            Forecast horizon to use (D-2, D-1, or D-0).

        Returns
        -------
        obj : cvxpy.Minimize
            Objective function.
        constraints : list
            List of cvxpy constraints.
        var_dict : dict
            Dictionary of decision variables.
        """
        GEN = cp.Variable(len(model.scenario.network.P), pos=True, name='GEN')
        INJ = cp.Variable(len(model.scenario.network.N), name='INJ')
        NP = cp.Variable(len(model.scenario.network.Z), name='NP')
        CU = cp.Variable(len(model.scenario.network.RES), pos=True, name='CU')

        idx_no_res_RD = np.where(model.scenario.network.res.RD == False)[0]

        obj = cp.Minimize(
            model.scenario.network.plants["mc"].values @ GEN
            + model.scenario.network.res["mc"].values
            @ (model.scenario.res_forecast[f'p_{forecast_day}'].values - CU)
            + (max(model.scenario.network.plants["mc"].values) + 10) * sum(CU)
        )

        constraints = [
            # Nodal energy balance
            model.scenario.network.map_np @ GEN
            + model.scenario.network.map_nres
            @ (model.scenario.res_forecast[f'p_{forecast_day}'].values - CU)
            - model.scenario.network.nodes.Pd.values
            == INJ,

            # Zonal energy balance
            model.scenario.network.map_nz.T @ INJ == NP,

            # Generator constraints
            GEN <= model.scenario.network.plants.g_max.values,

            # Curtailment
            CU <= model.scenario.res_forecast[f'p_{forecast_day}'].values,
            CU[idx_no_res_RD] == 0,

            # System balance
            sum(INJ) == 0,
        ]

        var_dict = {"GEN": GEN, "INJ": INJ, "NP": NP, "CU": CU}
        return obj, constraints, var_dict

    # ------------------------------------------------------------------
    # Market Coupling
    # ------------------------------------------------------------------

    @classmethod
    def market_model(
        cls,
        model,
        forecast_day: str = "d2",
        option: str = "uniform",
        name: str = "basecase",
        fb_parameter: str = None,
    ) -> Result:
        """
        Build and solve the market coupling model based on EDP.

        Parameters
        ----------
        model : CACMModel
            The CACM model instance with attached scenario.
        forecast_day : {"d2", "d1"}
            Forecast horizon (D-2 for basecase, D-1 for market coupling).
        option : {"opf", "noex", "fbmc", "uniform"}
            Market coupling formulation.
        name : str, optional
            Label for the result.
        fb_parameter : str, optional
            Which flow-based parameter set to use (e.g. "minram", "iva").

        Returns
        -------
        result : Result
            Structured result object with decision variable values and KPIs.
        """
        obj, constraints, var_dict = cls.build_EDP(model, forecast_day)

        if option in ["opf", "noex"]:
            INJ = var_dict["INJ"]
            constraints.append(model.scenario.network.ptdf_nes @ INJ <= model.scenario.network.nes.f_max)

        if option == "noex":
            NP = var_dict["NP"]
            constraints.append(NP == 0)

        elif option == "fbmc":
            A = np.array(model.scenario.fb_parameters[fb_parameter][model.scenario.network.Z])
            b = np.array(model.scenario.fb_parameters[fb_parameter].RAM)
            NP = var_dict["NP"]
            constraints.append(A @ NP <= b)

            if model.options["market_model"]["reserve_generators"] and model.scenario.GENMargin is not None:
                GEN = var_dict["GEN"]
                constraints.append(GEN >= model.scenario.GENMargin)

        # Solve
        prob = cp.Problem(obj, constraints)
        prob.solve(verbose=False)

        result = Result(
            GEN=prob.var_dict["GEN"].value,
            INJ=prob.var_dict["INJ"].value,
            NP=prob.var_dict["NP"].value,
            CU=prob.var_dict["CU"].value,
            network=model.scenario.network,
            res_max=model.scenario.res_forecast[f"p_{forecast_day}"].values,
        )
        result.RES_GEN = model.scenario.res_forecast[f"p_{forecast_day}"].values - result.CU
        result.EX = compute_sec(result.network.zone_border, result.NP.values)

        result.config["type"] = option
        result.config["name"] = name
        result.config["objective_value"] = prob.value

        return result

    # ------------------------------------------------------------------
    # Redispatch
    # ------------------------------------------------------------------

    @staticmethod
    def redispatch_model(
        model,
        market_result: Result,
        forecast_day: str = "d0",
        name: str = "redispatch",
    ) -> Result:
        """
        Build and solve the redispatch optimization model.

        Parameters
        ----------
        model : CACMModel
            CACM model instance with attached scenario.
        market_result : Result
            Result object from the prior market coupling stage.
        forecast_day : {"d0"}
            Realized (D-0) RES forecast horizon.
        name : str, optional
            Label for the result.

        Returns
        -------
        result : Result
            Structured result object with redispatch decisions.
        """
        xb_limit = model.options["redispatch"]["xb_limit"] if model.options["redispatch"]["xb"] else 0

        res_gen = (
            market_result.RES_GEN.values
            + model.scenario.res_forecast[f"p_{forecast_day}"].values
            - market_result.res_max.values
        )

        # Decision variables
        RD_POS = cp.Variable(len(model.scenario.network.P), pos=True)
        RD_NEG = cp.Variable(len(model.scenario.network.P), pos=True)
        XBRD_POS = cp.Variable(len(model.scenario.network.Z), pos=True)
        XBRD_NEG = cp.Variable(len(model.scenario.network.Z), pos=True)
        INJ = cp.Variable(len(model.scenario.network.N))
        NP = cp.Variable(len(model.scenario.network.Z))
        CU = cp.Variable(len(model.scenario.network.RES), pos=True)
        RD_res = cp.Variable(len(model.scenario.network.RES), pos=True)
        SLACK = cp.Variable(len(model.scenario.network.nes), pos=True)
        SLACK_XB = cp.Variable(len(model.scenario.network.Z), pos=True)

        idx_no_res_RD = np.where(model.scenario.network.res.RD == False)[0]

        # Objective
        obj = cp.Minimize(
            (model.scenario.network.plants["mc"].values + model.options["redispatch"]["rd_cost"]) @ RD_POS
            + (max(model.scenario.network.plants["mc"].values) - model.scenario.network.plants["mc"].values
               + model.options["redispatch"]["rd_cost"]) @ RD_NEG
            + (max(model.scenario.network.plants["mc"].values) + 100) * sum(CU)
            + model.options["redispatch"]["xb_cost"] * sum(XBRD_POS + XBRD_NEG)
            + model.options["redispatch"]["slack_cost"] * sum(SLACK)
            + (model.options["redispatch"]["slack_cost"] + 1000) * sum(SLACK_XB)
        )

        # Constraints
        constraints = [
            # Nodal balance
            model.scenario.network.map_np
            @ (market_result.GEN.values - RD_NEG + RD_POS)
            + model.scenario.network.map_nres @ (res_gen - CU + RD_res)
            - model.scenario.network.nodes.Pd.values
            == INJ,

            # Zonal balance
            model.scenario.network.map_nz.T @ INJ == NP,

            # Generator limits
            (market_result.GEN.values + RD_POS - RD_NEG) <= model.scenario.network.plants.g_max.values,
            (market_result.GEN.values + RD_POS - RD_NEG) >= 0,

            # Curtailment
            CU <= res_gen,
            CU[idx_no_res_RD] == 0,
            RD_res[idx_no_res_RD] == 0,

            # System balance
            sum(INJ) == 0,

            # Line constraints with slack
            model.scenario.network.ptdf_nes @ INJ <= model.scenario.network.nes.f_max.values + SLACK,

            # Cross-border redispatch
            XBRD_POS <= xb_limit + SLACK_XB,
            XBRD_NEG <= xb_limit + SLACK_XB,
            NP - market_result.NP.values == XBRD_POS - XBRD_NEG,

            # Reverse curtailment of RES not allowed
            RD_res == 0,
        ]

        prob = cp.Problem(obj, constraints)
        prob.solve(verbose=False)

        result = Result(
            INJ=INJ.value,
            NP=NP.value,
            CU=CU.value,
            RD_POS=RD_POS.value,
            RD_NEG=RD_NEG.value,
            XBRD_NEG=XBRD_NEG.value,
            XBRD_POS=XBRD_POS.value,
            RD_res=RD_res.value,
            SLACK=SLACK.value,
            SLACK_XB=SLACK_XB.value,
            network=model.scenario.network,
            res_max=model.scenario.res_forecast[f"p_{forecast_day}"].values,
        )
        result.RES_GEN = res_gen - result.CU + result.RD_RES
        result.GEN = market_result.GEN - result.RD_NEG + result.RD_POS

        result.config["type"] = "redispatch"
        result.config["name"] = name
        result.config["objective_value"] = prob.value

        return result
