"""
munacco.model.validation

Validation module for the CACM model. Implements validation of
flow-based domains by computing vertices, performing factor
validation, and generating factored FB constraints.

Includes:
- Standard factor validation (deterministic)
- Chance-constrained validation (robust, with optimized alpha)
- Chance-constrained validation (robust, with fixed alpha)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import cvxpy as cp
import pandas as pd
import scipy.stats as sps

from munacco.tools import compute_polytope_vertices, compute_sec
from munacco.model.result import Result

logger = logging.getLogger(__name__)


class Validator:
    """
    Validates flow-based domains by testing feasibility of extreme points (vertices).

    This class provides methods to:
    - Compute polytope vertices of the flow-based domain
    - Perform deterministic or robust factor validation
    - Generate factored FB constraints including IVAs

    Parameters
    ----------
    model : CACMModel
        The parent CACM model instance providing network, scenario, and options.
    """

    def __init__(self, model):
        self.model = model

    # ------------------------------------------------------------------
    # Vertex computation
    # ------------------------------------------------------------------

    def compute_vertices(self) -> pd.DataFrame:
        """
        Compute vertices of the flow-based domain polytope.

        Returns
        -------
        pd.DataFrame
            Vertices with zonal coordinates.
        """
        A = self.model.scenario.fb_parameters["minram"][
            list(self.model.scenario.network.zones.index)
        ]
        b = self.model.scenario.fb_parameters["minram"]["RAM"]

        # Add system balance constraints
        A_hat = np.vstack(
            [
                A,
                np.ones(len(self.model.scenario.network.zones.index)),
                -np.ones(len(self.model.scenario.network.zones.index)),
            ]
        )
        b_hat = np.hstack([b, 0, 0])

        V = compute_polytope_vertices(A_hat, b_hat)
        return pd.DataFrame(columns=self.model.scenario.network.Z, data=V)

    # ------------------------------------------------------------------
    # Validation pipeline
    # ------------------------------------------------------------------

    def validate_vertices(
        self, vertices: pd.DataFrame, forecast_day: str = "p_d2"
    ) -> pd.DataFrame:
        """
        Validate vertices of the FB domain by computing scaling factors.

        Parameters
        ----------
        vertices : pd.DataFrame
            Vertices to validate (with selection column).
        forecast_day : str, default="p_d2"
            RES forecast horizon to use.

        Returns
        -------
        pd.DataFrame
            Vertices augmented with amax_factor, a_factor, iva_factor, and cnec_idx.
        """
        amax, a_factor, cnec_idx = [], [], []

        for i, row in vertices.iterrows():
            if row.get("selection", True):
                vertex = row[self.model.scenario.network.Z]

                # Choose robust or deterministic validation
                if self.model.options["validation"]["robust"]:
                    if self.model.options["validation"]["robust_method"] == "chance_constrained":
                        if self.model.options["validation"]["alpha"] == "gmax":
                            gmax_plants = self.model.scenario.network.plants.loc[
                                self.model.scenario.network.plants.alpha, "g_max"
                            ]
                            self.model.scenario.alpha = gmax_plants / gmax_plants.sum()

                        if np.sum(self.model.scenario.alpha) > 0:
                            result_amax = self.factor_validation_cc_fix_alpha(
                                vertex, forecast_day, modus="amax"
                            )
                            result_a = self.factor_validation_cc_fix_alpha(
                                vertex, forecast_day, modus="a"
                            )
                        else:
                            result_amax = self.factor_validation_cc(
                                vertex, forecast_day, modus="amax"
                            )
                            result_a = self.factor_validation_cc(
                                vertex, forecast_day, modus="a"
                            )
                    elif self.model.options["validation"]["robust_method"] == "margin":
                        result_amax = list(
                            self.factor_validation(vertex, forecast_day, modus="amax")
                        )
                        result_a = list(
                            self.factor_validation(vertex, forecast_day, modus="a")
                        )
                        if result_amax[0] is not None:
                            result_amax[0] *= 0.8
                        if result_a[0] is not None:
                            result_a[0] *= 0.8
                else:
                    result_amax = self.factor_validation(vertex, forecast_day, modus="amax")
                    result_a = self.factor_validation(vertex, forecast_day, modus="a")

                # Store results in scenario
                self.model.scenario.results[f"validation_amax_{i}"] = result_amax[3]
                self.model.scenario.results[f"validation_a_{i}"] = result_a[3]

                amax.append(result_amax[0])
                a_factor.append(result_a[0])
                try:
                    cnec_idx.append(result_a[1])
                except Exception:
                    cnec_idx.append("")
            else:
                amax.append(None)
                a_factor.append(None)
                cnec_idx.append("")

        # Round and assign factors
        precision = self.model.options["model"]["precision_digits"]
        vertices["amax_factor"] = [
            round(x, precision) if x is not None else None for x in amax
        ]
        vertices["a_factor"] = [
            round(x, precision) if x is not None else None for x in a_factor
        ]

        # Compute IVA factors
        vertices["iva_factor"] = 1.0
        cond = vertices.a_factor < np.minimum(vertices.amax_factor, 1)
        vertices.loc[cond, "iva_factor"] = vertices.loc[cond, "a_factor"]
        vertices["cnec_idx"] = cnec_idx

        self.model.scenario.GENMargin = result_a[2]

        return vertices

    # ------------------------------------------------------------------
    # FB constraints
    # ------------------------------------------------------------------

    def create_factored_fb_constraints(
        self, vertices: pd.DataFrame, factor: str
    ) -> pd.DataFrame:
        """
        Create factored FB constraints based on vertex scaling factors.

        Parameters
        ----------
        vertices : pd.DataFrame
            Vertices with factor values.
        factor : {"a", "amax", "iva"}
            Which factor to apply.

        Returns
        -------
        pd.DataFrame
            Adjusted FB constraints with IVA column.
        """
        constraints = self.model.scenario.fb_parameters["minram"].copy()
        constraints["VERTEX"] = ""
        A = constraints[self.model.scenario.network.Z]
        b = constraints.RAM
        iva = constraints.IVA

        cs, ivas = [constraints], [iva]

        for i in vertices.index:
            vertex = np.round(
                vertices.loc[i, self.model.scenario.network.Z].values.astype(float), 4
            )
            cnec_idx = np.where(np.round(A @ vertex - b, 4) == 0)[0]
            c_temp = self.model.scenario.fb_parameters["minram"].iloc[cnec_idx].copy()
            b_temp = b[cnec_idx].copy()
            c_temp["VERTEX"] = f"V{i}"

            b_tt = b_temp * vertices.loc[i, f"{factor}_factor"]
            iva = b_temp - b_tt
            ivas.append(iva)
            cs.append(c_temp)

        constraints = pd.concat(cs, ignore_index=True, axis=0)
        ivas = pd.concat(ivas, ignore_index=True, axis=0)
        constraints["IVA"] = ivas
        constraints["RAM"] -= ivas

        constraints.drop_duplicates(
            subset=constraints.columns[0:-1], inplace=True, ignore_index=True
        )
        return constraints

    # ------------------------------------------------------------------
    # Factor validation (deterministic)
    # ------------------------------------------------------------------

    def factor_validation(
        self, vertex, forecast_day: str, modus: str = "amax"
    ) -> Tuple[float, Optional[str], int, Optional[Result]]:
        """
        Deterministic factor validation.

        Parameters
        ----------
        vertex : array-like
            Zonal net position direction.
        forecast_day : str
            Forecast day key (e.g. "p_d2").
        modus : {"amax", "a"}, default="amax"
            Whether to enforce line constraints ("a") or not ("amax").

        Returns
        -------
        Tuple
            (a_factor, cnec_idx, GENMargin, Result)
        """
        net = self.model.scenario.network
        GEN = cp.Variable(len(net.P), pos=True)
        INJ = cp.Variable(len(net.N))
        CU = cp.Variable(len(net.RES), pos=True)
        af = cp.Variable(pos=True)

        D = net.nodes.Pd.values
        idx_no_res_RD = np.where(net.res.RD == False)[0]

        obj = cp.Maximize(af)
        constraints = [
            net.map_np @ GEN
            + net.map_nres @ (self.model.scenario.res_forecast[forecast_day].values - CU)
            - D
            == INJ,
            net.map_nz.T @ INJ == af * vertex,
            CU <= self.model.scenario.res_forecast[forecast_day].values,
            GEN <= net.plants.g_max.values,
            sum(INJ) == 0,
            CU[idx_no_res_RD] == 0,
        ]
        if modus == "a":
            constraints.append(net.ptdf_nes @ INJ <= net.nes.f_max.values)

        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            logger.warning("Clarabel failed, retrying with SCS...")
            prob.solve(solver=cp.SCS)

        result, cnec_idx = None, None
        if prob.status == "optimal":
            result = Result(
                GEN=GEN.value,
                INJ=INJ.value,
                NP=af.value * vertex,
                CU=CU.value,
                network=net,
                res_max=self.model.scenario.res_forecast[forecast_day].values,
            )
            result.RES_GEN = self.model.scenario.res_forecast[forecast_day].values - result.CU
            result.EX = compute_sec(result.network.zone_border, result.NP.values)
            result.config["name"] = "validation"

            if modus == "a":
                cond = (
                    np.round(
                        net.ptdf_nes @ INJ.value - net.nes.f_max,
                        self.model.options["model"]["precision_digits"],
                    )
                    == 0
                )
                if cond.any():
                    cnec_idx = net.nes[cond].index[0]

        return af.value, cnec_idx, 0, result

    # ------------------------------------------------------------------
    # Robust factor validation (optimized alpha & fixed alpha)
    # ------------------------------------------------------------------


    def factor_validation_cc(
        self,
        vertex,
        forecast_day: str,
        modus: str = "amax",
    ):
        """
        Chance-constrained (robust) factor validation with *optimized* alpha.

        Solves an SOCP to find the largest scaling factor A_f such that
        A_f * vertex is feasible under:
          - nodal/zonal balances,
          - generator limits tightened by uncertainty,
          - (optionally) line limits tightened by uncertainty,
          - RES curtailment bounds,
          - chance constraint level epsilon (converted to z-score).

        Uncertainty model:
          - RES forecast errors are independent with std = sigma * g_max (per RES unit).
          - Alpha distributes balancing of RES errors across generators via map_palpha.
          - Line chance constraints are imposed via SOC: ||R_i * sigma_root||_2 ≤ CC_i,
            and FLOW_i + z * CC_i ≤ FMAX_i, where R_i depends on PTDF and alpha.

        Side effects
        ------------
        - Stores CCMargin (per CNEC) in `self.model.scenario.CCMargin`.
        - Stores chosen alpha in `self.model.scenario.alpha`.

        Parameters
        ----------
        vertex : array-like
            Zonal net position direction to be scaled (length |Z|).
        forecast_day : str
            Column key in `scenario.res_forecast` to use (e.g. "p_d2").
        modus : {"amax", "a"}, optional
            - "amax": maximize A_f without enforcing line constraints.
            - "a": also enforce line constraints (robust).

        Returns
        -------
        A_f : float
            Optimal scaling factor for the vertex.
        cnec_idx : Optional[str]
            Index of the limiting CNEC when `modus == "a"` (else None).
        GENMargin : Optional[np.ndarray]
            Generator margin tightening vector (length |P|).
        result : Optional[Result]
            Result object for the optimal solution (None if not optimal).
        """
        # Zero-out contr RES uncertainty (as in original code)
        self.model.scenario.res_forecast.loc[
            self.model.scenario.res_forecast.sigma.index.str.contains("contr"), "sigma"
        ] = 0

        # Uncertainty parameters
        sigma_vec = self.model.scenario.res_forecast.sigma * self.model.scenario.res_forecast.g_max
        sigma_mat = np.diag((sigma_vec.values if hasattr(sigma_vec, "values") else sigma_vec) ** 2)
        sigma_mat_root = sigma_mat ** 0.5
        sum_sigma_root = sigma_mat_root.sum()  # scalar upper bound used for GEN tightening

        eps = self.model.options["validation"]["epsilon"]
        z = sps.norm(0, 1).ppf(1 - eps)

        net = self.model.scenario.network
        P, N, Z, nes = len(net.P), len(net.N), len(net.Z), len(net.nes)

        # Decision variables
        GEN   = cp.Variable(P, pos=True)
        INJ   = cp.Variable(N)
        CU    = cp.Variable(len(net.RES), pos=True)
        A_f   = cp.Variable(pos=True)

        INFPOS   = cp.Variable(N, pos=True)  # feasibility slacks on nodal balance (+)
        INFNEG   = cp.Variable(N, pos=True)  # feasibility slacks on nodal balance (-)
        ALPHA    = cp.Variable(len(net.A), pos=True)  # balancing distribution (optimized)
        CCMargin = cp.Variable(nes, pos=True)         # per-CNEC chance margin

        D = net.nodes.Pd.values
        idx_no_res_RD = np.where(self.model.scenario.network.res.RD == False)[0]

        # Nominal flow
        FLOW = net.ptdf_nes @ INJ

        # Alpha effect on line flows (maps RES uncertainty through alpha and PTDF)
        # Shape gymnastics follow the original approach
        ALPHA_FLOW = net.ptdf_nes @ (
            net.map_nres - cp.multiply(
                cp.reshape((net.map_np @ (net.map_palpha @ ALPHA)), (N, 1), order="C"),
                np.ones((1, len(net.RES)))
            )
        )

        # Objective: maximize A_f, penalize CCMargin and infeasibility slacks
        # (weights chosen to strongly discourage infeasibility)
        obj = cp.Maximize(A_f - cp.sum(CCMargin) - 1e3 * cp.sum(INFPOS + INFNEG))

        constraints = [
            # Nodal balance with feasibility slacks
            net.map_np @ GEN
            + net.map_nres @ (self.model.scenario.res_forecast[forecast_day].values - CU)
            - D + INFPOS - INFNEG
            == INJ,

            # Zonal balance along the vertex direction
            net.map_nz.T @ INJ == A_f * vertex,

            # RES curtailment bounds
            CU <= self.model.scenario.res_forecast[forecast_day].values,
            CU[idx_no_res_RD] == 0,

            # Generator upper bounds (robust tightening via alpha)
            GEN <= net.plants.g_max.values,
            sum(INJ) == 0,

            # Alpha normalization
            cp.sum(ALPHA) == 1,
        ]

        # Robust line constraints only in "a" mode
        if modus == "a":
            constraints.append(FLOW + z * CCMargin <= net.nes.f_max.values)

        # Robust generator tightening — two-sided via alpha
        gen_constraints = []
        for i in range(P):
            tighten = z * (net.map_palpha @ ALPHA)[i] * sum_sigma_root
            gen_constraints += [
                GEN[i] + tighten <= net.plants.g_max.values[i],
                -GEN[i] + tighten <= 0,
            ]

        # SOC constraints for line margins: ||R_i * sigma_root||_2 ≤ CCMargin[i]
        soc_constraints = [
            cp.SOC(CCMargin[i], (ALPHA_FLOW[i, :].T @ sigma_mat_root).T)
            for i in range(nes)
        ]

        prob = cp.Problem(obj, constraints + gen_constraints + soc_constraints)
        try:
            # ECOS is generally reliable for SOCP here
            prob.solve(solver=cp.ECOS, verbose=False, max_iters=500)
        except cp.SolverError:
            # Fallback
            prob.solve(solver=cp.SCS, verbose=False)

        result, GENMargin, cnec_idx = None, None, None
        if prob.status == "optimal":
            result = Result(
                GEN=GEN.value,
                INJ=INJ.value,
                NP=A_f.value * vertex,
                CU=CU.value,
                network=net,
                res_max=self.model.scenario.res_forecast[forecast_day].values,
            )
            result.RES_GEN = self.model.scenario.res_forecast[forecast_day].values - result.CU
            result.EX = compute_sec(result.network.zone_border, result.NP.values)
            result.config["name"] = "validation"

            # Store outputs used later
            GENMargin = z * (net.map_palpha @ ALPHA.value) * sum_sigma_root
            self.model.scenario.CCMargin = CCMargin.value
            self.model.scenario.alpha = ALPHA.value

            if modus == "a":
                cond = (
                    np.round(
                        net.ptdf_nes @ INJ.value - net.nes.f_max,
                        self.model.options["model"]["precision_digits"],
                    )
                    == 0
                )
                if cond.any():
                    cnec_idx = net.nes[cond].index[0]

        return A_f.value, cnec_idx, GENMargin, result




    # ------------------------------------------------------------------
    # Robust validation (chance-constrained)
    # ------------------------------------------------------------------
    
    def factor_validation_cc_fix_alpha(
        self,
        vertex,
        forecast_day: str,
        modus: str = "amax",
    ):
        """
        Chance-constrained (robust) factor validation with *fixed* alpha.

        Same model as `factor_validation_cc`, but uses the alpha vector from
        `self.model.scenario.alpha` (must be set beforehand).

        Side effects
        ------------
        - Stores CCMargin (per CNEC) in `self.model.scenario.CCMargin`.
        - Leaves the (fixed) alpha unchanged in `self.model.scenario.alpha`.

        Parameters
        ----------
        vertex : array-like
            Zonal net position direction to be scaled (length |Z|).
        forecast_day : str
            Column key in `scenario.res_forecast` to use (e.g. "p_d2").
        modus : {"amax", "a"}, optional
            - "amax": maximize A_f without enforcing line constraints.
            - "a": also enforce line constraints (robust).

        Returns
        -------
        A_f : float
            Optimal scaling factor for the vertex.
        cnec_idx : Optional[str]
            Index of the limiting CNEC when `modus == "a"` (else None).
        GENMargin : Optional[np.ndarray]
            Generator margin tightening vector (length |P|) implied by fixed alpha.
        result : Optional[Result]
            Result object for the optimal solution (None if not optimal).
        """
        # Zero-out contr RES uncertainty (as in original code)
        self.model.scenario.res_forecast.loc[
            self.model.scenario.res_forecast.sigma.index.str.contains("contr"), "sigma"
        ] = 0

        # Uncertainty parameters
        sigma_vec = self.model.scenario.res_forecast.sigma * self.model.scenario.res_forecast.g_max
        sigma_mat = np.diag((sigma_vec.values if hasattr(sigma_vec, "values") else sigma_vec) ** 2)
        sigma_mat_root = sigma_mat ** 0.5
        sum_sigma_root = sigma_mat_root.sum()

        eps = self.model.options["validation"]["epsilon"]
        z = sps.norm(0, 1).ppf(1 - eps)

        net = self.model.scenario.network
        alpha = self.model.scenario.alpha  # fixed vector
        assert alpha is not None and np.isfinite(alpha).all(), "Fixed alpha must be set before calling this method."

        P, N, Z, nes = len(net.P), len(net.N), len(net.Z), len(net.nes)

        # Decision variables
        GEN   = cp.Variable(P, pos=True)
        INJ   = cp.Variable(N)
        CU    = cp.Variable(len(net.RES), pos=True)
        A_f   = cp.Variable(pos=True)

        INFPOS   = cp.Variable(N, pos=True)
        INFNEG   = cp.Variable(N, pos=True)
        CCMargin = cp.Variable(nes, pos=True)

        D = net.nodes.Pd.values
        idx_no_res_RD = np.where(self.model.scenario.network.res.RD == False)[0]

        # Nominal flow
        FLOW = net.ptdf_nes @ INJ

        # Fixed alpha in the ALPHA_FLOW map
        ALPHA_FLOW = net.ptdf_nes @ (
            net.map_nres - cp.multiply(
                cp.reshape((net.map_np @ (net.map_palpha @ alpha)), (N, 1), order="C"),
                np.ones((1, len(net.RES)))
            )
        )

        obj = cp.Maximize(A_f - cp.sum(CCMargin) - 1e3 * cp.sum(INFPOS + INFNEG))

        constraints = [
            # Nodal balance with feasibility slacks
            net.map_np @ GEN
            + net.map_nres @ (self.model.scenario.res_forecast[forecast_day].values - CU)
            - D + INFPOS - INFNEG
            == INJ,

            # Zonal balance along the vertex direction
            net.map_nz.T @ INJ == A_f * vertex,

            # RES bounds
            CU <= self.model.scenario.res_forecast[forecast_day].values,
            CU[idx_no_res_RD] == 0,

            # Generator box (nominal); robust tightening added below
            GEN <= net.plants.g_max.values,
            sum(INJ) == 0,
        ]

        if modus == "a":
            constraints.append(FLOW + z * CCMargin <= net.nes.f_max.values)

        # Robust generator tightening from fixed alpha
        gen_constraints = []
        for i in range(P):
            tighten = z * (net.map_palpha @ alpha)[i] * sum_sigma_root
            gen_constraints += [
                GEN[i] + tighten <= net.plants.g_max.values[i],
                -GEN[i] + tighten <= 0,
            ]

        # SOC constraints for line margins
        soc_constraints = [
            cp.SOC(CCMargin[i], (ALPHA_FLOW[i, :].T @ sigma_mat_root).T)
            for i in range(nes)
        ]

        prob = cp.Problem(obj, constraints + gen_constraints + soc_constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False, max_iters=500)
        except cp.SolverError:
            prob.solve(solver=cp.SCS, verbose=False)

        result, GENMargin, cnec_idx = None, None, None
        if prob.status == "optimal":
            result = Result(
                GEN=GEN.value,
                INJ=INJ.value,
                NP=A_f.value * vertex,
                CU=CU.value,
                network=net,
                res_max=self.model.scenario.res_forecast[forecast_day].values,
            )
            result.RES_GEN = self.model.scenario.res_forecast[forecast_day].values - result.CU
            result.EX = compute_sec(result.network.zone_border, result.NP.values)
            result.config["name"] = "validation"

            GENMargin = z * (net.map_palpha @ alpha) * sum_sigma_root
            self.model.scenario.CCMargin = CCMargin.value
            self.model.scenario.alpha = alpha  # keep as is

            if modus == "a":
                cond = (
                    np.round(
                        net.ptdf_nes @ INJ.value - net.nes.f_max,
                        self.model.options["model"]["precision_digits"],
                    )
                    == 0
                )
                if cond.any():
                    cnec_idx = net.nes[cond].index[0]

        return A_f.value, cnec_idx, GENMargin, result

            