"""
model/model.py

Defines the CACMModel class, the central orchestrator of the Capacity Calculation
and Congestion Management (CACM) process in munacco.

The CACM process includes:
    1. Capacity Calculation
    2. Validation (optional: normal or robust)
    3. Market Coupling
    4. Redispatch

This module provides the high-level pipeline (stages) that are executed on
NetworkData + Scenario objects, storing results and KPIs for analysis.
"""

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Dict

from munacco.tools import angle, compute_sec
from munacco.input.network_data import NetworkData
from munacco.scenario.scenario import Scenario
from munacco.model.market_model import MarketModel
from munacco.model.validation import Validator


class CACMModel:
    """
    Orchestrates one run of the Capacity Calculation and Congestion Management (CACM) process.

    Workflow stages:
        1. Capacity Calculation
        2. Validation (optional, normal or robust)
        3. Market Coupling
        4. Redispatch

    Parameters
    ----------
    options_path : str, optional
        Path to JSON file with model configuration (default:
        "munacco/model/options_default.json").

    Attributes
    ----------
    options : dict
        Configuration dictionary loaded from JSON.
    network : NetworkData
        Network object associated with the current scenario.
    scenario : Scenario
        Scenario currently being processed.
    results : dict
        Stores results of each CACM stage.
    kpis : dict
        Key performance indicators collected at the end of each run.
    """

    def __init__(self, options_path: str = "munacco/model/options_default.json"):
        with open(options_path, "r") as f:
            self.options = json.load(f)

        self.network: Optional[NetworkData] = None
        self.scenario: Optional[Scenario] = None
        self.results: Dict = {}
        self.kpis: Dict = {}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_scenario(self, scenario: Scenario) -> None:
        """Attach a scenario and its network to the model instance."""
        self.network = scenario.network
        self.scenario = scenario

    # ------------------------------------------------------------------
    # Execution pipeline
    # ------------------------------------------------------------------

    def run(self, scenario: Scenario) -> None:
        """
        Run all CACM stages sequentially for a given scenario.

        Parameters
        ----------
        scenario : Scenario
            The scenario instance to run the process on.
        """
        self.initialize_scenario(scenario)

        self.run_capacity_calculation()
        if self.options['validation']['include']:
            self.run_validation()
        self.run_market_coupling()
        self.run_redispatch()

        self.collect_kpis(scenario)

    def batchrun(self, scenarios: List[Scenario]) -> None:
        """
        Run all CACM stages for a list of scenarios.

        Parameters
        ----------
        scenarios : list of Scenario
            Scenarios to run the CACM pipeline on.
        """
        self.options['model']['print'] = False
        for scenario in tqdm(scenarios):
            self.run(scenario)

    # ------------------------------------------------------------------
    # Stage 1: Capacity Calculation
    # ------------------------------------------------------------------

    def run_capacity_calculation(self, scenario: Optional[Scenario] = None) -> None:
        """Run the capacity calculation stage for a given or current scenario."""
        if scenario is not None:
            self.initialize_scenario(scenario)

        # Basecase
        self.scenario.results['basecase'] = MarketModel.market_model(
            model=self,
            forecast_day='d2',
            option=self.options['capacity_calculation']['basecase'],
            name='basecase'
        )

        # GSKs
        self.scenario.gsk = self.calculate_gsk(
            network=self.scenario.network,
            option=self.options['capacity_calculation']['gsk']
        )

        # Flow-based parameters
        self.scenario.z_ptdf = self.scenario.network.ptdf_nes @ self.scenario.gsk
        self.scenario.fb_parameters['initial'] = self.calculate_fb_parameter(self.scenario.results['basecase'])

        # MinRAM adjustment
        self.scenario.fb_parameters['minram'] = self.scenario.fb_parameters['initial'].copy()
        if self.options['capacity_calculation']['include_minram']:
            amr = np.maximum(
                self.scenario.fb_parameters['minram'].FMAX
                * self.options['capacity_calculation']['minram']
                - self.scenario.fb_parameters['minram'].RAM,
                0
            )
            self.scenario.fb_parameters['minram']['AMR'] = amr
            self.scenario.fb_parameters['minram']['RAM'] += amr

        self.scenario.npf = self.scenario.results['basecase'].NP

        if self.options['model']['print']:
            print('Capacity Calculation completed.')

    def calculate_fb_parameter(self, basecase, scenario: Optional[Scenario] = None) -> pd.DataFrame:
        """
        Compute flow-based parameters from a basecase.

        Parameters
        ----------
        basecase : object
            MarketModel result for the basecase.
        scenario : Scenario, optional
            Scenario to initialize if not already set.

        Returns
        -------
        fb_parameter : pd.DataFrame
            Flow-based parameters with PTDF and RAM values.
        """
        if scenario is not None:
            self.initialize_scenario(scenario)

        fb_parameter = pd.DataFrame(columns=['NEC_ID','CB', 'CO', 'DIR', 'FMAX',
                                             'FRM', 'FREF','F0', 'AMR', 'RAM', 'IVA'])
        fb_parameter['CB'] = self.scenario.network.nes.name
        fb_parameter['DIR'] = self.scenario.network.nes.dir
        fb_parameter['FMAX'] = self.scenario.network.nes.f_max
        fb_parameter['NEC_ID'] = fb_parameter.index
        fb_parameter['IVA'] = 0
        fb_parameter['FRM'] = fb_parameter['FMAX'] * self.options['capacity_calculation']['frm']

        f_ref = basecase.F
        f_0 = basecase.F - self.scenario.z_ptdf @ basecase.NP
        ram_temp = fb_parameter.FMAX - f_0 - fb_parameter.FRM

        fb_parameter['FREF'] = f_ref
        fb_parameter['F0'] = f_0
        fb_parameter['RAM'] = ram_temp

        # add zonal PTDFs
        fb_parameter[list(self.scenario.network.zones.index)] = self.scenario.z_ptdf

        return fb_parameter

    def calculate_gsk(self, network: NetworkData, option: str = 'flat') -> pd.DataFrame:
        """
        Compute a generation shift key (GSK) matrix.

        Parameters
        ----------
        network : NetworkData
            Network data instance.
        option : {"flat", "gmax"}
            Method for constructing GSK.
            - "flat": equal participation of each node to the NEX
            - "gmax": weighted by installed conventional generation

        Returns
        -------
        gsk : pd.DataFrame
            GSK matrix (nodes × zones).
        """
        if option == 'gmax':
            gsk = (network.map_nz.T * np.array(network.nodes.g_max)).T / np.array(network.zones.g_max)
        else:
            gsk = network.map_nz / np.array(network.zones.n_nodes)
        return gsk

    # ------------------------------------------------------------------
    # Stage 2: Validation
    # ------------------------------------------------------------------

    def run_validation(self, scenario: Optional[Scenario] = None) -> None:
        """Perform validation of the flow-based domain and compute IVAs."""
        if scenario is not None:
            self.initialize_scenario(scenario)

        self.validation = Validator(self)

        # Compute vertices
        vertices = self.validation.compute_vertices()

        # Vertex selection
        vertices['angle_refprog'] = vertices[self.scenario.network.Z].apply(
            lambda x: angle(self.scenario.npf, x) / np.pi * 180,
            axis=1
        )
        vertices['selection'] = True

        if self.options['validation']['vertex_selection']:
            vertices.loc[vertices['angle_refprog'] > self.options['validation']['max_vertex_angle'], 'selection'] = False
            min_vertices = self.options['validation']['min_vertices']
            idx_min_angle = vertices.angle_refprog.nsmallest(min_vertices).index
            vertices.loc[idx_min_angle, "selection"] = True
            idx_max_angle = vertices.angle_refprog.nsmallest(self.options['validation']['max_vertices']).index
            vertices.loc[~vertices.index.isin(idx_max_angle), "selection"] = False

        # Validation
        vertices = self.validation.validate_vertices(vertices)

        # Create FB constraints
        for factor in ['iva', 'amax', 'a']:
            self.scenario.fb_parameters[factor] = self.validation.create_factored_fb_constraints(
                vertices[vertices.selection], factor
            )

        self.scenario.vertices = vertices

        if self.options['model']['print']:
            print('Validation completed.')

    # ------------------------------------------------------------------
    # Stage 3: Market Coupling
    # ------------------------------------------------------------------

    def run_market_coupling(self, scenario: Optional[Scenario] = None) -> None:
        """Run the market coupling stage (FBMC)."""
        if scenario is not None:
            self.initialize_scenario(scenario)

        fb_param = 'iva' if self.options['validation']['include'] else 'minram'

        self.scenario.results['market_coupling'] = MarketModel.market_model(
            model=self,
            forecast_day='d1',
            option='fbmc',
            name='market_coupling',
            fb_parameter=fb_param
        )

        if self.options['model']['print']:
            print('Market Coupling completed.')

    # ------------------------------------------------------------------
    # Stage 4: Redispatch
    # ------------------------------------------------------------------

    def run_redispatch(self, scenario: Optional[Scenario] = None) -> None:
        """Run the redispatch stage after market coupling."""
        if scenario is not None:
            self.initialize_scenario(scenario)

        self.scenario.results['redispatch'] = MarketModel.redispatch_model(
            model=self,
            forecast_day='d0',
            name='redispatch',
            market_result=self.scenario.results['market_coupling']
        )

        self.scenario.results['redispatch'].EX = compute_sec(
            self.scenario.results['redispatch'].network.zone_border,
            self.scenario.results['redispatch'].NP.values
        )

        if self.options['model']['print']:
            print('Redispatch completed.')

    # ------------------------------------------------------------------
    # KPI collection
    # ------------------------------------------------------------------

    def collect_kpis(self, scenario: Optional[Scenario] = None) -> None:
        """Collect KPIs for the current scenario and store them in `scenario.kpis`."""
        self.initialize_scenario(scenario)

        kpis = {}
        kpis['scenario'] = self.scenario.id

        # Market
        kpis['market_cost'] = (
            sum(self.scenario.results['market_coupling'].GEN * self.scenario.network.plants.mc)
            + sum(self.scenario.results['market_coupling'].RES_GEN * self.scenario.network.res.mc)
            + sum(self.scenario.results['market_coupling'].CU) * self.options['market_model']['curtailment_cost']
        )

        # Redispatch
        kpis['rd_cost'] = (
            sum(self.scenario.results['redispatch'].RD_POS * self.scenario.network.plants.mc)
            + sum(self.scenario.results['redispatch'].RD_NEG * self.scenario.network.plants.mc)
            + sum(self.scenario.results['redispatch'].CU) * self.options['market_model']['curtailment_cost']
            + sum(self.scenario.results['redispatch'].RD_POS + self.scenario.results['redispatch'].RD_NEG)
            * self.options['redispatch']['rd_cost']
        )
        kpis['rd_volume'] = (
            sum(self.scenario.results['redispatch'].RD_POS)
            + sum(self.scenario.results['redispatch'].RD_NEG)
            + sum(self.scenario.results['redispatch'].RD_RES)
        )

        # System
        kpis['total_system_cost'] = kpis['market_cost'] + kpis['rd_cost']
        kpis['total_curtailment'] = (
            sum(self.scenario.results['market_coupling'].CU)
            + sum(self.scenario.results['redispatch'].CU)
            - sum(self.scenario.results['redispatch'].RD_RES)
        )
        kpis['exchange_volume'] = sum(self.scenario.results['redispatch'].EX)

        # Validation
        if self.options['validation']['include']:
            kpis['iva_sum'] = sum(self.scenario.fb_parameters['iva']['IVA'])
            kpis['iva_count'] = sum(self.scenario.fb_parameters['iva']['IVA'] > 0)
        else:
            kpis['iva_sum'] = 0
            kpis['iva_count'] = 0

        # Network Security
        kpis['remaining_overload'] = (
            sum(self.scenario.results['redispatch'].SLACK)
            + sum(self.scenario.results['redispatch'].SLACK_XB) > 1e-2
        )
        kpis['total_slack'] = sum(self.scenario.results['redispatch'].SLACK)
        kpis['n_slack'] = sum(self.scenario.results['redispatch'].SLACK > 1e-5)
        kpis['slack_XB'] = sum(self.scenario.results['redispatch'].SLACK_XB)

        idx = list(np.where(self.scenario.results['redispatch'].SLACK > 1e-5)[0])
        kpis['nes_w_remaining_congestions'] = self.scenario.network.nes.loc[idx]

        self.scenario.kpis = kpis

        if self.options['model']['print']:
            print(f"KPIs collected for scenario: {getattr(self.scenario, 'name', 'Unnamed')}")
