"""
Main Run Script for 50-Node CACM Experiments
--------------------------------------------

This script executes the full thesis analysis on the 50-node PyPSA-Eur model.

Each snapshot is run twice:
  • once with p_nom_opt = False (low_res)
  • once with p_nom_opt = True  (high_res)

For each run:
  • 100 stochastic scenarios are generated (seeded by snapshot index for reproducibility)
  • Both deterministic (robust=False) and robust (robust=True, chance-constrained, ε=0.1) 
    validation are executed
  • Results are saved as 4 files per run:
      - snapshot_overview_det_XXX.pkl
      - all_data_det_XXX.pkl
      - snapshot_overview_robust_XXX.pkl
      - all_data_robust_XXX.pkl

Outputs:
  - snapshot_overview_*: one row per snapshot, containing KPIs from the first scenario
  - all_data_*: full KPI dataset for all 100 scenarios per snapshot

Assumptions:
  - MinRAM = 0.7
  - FRM = 0.0
  - Validation with vertex selection, angle/max-vertices constraints
  - Robust validation uses chance constraints with α = gmax, ε = 0.1
"""

import munacco as mc
import pypsa
import numpy as np
import pandas as pd
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib


#%% Step 1: Load base network once
n = pypsa.Network("thesis-data/pypsa-eur-50node/cwe_network_data/solved_network/base_s_50_elec_.nc")

curtailable_share = {
    "offwind-dc": 0.7,
    "offwind-ac": 0.7,
    "onwind": 0.7,
    "solar": 0.5,
}

loader = mc.InputLoader()


def run_snapshot(i, base_network, curtailable_share, p_nom_opt, num_scenarios=100):
    """
    Run one snapshot of the 50-node network with deterministic and robust validation.

    Returns
    -------
    result_data : dict
        {robust_flag: DataFrame with scenario KPIs}
    row_data : dict
        {robust_flag: dict with snapshot overview KPIs or error status}
    """
    result_data = {False: None, True: None}
    row_data = {False: None, True: None}

    try:
        # --- Load snapshot network ---
        network = loader.load_from_pypsa(network=base_network, snapshot=i,
                                         initialize=False, p_nom_opt=p_nom_opt)
        if p_nom_opt:
            # Small correction: reduce some solar capacities
            network.res.loc['AT0 1 0 solar', 'g_max'] /= 2
            network.res.loc['AT0 2 0 solar', 'g_max'] /= 2

        network.res['sigma'] = 0.2
        network.split_res_generators(curtailable_share)

        # Adjustments
        network.plants.loc[network.plants.g_max < 800, 'alpha'] = False
        network.lines['f_max'] = np.maximum(network.lines['f_max'] * 0.7, 400)

        # Zone adjustments
        for z in network.zones.index:
            n_in_zone = network.nodes[network.nodes.zone == z].index

            g_in_zone = sum(network.plants.loc[network.plants.bus.isin(n_in_zone), 'g_max']
                            * network.plants.loc[network.plants.bus.isin(n_in_zone), 'p_max_pu'])
            res_in_zone = sum(network.res.loc[network.res.bus.isin(n_in_zone), 'g_max']
                              * network.res.loc[network.res.bus.isin(n_in_zone), 'p_max_pu'])
            res_fix_in_zone = sum(network.res.loc[network.res.bus.isin(n_in_zone) & ~network.res.RD, 'g_max']
                                  * network.res.loc[network.res.bus.isin(n_in_zone) & ~network.res.RD, 'p_max_pu'])

            load_in_zone = network.nodes.loc[network.nodes.zone == z, 'Pd'].sum()
            faktor1 = (g_in_zone + res_in_zone) / load_in_zone

            if faktor1 < 1:
                network.nodes.loc[network.nodes.zone == z, 'Pd'] *= faktor1 * 0.99

            if res_fix_in_zone > 0:
                faktor2 = load_in_zone / res_fix_in_zone
                if faktor2 < 1:
                    network.nodes.loc[network.nodes.zone == z, 'Pd'] *= 1 / (faktor2 * 0.99)

        network.initialize()

        # --- Generate scenarios (seeded by snapshot index) ---
        scenarios = mc.ScenarioGenerator(random_seed=i).generate(
            network, num_scenarios, forecast_timing=['d0', 'd1']
        )

        def run_model(scenarios, robust):
            """Run CACM model with deterministic or robust validation."""
            model = mc.CACMModel(options_path="munacco/model/options_default.json")
            model.options['capacity_calculation']['include_minram'] = True
            model.options['capacity_calculation']['minram'] = 0.7
            model.options['capacity_calculation']['frm'] = 0
            model.options['validation']['include'] = True
            model.options['validation']['vertex_selection'] = True
            model.options['validation']['max_vertex_angle'] = 40
            model.options['validation']['min_vertices'] = 4
            model.options['validation']['robust'] = robust
            model.options['validation']['robust_method'] = 'chance_constrained'
            model.options['validation']['alpha'] = 'gmax'
            model.options['validation']['epsilon'] = 0.1
            model.options['market_model']['reserve_generators'] = False
            model.options['model']['print'] = False

            scenario_base = copy.deepcopy(scenarios[0])
            model.run_capacity_calculation(scenario_base)
            model.run_validation(scenario_base)

            for scenario in scenarios:
                scenario.fb_parameters = scenario_base.fb_parameters
                scenario.results = copy.deepcopy(scenario_base.results)
                model.run_market_coupling(scenario)
                model.run_redispatch(scenario)
                model.collect_kpis(scenario)

            analyzer = mc.Analyzer(scenarios)
            df = analyzer.df
            df["snapshot"] = i
            return df, scenarios[0].kpis

        for robust in [False, True]:
            result, kpis = run_model(scenarios, robust)
            result_data[robust] = result
            row = {'snapshot': i, 'status': 'ok', 'error': np.nan}
            row.update(kpis)
            row_data[robust] = row

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        for robust in [False, True]:
            row_data[robust] = {'snapshot': i, 'status': 'failed', 'error': err}

    return result_data, row_data


#%% Step 2: Main execution
num_cores = 6
snapshots = list(range(0, 8760, 5))

for p_nom_opt in [False, True]:
    suffix = "high_res" if p_nom_opt else "low_res"
    print(f"\n=== Running with p_nom_opt={p_nom_opt} ({suffix}) ===")
    with tqdm_joblib(tqdm(total=len(snapshots), desc=f"Snapshots {suffix}")):
        results = Parallel(n_jobs=num_cores)(
            delayed(run_snapshot)(i, n, curtailable_share, p_nom_opt) for i in snapshots
        )

    # Collect results
    det_dataframes, det_rows = [], []
    robust_dataframes, robust_rows = [], []

    for res, rows in results: 
        if res[False] is not None:
            det_dataframes.append(res[False])
        if rows[False] is not None:
            det_rows.append(rows[False])
        if res[True] is not None:
            robust_dataframes.append(res[True])
        if rows[True] is not None:
            robust_rows.append(rows[True])

    df_det = pd.DataFrame(det_rows)
    result_det = pd.concat(det_dataframes, ignore_index=True) if det_dataframes else None
    df_robust = pd.DataFrame(robust_rows)
    result_robust = pd.concat(robust_dataframes, ignore_index=True) if robust_dataframes else None

    # Save outputs
    if result_det is not None:
        df_det.to_pickle(f"test_pypsa/result_data3/snapshot_overview_det_{suffix}.pkl")
        result_det.to_pickle(f"test_pypsa/result_data3/all_data_det_{suffix}.pkl")
    if result_robust is not None:
        df_robust.to_pickle(f"test_pypsa/result_data3/snapshot_overview_robust_{suffix}.pkl")
        result_robust.to_pickle(f"test_pypsa/result_data3/all_data_robust_{suffix}.pkl")
