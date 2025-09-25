"""
50-Node Case Study: Single Snapshot Analysis
--------------------------------------------

This script evaluates one snapshot of the PyPSA-Eur 50-node model
within the `munacco` framework. Two workflows are demonstrated:

1. **Single Scenario**
   - Load snapshot network
   - Apply RES splits, scaling, and load adjustments
   - Generate 1 scenario
   - Run full CACM pipeline (Capacity Calculation, Validation, Market Coupling, Redispatch)
   - Visualize results with network and domain plots

2. **Multiple Scenarios**
   - Generate 100 stochastic scenarios (forecast errors on D-1 and D-0)
   - Run CACM pipeline with robust validation (chance-constrained)
   - Collect KPIs and analyze distributions

Purpose:
Provide a benchmark workflow for analyzing uncertainty-robust
capacity calculation and validation in a realistic 50-node setting.

"""

import munacco as mc
import pypsa
from tqdm import tqdm
import numpy as np
import pandas as pd
import copy


# ====================================================
# Step 1: Load base network (snapshot selection)
# ====================================================
snapshot = 1570
n = pypsa.Network("thesis-data/pypsa-eur-50node/cwe_network_data/solved_network/base_s_50_elec_.nc")

curtailable_share = {
    "offwind-dc": 0.7,
    "offwind-ac": 0.7,
    "onwind": 0.7,
    "solar": 0.7
}

network = mc.InputLoader().load_from_pypsa(network=n, snapshot=snapshot, initialize=False, p_nom_opt=True)

# RES adjustments (reduce solar in AT)
network.res.loc['AT0 1 0 solar', 'g_max'] /= 2
network.res.loc['AT0 2 0 solar', 'g_max'] /= 2
network.res['sigma'] = 0.2
network.split_res_generators(curtailable_share)

# Generator and line adjustments
network.plants.loc[network.plants.g_max < 800, 'alpha'] = False
network.lines['f_max'] = np.maximum(network.lines['f_max'] * 0.7, 400)

# ====================================================
# Step 2: Adjust loads per zone to avoid infeasibilities
# ====================================================
zone_data = []
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
    else:
        faktor2 = 2

    load_in_zone_corr = network.nodes.loc[network.nodes.zone == z, 'Pd'].sum()

    zone_data.append({
        "zone": z,
        "g_in_zone": g_in_zone,
        "res_in_zone": res_in_zone,
        "res_fix_in_zone": res_fix_in_zone,
        "load_in_zone": load_in_zone,
        "load_in_zone_corr": load_in_zone_corr,
        "faktor1": faktor1,
        "faktor2": faktor2
    })

df_zone_data = pd.DataFrame(zone_data).set_index("zone")
network.initialize()

#%%
# ====================================================
# Step 3: Single Scenario Analysis
# ====================================================
scenario = mc.ScenarioGenerator().generate(network, 1, forecast_timing=['d0'])[0]

model = mc.CACMModel(options_path="munacco/model/options_default.json")
model.options['capacity_calculation']['basecase'] = 'opf'
model.options['capacity_calculation']['include_minram'] = True
model.options['capacity_calculation']['minram'] = 0.7
model.options['capacity_calculation']['frm'] = 0
model.options['validation']['include'] = True
model.options['validation']['vertex_selection'] = True
model.options['validation']['max_vertex_angle'] = 40
model.options['validation']['min_vertices'] = 4
model.options['validation']['max_vertices'] = 10
model.options['validation']['robust'] = True
model.options['validation']['robust_method'] = 'chance_constrained'
model.options['validation']['alpha'] = 'gmax'
model.options['validation']['epsilon'] = 0.1
model.options['market_model']['reserve_generators'] = False
model.options['model']['print'] = False

model.run(scenario)

# Visual inspection
inspector = mc.ScenarioInspector(scenario)
inspector.plot_network_plotly('basecase')
inspector.plot_network_plotly('market_coupling')
inspector.plot_network_plotly('redispatch')
p = inspector.create_domain_plot(('DE', 'NL'), ('DE', 'AT'), 'market_coupling', show=True)

#%%
# ====================================================
# Step 4: Multiple Scenario Analysis
# ====================================================
scenarios = mc.ScenarioGenerator(random_seed=snapshot).generate(
    network, 100, forecast_timing=['d1', 'd0']
)

# Run capacity calculation & validation once on base scenario
scenario_base = scenarios[0]
model.run_capacity_calculation(scenario_base)
model.run_validation(scenario_base)

# Run market coupling & redispatch for all scenarios
for scenario in tqdm(scenarios):
    scenario.npf = scenario_base.npf
    scenario.fb_parameters = scenario_base.fb_parameters
    scenario.results = copy.deepcopy(scenario_base.results)
    model.options['model']['print'] = False
    model.run_market_coupling(scenario)
    model.run_redispatch(scenario)
    model.collect_kpis(scenario)

# Collect KPIs
analyzer = mc.Analyzer(scenarios)

# Example outputs
print("Number of infeasible scenarios:", sum(analyzer.df.remaining_overload))
