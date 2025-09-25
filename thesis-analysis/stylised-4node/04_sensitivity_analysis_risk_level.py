"""
4-Node Experiment: Sensitivity to Validation Risk Parameter (ε)
---------------------------------------------------------------

This script evaluates how the choice of ε in the chance-constrained 
validation formulation affects system costs and feasibility.

Steps:
1. Load a 4-node network.
2. Generate multiple scenarios with forecast uncertainty (d0, d1).
3. Run the CACM model with chance-constrained validation for a range of ε.
4. Collect KPIs and error flows for each ε.
5. Save results for later visualization and analysis.
6. Additionally, run deterministic validation as a baseline for comparison.
"""

import munacco as mc
from tqdm import tqdm
import pandas as pd
import numpy as np
import copy

# ----------------------------
# Step 1: Load base network
# ----------------------------
network = mc.InputLoader().load_from_csv("thesis-data/stylised-4node/base_model")
network.split_res_generators({'wind': 0.7, 'solar': 0.5})
network.initialize()

# ----------------------------
# Step 2: Generate scenarios
# ----------------------------
scenarios = mc.ScenarioGenerator().generate(network, 100, forecast_timing=['d0', 'd1'])

kpis_all = {}
err_flows_all = {}

#%%
# ----------------------------
# Step 3: Run chance-constrained validation for multiple epsilons
# ----------------------------
for eps in np.arange(0.01, 0.99, 0.01):
    model = mc.CACMModel(options_path="munacco/model/options_default.json")
    model.options['capacity_calculation']['basecase'] = 'opf'
    model.options['capacity_calculation']['include_minram'] = True
    model.options['capacity_calculation']['minram'] = 0.5
    model.options['capacity_calculation']['frm'] = 0.1

    model.options['validation']['include'] = True
    model.options['validation']['vertex_selection'] = True
    model.options['validation']['robust'] = True
    model.options['validation']['robust_method'] = 'chance_constrained'
    model.options['validation']['epsilon'] = eps

    # Base scenario CC and VAL
    scenario_base = copy.deepcopy(scenarios[0])
    model.run_capacity_calculation(scenario_base)
    model.run_validation(scenario_base)

    # Run all scenarios (reuse base results where possible)
    for scenario in tqdm(scenarios, desc=f"ε = {eps:.2f}"):
        scenario.alpha = scenario_base.alpha
        scenario.fb_parameters = scenario_base.fb_parameters
        scenario.results = scenario_base.results
        model.options['model']['print'] = False
        model.run_market_coupling(scenario)
        model.run_redispatch(scenario)
        model.collect_kpis(scenario)

    analyzer = mc.Analyzer(scenarios)
    kpis_all[eps] = analyzer.df
    err_flows_all[eps] = analyzer.err_flows

# ----------------------------
# Step 4: Collect all results
# ----------------------------
all_data = []
for eps, df in kpis_all.items():
    temp = df.copy()
    temp['epsilon'] = float(eps)
    all_data.append(temp)

df_all = pd.concat(all_data).sort_values('epsilon')
df_all.to_pickle('thesis-data/stylised-4node/04_results_sensitivity_100.pkl')

# ----------------------------
# Step 5: Deterministic validation benchmark
# ----------------------------
model = mc.CACMModel(options_path="munacco/model/options_default.json")
model.options['capacity_calculation']['basecase'] = 'opf'
model.options['capacity_calculation']['include_minram'] = True
model.options['capacity_calculation']['minram'] = 0.5
model.options['capacity_calculation']['frm'] = 0.1

model.options['validation']['include'] = True
model.options['validation']['vertex_selection'] = True
model.options['validation']['robust'] = False   # <--- deterministic
model.options['validation']['robust_method'] = 'chance_constrained'

scenario_base = scenarios[0]
model.run_capacity_calculation(scenario_base)
model.run_validation(scenario_base)

for scenario in tqdm(scenarios, desc="Deterministic Validation"):
    scenario.alpha = scenario_base.alpha
    scenario.fb_parameters = scenario_base.fb_parameters
    scenario.results = scenario_base.results
    model.options['model']['print'] = False
    model.run_market_coupling(scenario)
    model.run_redispatch(scenario)
    model.collect_kpis(scenario)

analyzer = mc.Analyzer(scenarios)
det_val_df = analyzer.df
det_val_df.to_pickle('thesis-data/stylised-4node/04_results_sensitivity_det.pkl')
