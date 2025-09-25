"""
Helper Script: Meshpoint Calculation for Stylised 4-Node Network
----------------------------------------------------------------
This script computes meshpoints of the flow-based domain for use in 
subsequent scenario experiments (deterministic and uncertain cases).

Two meshpoint sets are produced:
    1. Deterministic case (no forecast uncertainty).
    2. Uncertain case (solar uncertainty, chance-constrained validation).

Outputs:
    - 02_meshpoints_det.pkl : meshpoints for deterministic case
    - 02_meshpoints_unc.pkl : meshpoints for uncertain case
"""

import munacco as mc

#%% Step 1: Load network
loader = mc.InputLoader()
network = loader.load_from_csv("thesis-data/stylised-4node/base_model")
network.split_res_generators({'wind': 0.7, 'solar': 0.5})
network.initialize()

#%% Step 2: Deterministic scenario (no uncertainty)
scenario = mc.ScenarioGenerator().generate(
    network, 1, forecast_timing=['d0'], forecast_sigma={'wind': 0, 'solar': 0}
)[0]

# Prepare model
model = mc.CACMModel(options_path="munacco/model/options_default.json")
model.options['capacity_calculation']['basecase'] = 'opf'
model.options['capacity_calculation']['frm'] = 0
model.options['capacity_calculation']['include_minram'] = False
model.options['validation']['include'] = False

# Run and create meshpoints
model.run(scenario)
inspector = mc.ScenarioInspector(scenario)
p = inspector.create_domain_plot(('Z1','Z2'), ('Z1','Z3'),
                                 'market_coupling', show=False)
p.xlims = [-210, 310]
p.ylims = [-220, 150]
p.create_mesh_points(230, 2, forecast_day='d0', center_np_shift=False)

meshpoints_det = p.mesh_points
meshpoints_det.to_pickle("thesis-data/stylised-4node/02_meshpoints_det.pkl")

#%% Step 3: Uncertain scenario (solar uncertainty, robust validation)
scenario = mc.ScenarioGenerator().generate(
    network, 1, forecast_timing=['d0'], forecast_sigma={'wind': 0, 'solar': 0.2}
)[0]

# Adjust RES forecast manually to reflect case study assumptions
scenario.res_forecast.loc['pv3', 'p_d0'] = 9.5
scenario.res_forecast.loc['pv4', 'p_d0'] = 30.7

# Model options for robust chance-constrained validation
model.options['validation']['include'] = True
model.options['validation']['vertex_selection'] = True
model.options['validation']['max_vertex_angle'] = 40
model.options['validation']['robust'] = True
model.options['validation']['robust_method'] = 'chance_constrained'

# Run and create meshpoints
model.run(scenario)
inspector = mc.ScenarioInspector(scenario)
inspector.plot_network(result_name='redispatch')  # optional sanity check
p = inspector.create_domain_plot(('Z1','Z2'), ('Z1','Z3'),
                                 'market_coupling', show=False)
p.xlims = [-210, 310]
p.ylims = [-220, 150]
p.create_mesh_points(230, 2, forecast_day='d0', center_np_shift=False)

meshpoints_unc = p.mesh_points
meshpoints_unc.to_pickle("thesis-data/stylised-4node/02_meshpoints_unc.pkl")
