import munacco as mc

# --- Step 1: Load network
network = mc.InputLoader().load_from_csv("examples/base_model")
network.split_res_generators({'wind': 0.7, 'solar': 0.5})
network.initialize()

# --- Step 2: Create a scenarios
scenarios = mc.ScenarioGenerator().generate(network, 100, forecast_timing=['d1','d0'])

# --- Step 3: Load Model and set Options
model = mc.CACMModel(options_path = "munacco/model/options_default.json")
model.options['capacity_calculation']['basecase'] = 'opf'
model.options['capacity_calculation']['include_minram'] = True
model.options['capacity_calculation']['minram'] = 0.7
model.options['capacity_calculation']['frm'] = 0
model.options['validation']['include'] = True
model.options['validation']['vertex_selection'] = True
model.options['validation']['max_vertex_angle'] = 40
model.options['validation']['robust'] = False
model.options['redispatch']['xb'] = False

# Run Model
model.batchrun(scenarios)

# --- Step 4: Analyze
analyzer = mc.Analyzer(scenarios)
n_overloads = sum(analyzer.df.remaining_overload)
print(f'Number of scenarios with remaining overloads: {n_overloads}')

# --- Step 5: Plot FB Domain
scenario = scenarios[0]
inspector = mc.ScenarioInspector(scenario)
inspector.create_domain_plot(('Z1','Z2'), ('Z2','Z3'), 'market_coupling', show=True)



