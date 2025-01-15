# Simple demo with flu model with "toy" (not fitted or realistic) parameters
# Includes wastewater addition -- from Sonny

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import clt_base as clt
import flu_model as flu

# Obtain path to folder with JSON input files
base_path = Path(__file__).parent / "flu_demo_input_files"

# Get filepaths for initial values of state variables, fixed parameters,
#   and configuration
state_vars_init_vals_filepath = base_path / "state_variables_init_vals.json"
params_filepath = base_path / "subpop_params.json"
config_filepath = base_path / "config.json"

state_dict = clt.load_json(state_vars_init_vals_filepath)
params_dict = clt.load_json(params_filepath)
config_dict = clt.load_json(config_filepath)

flu_demo_model = flu.FluSubpopModel(state_dict,
                                    params_dict,
                                    config_dict,
                                    np.random.default_rng(88888))

flu_demo_model.run_model_checks()

breakpoint()

flu_demo_model.display()

# Simulate 300 days
flu_demo_model.simulate_until_time_period(300)

# Plot
clt.create_basic_compartment_history_plot(flu_demo_model, "flu_demo_model.png")

if flu_demo_model.wastewater_enabled:
    ww = flu_demo_model.epi_metrics.wastewater.history_vals_list
    print(ww)
    plt.plot(ww)
    plt.grid(True)
    plt.show()