# Simple demo with flu model with "toy" (not fitted or realistic) parameters
# Includes wastewater addition -- from Sonny

from pathlib import Path
import numpy as np

import clt_base as base
import flu_model as flu

# Obtain path to folder with JSON input files
base_path = Path(__file__).parent / "flu_demo_input_files"

# Get filepaths for configuration, fixed parameter values, and
#   initial values of state variables
config_filepath = base_path / "config.json"
fixed_params_filepath = base_path / "fixed_params.json"
state_vars_init_vals_filepath = base_path / "state_variables_init_vals.json"

config = base.make_dataclass_from_json(base.Config, config_filepath)
fixed_params = base.make_dataclass_from_json(flu.FluFixedParams, fixed_params_filepath)
sim_state = base.make_dataclass_from_json(flu.FluSimState, state_vars_init_vals_filepath)

flu_demo_model = flu.FluSubpopModel(sim_state,
                                    fixed_params,
                                    config,
                                    np.random.default_rng(88888))

# Simulate 300 days
flu_demo_model.simulate_until_time_period(300)

# Plot
base.create_basic_compartment_history_plot(flu_demo_model, "flu_demo_model.png")
