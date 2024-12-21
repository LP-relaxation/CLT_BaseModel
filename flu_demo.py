from pathlib import Path

import numpy as np

import clt_base as clt
import flu_model as fm

# Obtain path to folder with JSON input files
base_path = Path(__file__).parent / "flu_demo_input_files"

# Get filepaths for configuration, fixed parameter values, and
#   initial values of state variables
config_filepath = base_path / "config.json"
fixed_params_filepath = base_path / "fixed_params.json"
state_vars_init_vals_filepath = base_path / "state_variables_init_vals.json"

config = clt.make_dataclass_from_json(clt.Config, config_filepath)

fixed_params = clt.make_dataclass_from_json(fm.FluFixedParams, fixed_params_filepath)

sim_state = clt.make_dataclass_from_json(fm.FluSimState, state_vars_init_vals_filepath)

# Create a constructor using these filepaths
flu_demo_model = \
    fm.FluSubpopModel(sim_state,
                      fixed_params,
                      config,
                      np.random.default_rng(88888))

# Simulate 300 days
flu_demo_model.simulate_until_time_period(300)

breakpoint()

# Plot
clt.create_basic_compartment_history_plot(flu_demo_model, "flu_demo_model.png")