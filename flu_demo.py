import numpy as np
from pathlib import Path

from flu_components import FluSubpopModel
from plotting import create_basic_compartment_history_plot

# Obtain path to folder with JSON input files
base_path = Path(__file__).parent / "flu_demo_input_files"

# Get filepaths for configuration, fixed parameter values, and
#   initial values of state variables
config_filepath = base_path / "config.json"
fixed_params_filepath = base_path / "fixed_params.json"
state_vars_init_vals_filepath = base_path / "state_variables_init_vals.json"

# Create a constructor using these filepaths
flu_demo_model = \
    FluSubpopModel(config_filepath,
                   fixed_params_filepath,
                   state_vars_init_vals_filepath)

# Need to add RNG
flu_demo_model.RNG = np.random.default_rng(88888)

# Simulate 300 days
flu_demo_model.simulate_until_time_period(300)

breakpoint()

# Plot
create_basic_compartment_history_plot(flu_demo_model,
                                      "flu_demo_model.png")