# Simple demo with flu model with "toy" (not fitted or realistic) parameters
# Includes wastewater addition -- from Sonny

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import clt_base as clt
import flu_model as flu

import copy

# Obtain path to folder with JSON input files
base_path = Path(__file__).parent / "flu_demo_input_files"

# Get filepaths for initial values of state variables, fixed parameters,
#   and configuration
state_vars_init_vals_filepath = base_path / "state_variables_init_vals.json"
params_filepath = base_path / "common_params.json"
config_filepath = base_path / "config.json"

state_dict = clt.load_json(state_vars_init_vals_filepath)
params_dict = clt.load_json(params_filepath)
config_dict = clt.load_json(config_filepath)

travel_proportions = pd.read_csv(base_path / "travel_proportions.csv")

bit_generator = np.random.MT19937(88888)
jumped_bit_generator = bit_generator.jumped(1)

north = flu.FluSubpopModel(state_dict,
                           params_dict,
                           config_dict,
                           np.random.Generator(bit_generator),
                           name="north")

south = flu.FluSubpopModel(state_dict,
                           params_dict,
                           config_dict,
                           np.random.Generator(jumped_bit_generator),
                           name="south")

flu_demo_model = flu.FluMetapopModel({"north": north, "south": south}, travel_proportions)

# flu_demo_model.run_model_checks()

# flu_demo_model.display()

breakpoint()

# Simulate 300 days
flu_demo_model.simulate_until_time_period(300)

breakpoint()

# Plot
clt.create_basic_compartment_history_plot(flu_demo_model, "flu_demo_model.png")

if flu_demo_model.wastewater_enabled:
    ww = flu_demo_model.epi_metrics.wastewater.history_vals_list
    print(ww)
    plt.plot(ww)
    plt.grid(True)
    plt.show()