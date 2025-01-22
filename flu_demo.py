# Simple demo with flu model with "toy" (not fitted or realistic) parameters
# Wastewater addition from Sonny is still in progress

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
state_vars_init_vals_filepath = base_path / "compartments_epi_metrics_init_vals.json"
params_filepath = base_path / "common_params.json"
config_filepath = base_path / "config.json"

state_dict = clt.load_json_new_dict(state_vars_init_vals_filepath)
params_dict = clt.load_json_new_dict(params_filepath)
config_dict = clt.load_json_new_dict(config_filepath)

travel_proportions = pd.read_csv(base_path / "travel_proportions.csv")

bit_generator = np.random.MT19937(88888)
jumped_bit_generator = bit_generator.jumped(1)

north = flu.FluSubpopModel(state_dict,
                           params_dict,
                           config_dict,
                           np.random.Generator(bit_generator),
                           name="north")

north.run_model_checks()
north.display()

south = flu.FluSubpopModel(state_dict,
                           params_dict,
                           config_dict,
                           np.random.Generator(jumped_bit_generator),
                           name="south")

south.run_model_checks()
south.display()

flu_demo_model = flu.FluMetapopModel({"north": north, "south": south}, travel_proportions)

flu_demo_model.simulate_until_time_period(100)

clt.plot_metapop_epi_metrics(flu_demo_model)

clt.plot_metapop_total_infected_deaths(flu_demo_model)

clt.plot_metapop_basic_compartment_history(flu_demo_model)

