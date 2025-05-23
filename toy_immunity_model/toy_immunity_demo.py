# Simple demo with flu model with "toy" (not fitted or realistic) parameters
# Wastewater addition from Sonny is still in progress

###########################################################
######################## IMPORTS ##########################
###########################################################

from pathlib import Path
import numpy as np
import pandas as pd

import toy_immunity_plotting

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import city-level transmission base components module
import clt_base as clt

# Import flu model module, which contains customized subclasses
import toy_immunity_components as imm

###########################################################
################# READ INPUT FILES ########################
###########################################################

# Obtain path to folder with JSON input files
base_path = Path(__file__).parent / "toy_immunity_input_files"

# Get filepaths for initial values of compartments and epi metrics, fixed parameters,
#   configuration, and travel proportions
compartments_epi_metrics_init_vals_filepath = base_path / "compartments_epi_metrics_init_vals.json"
params_filepath = base_path / "params.json"
config_filepath = base_path / "config.json"

humidity_filepath = base_path / "humidity_austin_2023_2024.csv"

# Read in files as dictionaries and dataframes
# Note that we can also create these dictionaries directly
#   rather than reading from a predefined input data file
compartments_epi_metrics_dict = \
    clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath)
params_dict = clt.load_json_new_dict(params_filepath)
config_dict = clt.load_json_new_dict(config_filepath)

# Create two independent bit generators
bit_generator = np.random.MT19937(88888)
jumped_bit_generator = bit_generator.jumped(1)

###########################################################
############# CREATE SUBPOPULATION MODELS #################
###########################################################

# Create two subpopulation models, one for the north
#   side of the city and one for the south side of the city
# In this case, these two (toy) subpopulations have the
#   same demographics, initial compartment and epi metric values,
#   fixed parameters, and school-work calendar.
# If we wanted the "north" subpopulation and "south"
#   subpopulation to have different aforementioned values,
#   we could read in two separate sets of files -- one
#   for each subpopulation
model = imm.ToyImmunitySubpopModel(compartments_epi_metrics_dict,
                                   params_dict,
                                   config_dict,
                                   np.random.Generator(bit_generator),
                                   humidity_filepath)

model.simulate_until_day(300)

linear_saturation_model = imm.LinearSaturationSubpopModel(compartments_epi_metrics_dict,
                                                          params_dict,
                                                          config_dict,
                                                          np.random.Generator(bit_generator),
                                                          humidity_filepath)

linear_saturation_model.simulate_until_day(300)

clt.plot_subpop_basic_compartment_history(model)

clt.plot_subpop_epi_metrics(model)

toy_immunity_plotting.make_graph_set(model)

toy_immunity_plotting.make_comparison_graph_set(model, linear_saturation_model)

toy_immunity_plotting.changing_param_val_graph(model, "param", "beta", [0.5, 1, 1.5, 2])

breakpoint()
