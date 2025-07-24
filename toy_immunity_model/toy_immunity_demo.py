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

params_dict["inf_induced_saturation"] = 0
params_dict["inf_induced_immune_wane"] = 0
params_dict["vax_induced_saturation"] = 0
params_dict["vax_induced_immune_wane"] = 0

model = imm.ToyImmunitySubpopModel(compartments_epi_metrics_dict,
                                   params_dict,
                                   config_dict,
                                   np.random.Generator(bit_generator),
                                   humidity_filepath)

model.simulate_until_day(100)

if np.isclose(np.sum(model.transition_variables.R_to_S.history_vals_list)/np.sum(model.params.total_pop_age_risk),
              np.sum(model.epi_metrics.M.history_vals_list[-1])):
    print("Anass's condition is correct with these parameters")

