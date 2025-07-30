# Simple demo with flu model with "toy" (not fitted or realistic) parameters

###########################################################
######################## IMPORTS ##########################
###########################################################

from pathlib import Path
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import city-level transmission base components module
import clt_base as clt

# Import flu model module, which contains customized subclasses
from flu_model import flu_components as flu

###########################################################
################# READ INPUT FILES ########################
###########################################################

# Obtain path to folder with JSON input files
base_path = Path(__file__).parent / "texas_input_files"

# Get filepaths
compartments_epi_metrics_init_vals_filepath = base_path / "init_vals.json"
subpop_params_filepath = base_path / "common_subpop_params.json"
mixing_params_filepath = base_path / "mixing_params.json"
config_filepath = base_path / "config.json"

# Get filepaths for school-work calendar CSV and humidity CSV
calendar_filepath = base_path / "school_work_calendar.csv"
humidity_filepath = base_path / "humidity_austin_2023_2024.csv"

# Read in files as dictionaries and dataframes
# Note that we can also create these dictionaries directly
#   rather than reading from a predefined input data file
compartments_epi_metrics_dict = \
    clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath)
subpop_params_dict = clt.load_json_new_dict(subpop_params_filepath)
mixing_params_dict = clt.load_json_new_dict(mixing_params_filepath)
config_dict = clt.load_json_new_dict(config_filepath)

calendar_df = pd.read_csv(calendar_filepath, index_col=0)

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
north = flu.FluSubpopModel(compartments_epi_metrics_dict,
                           subpop_params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(bit_generator),
                           humidity_filepath,
                           name="north")

south = flu.FluSubpopModel(compartments_epi_metrics_dict,
                           subpop_params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(jumped_bit_generator),
                           humidity_filepath,
                           name="south")

# The structure of the code allows us to access
#   the current state and fixed parameters of each
#   subpopulation model.
# For example, here we print out the fixed parameter
#   value for beta_baseline for the "south" subpopulation
print(south.params.beta_baseline)
# 1

# We can also manually change a fixed parameter value
#   after we have created a SubpopModel -- like so...
# Note that this is quite a large and unrealistic value of
#   beta_baseline, but we'll use this to create
#   a dramatic difference between the two subpopulations
south.params.beta_baseline = 10

###########################################################
############# CREATE METAPOPULATION MODEL #################
###########################################################

# Combine two subpopulations into one metapopulation model (travel model)
flu_demo_model = flu.FluMetapopModel([north, south],
                                     mixing_params_dict)

###########################################################
################# SIMULATE & ANALYZE ######################
###########################################################

# Simulate for 50 days
flu_demo_model.simulate_until_day(50)

# Get the current real date of the simulation and the
#   current simulation day
print(flu_demo_model.current_simulation_day)
print(flu_demo_model.current_real_date)

# Simulate for another 50 days, from where we last left off
flu_demo_model.simulate_until_day(100)

# We can "unpack" our flu model and access the current state
#   of each subpopulation -- here's an example with the "north"
#   subpopulation -- this is the state after we have simulated
#   our 100 days
print(flu_demo_model.subpop_models.north.state)

# Remember that we can easily access the objects that
#   make up our subpopulation model -- here's an
#   example of accessing the "north" subpopulation's
#   compartments
# See API references for more attribute access syntax
print(flu_demo_model.subpop_models.north.compartments)

# Generate simple compartment history plot for flu model
clt.plot_metapop_basic_compartment_history(flu_demo_model, "basic_compartment_history.png")

###########################################################
######## MAKE MODIFICATIONS, SIMULATE & ANALYZE ###########
###########################################################

# Reset the simulation
# Note -- does NOT reset the RNG! Only clears each object's
#   history, resets the simulation day/date to the starting
#   day/date, and returns state variables to their initial values.
flu_demo_model.reset_simulation()

# Set school and work contact matrices to zero matrices for both
#   subpopulations and demonstrate that this removes the calendar-induced
#   periodicity
num_age_groups = flu_demo_model.subpop_models.north.params.num_age_groups

for subpop_model in flu_demo_model.subpop_models.values():
    subpop_model.params.school_contact_matrix = np.zeros((num_age_groups, num_age_groups))
    subpop_model.params.work_contact_matrix = np.zeros((num_age_groups, num_age_groups))

flu_demo_model.simulate_until_day(100)

clt.plot_metapop_basic_compartment_history(flu_demo_model,
                                           "basic_compartment_history_no_periodicity.png")