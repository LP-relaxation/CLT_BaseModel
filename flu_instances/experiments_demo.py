# Simple demo with flu model with "toy" (not fitted or realistic) parameters
# Wastewater addition from Sonny is still in progress

###########################################################
######################## IMPORTS ##########################
###########################################################

import numpy as np
import pandas as pd

import clt_base as clt
import flu_core as flu

###########################################################
################# READ INPUT FILES ########################
###########################################################

# Obtain path to folder with JSON input files
base_path = clt.utils.PROJECT_ROOT / "flu_instances" / "texas_input_files"

# Get filepaths for initial values of compartments and epi metrics,
#   fixed parameters, configuration, and travel proportions
compartments_epi_metrics_init_vals_filepath = base_path / "compartments_epi_metrics_init_vals.json"
params_filepath = base_path / "common_params.json"
config_filepath = base_path / "config.json"
travel_proportions_filepath = base_path / "travel_proportions.json"

# Get filepaths for school-work calendar CSV
calendar_filepath = base_path / "school_work_calendar.csv"

# Read in files as dictionaries and dataframes
# Note that we can also create these dictionaries directly
#   rather than reading from a predefined input data file
compartments_epi_metrics_dict = \
    clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath)
params_dict = clt.load_json_new_dict(params_filepath)
config_dict = clt.load_json_new_dict(config_filepath)
travel_proportions = clt.load_json_new_dict(travel_proportions_filepath)

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
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(bit_generator),
                           name="north")

south = flu.FluSubpopModel(compartments_epi_metrics_dict,
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(jumped_bit_generator),
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

flu_inter_subpop_repo = flu.FluInterSubpopRepo({"north": north, "south": south},
                                               travel_proportions["subpop_names_mapping"],
                                               travel_proportions["travel_proportions"])

jumped_again_bit_generator = bit_generator.jumped(2)

# Combine two subpopulations into one metapopulation model (travel model)
flu_demo_model = flu.FluMetapopModel(flu_inter_subpop_repo)

test = clt.Experiment(flu_demo_model,
                      ["H"],
                      "test.db")

test.run_random_inputs(num_reps=10,
                       simulation_end_day=100,
                       random_inputs_RNG=np.random.Generator(jumped_again_bit_generator),
                       random_inputs_spec={"north": {"beta_baseline": (1, 10)},
                                           "south": {"beta_baseline": (700, 7000)}},
                       results_filename="test_results.csv",
                       inputs_filename_suffix="test_inputs.csv")

breakpoint()
