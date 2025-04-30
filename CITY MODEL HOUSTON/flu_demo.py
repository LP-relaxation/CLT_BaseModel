# Simple demo with flu model with "toy" (not fitted or realistic) parameters
# Wastewater addition from Sonny is still in progress

###########################################################
######################## IMPORTS ##########################
###########################################################

from pathlib import Path
import numpy as np
import pandas as pd

# Import city-level transmission base components module
import clt_base as clt

# Import flu model module, which contains customized subclasses
import flu_model as flu

###########################################################
################# READ INPUT FILES ########################
###########################################################

# Obtain path to folder with JSON input files
base_path = Path(__file__).parent / "flu_demo_input_files"

# Get filepaths for initial values of compartments and epi metrics, fixed parameters,
#   configuration, and travel proportions
compartments_epi_metrics_init_vals_filepath1 = base_path / "compartments_epi_metrics_init_vals.json"
compartments_epi_metrics_init_vals_filepath2 = base_path / "compartments_epi_metrics_init_vals copy 2.json"
compartments_epi_metrics_init_vals_filepath3 = base_path / "compartments_epi_metrics_init_vals copy 3.json"
compartments_epi_metrics_init_vals_filepath4 = base_path / "compartments_epi_metrics_init_vals copy 4.json"
compartments_epi_metrics_init_vals_filepath5 = base_path / "compartments_epi_metrics_init_vals copy 5.json"
compartments_epi_metrics_init_vals_filepath6 = base_path / "compartments_epi_metrics_init_vals copy 6.json"
compartments_epi_metrics_init_vals_filepath7 = base_path / "compartments_epi_metrics_init_vals copy 7.json"
compartments_epi_metrics_init_vals_filepath8 = base_path / "compartments_epi_metrics_init_vals copy 8.json"
compartments_epi_metrics_init_vals_filepath9 = base_path / "compartments_epi_metrics_init_vals copy 9.json"
compartments_epi_metrics_init_vals_filepath10 = base_path / "compartments_epi_metrics_init_vals copy 10.json"
compartments_epi_metrics_init_vals_filepath11 = base_path / "compartments_epi_metrics_init_vals copy 11.json"
params_filepath = base_path / "common_params.json"
config_filepath = base_path / "config.json"
travel_proportions_filepath = base_path / "travel_proportions.json"

# Get filepaths for school-work calendar CSV
calendar_filepath = base_path / "school_work_calendar.csv"

# Read in files as dictionaries and dataframes
# Note that we can also create these dictionaries directly
#   rather than reading from a predefined input data file
compartments_epi_metrics_dict1 = \
    clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath1)
compartments_epi_metrics_dict2 = \
    clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath2)
compartments_epi_metrics_dict3 = \
    clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath3)
compartments_epi_metrics_dict4 = \
    clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath4)
compartments_epi_metrics_dict5 = \
    clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath5)
compartments_epi_metrics_dict6 = \
    clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath6)
compartments_epi_metrics_dict7 = \
    clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath7)
compartments_epi_metrics_dict8 = \
    clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath8)
compartments_epi_metrics_dict9 = \
    clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath9)
compartments_epi_metrics_dict10 = \
    clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath10)
compartments_epi_metrics_dict11 = \
    clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath11)
params_dict = clt.load_json_new_dict(params_filepath)
config_dict = clt.load_json_new_dict(config_filepath)

calendar_df = pd.read_csv(calendar_filepath, index_col=0)
travel_proportions = clt.load_json_new_dict(travel_proportions_filepath)

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
R1 = flu.FluSubpopModel(compartments_epi_metrics_dict1,
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(bit_generator),
                           name="R1")

R2 = flu.FluSubpopModel(compartments_epi_metrics_dict2,
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(jumped_bit_generator),
                           name="R2")

R3 = flu.FluSubpopModel(compartments_epi_metrics_dict3,
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(bit_generator),
                           name="R3")

R4 = flu.FluSubpopModel(compartments_epi_metrics_dict4,
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(jumped_bit_generator),
                           name="R4")
R5 = flu.FluSubpopModel(compartments_epi_metrics_dict5,
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(bit_generator),
                           name="R5")

R6 = flu.FluSubpopModel(compartments_epi_metrics_dict6,
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(jumped_bit_generator),
                           name="R6")
R7 = flu.FluSubpopModel(compartments_epi_metrics_dict7,
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(bit_generator),
                           name="R7")

R8 = flu.FluSubpopModel(compartments_epi_metrics_dict8,
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(jumped_bit_generator),
                           name="R8")
R9 = flu.FluSubpopModel(compartments_epi_metrics_dict9,
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(bit_generator),
                           name="R9")

R10 = flu.FluSubpopModel(compartments_epi_metrics_dict10,
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(jumped_bit_generator),
                           name="R10")
R11 = flu.FluSubpopModel(compartments_epi_metrics_dict11,
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(jumped_bit_generator),
                           name="R11")

# The structure of the code allows us to access
#   the current state and fixed parameters of each
#   subpopulation model.
# For example, here we print out the fixed parameter
#   value for beta_baseline for the "south" subpopulation
print(R1.params.beta_baseline)
# 1

# We can also manually change a fixed parameter value
#   after we have created a SubpopModel -- like so...
# Note that this is quite a large and unrealistic value of
#   beta_baseline, but we'll use this to create
#   a dramatic difference between the two subpopulations
R1.params.beta_baseline = 10

###########################################################
############# CREATE METAPOPULATION MODEL #################
###########################################################

# Create FluInterSubpopRepo instance that manages the subpopulation models
#   and the travel dynamics that link them together
flu_inter_subpop_repo = flu.FluInterSubpopRepo({"R1":  R1, "R2":  R2,"R3":  R3, "R4":  R4,"R5":  R5, "R6":  R6,"R7":  R7, "R8":  R8,"R9":  R9, "R10":  R10,"R11":  R11},
                                               travel_proportions["subpop_names_mapping"],
                                               travel_proportions["travel_proportions_array"])

# Combine two subpopulations into one metapopulation model (travel model)
flu_demo_model = flu.FluMetapopModel(flu_inter_subpop_repo)

# Display written forms of both subpopulation models
# Check that model inputs are properly formatted and sensible
flu_demo_model.display()
flu_demo_model.run_model_checks()

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
print(flu_demo_model.subpop_models.R1.state)

# Remember that we can easily access the objects that
#   make up our subpopulation model -- here's an
#   example of accessing the "north" subpopulation's
#   compartments
# See API references for more attribute access syntax
print(flu_demo_model.subpop_models.R1.compartments)

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
num_age_groups = flu_demo_model.subpop_models.R1.params.num_age_groups

for subpop_model in flu_demo_model.subpop_models.values():
    subpop_model.params.school_contact_matrix = np.zeros((num_age_groups, num_age_groups))
    subpop_model.params.work_contact_matrix = np.zeros((num_age_groups, num_age_groups))

flu_demo_model.simulate_until_day(100)

clt.plot_metapop_basic_compartment_history(flu_demo_model,
                                           "basic_compartment_history_no_periodicity.png")