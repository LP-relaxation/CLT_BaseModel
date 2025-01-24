# Simple demo with flu model with "toy" (not fitted or realistic) parameters
# Wastewater addition from Sonny is still in progress

from pathlib import Path
import numpy as np
import pandas as pd

# Import city-level transmission base components module
import clt_base as clt

# Import flu model module, which contains customized subclasses
import flu_model as flu

# Obtain path to folder with JSON input files
base_path = Path(__file__).parent / "flu_demo_input_files"

# Get filepaths for initial values of compartments and epi metrics, fixed parameters,
#   and configuration
compartments_epi_metrics_init_vals_filepath = base_path / "compartments_epi_metrics_init_vals.json"
params_filepath = base_path / "common_params.json"
config_filepath = base_path / "config.json"

# Get filepaths for school-work calendar CSV and travel proportions CSV
calendar_filepath = base_path / "school_work_calendar.csv"
travel_proportions_filepath = base_path / "travel_proportions.csv"

# Read in files as dictionaries and dataframes
state_dict = clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath)
params_dict = clt.load_json_new_dict(params_filepath)
config_dict = clt.load_json_new_dict(config_filepath)

calendar_df = pd.read_csv(calendar_filepath, index_col=0)
travel_proportions_df = pd.read_csv(travel_proportions_filepath)

# Create two independent bit generators
bit_generator = np.random.MT19937(88888)
jumped_bit_generator = bit_generator.jumped(1)

# Create two subpopulation models, one for the north
#   side of the city and one for the south side of the city
# In this case, these two (toy) subpopulations have the
#   same demographics, initial compartment and epi metric values,
#   fixed parameters, and school-work calendar.
north = flu.FluSubpopModel(state_dict,
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(bit_generator),
                           name="north")

south = flu.FluSubpopModel(state_dict,
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(jumped_bit_generator),
                           name="south")

# Combine two subpopulations into one metapopulation model (travel model)
flu_demo_model = flu.FluMetapopModel({"north": north, "south": south},
                                     travel_proportions_df)

# Display written forms of both subpopulation models
# Check that model inputs are properly formatted and sensible
flu_demo_model.display()
flu_demo_model.run_model_checks()

# Simulate for 100 days
flu_demo_model.simulate_until_time_period(100)

# Generate different types of plots for flu model
clt.plot_metapop_epi_metrics(flu_demo_model)
clt.plot_metapop_total_infected_deaths(flu_demo_model)
clt.plot_metapop_basic_compartment_history(flu_demo_model)
