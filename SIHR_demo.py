import clt_base as clt
import SIHR_model as SIHR

from pathlib import Path
import numpy as np

# Obtain path to folder with JSON input files
base_path = Path(__file__).parent / "SIHR_input_files"

# Get filepaths for initial values of state variables, fixed parameters,
#   and configuration
compartments_epi_metrics_init_vals_filepath = base_path / "compartments_epi_metrics_init_vals.json"
params_filepath = base_path / "common_params.json"
config_filepath = base_path / "config.json"

compartments_epi_metrics_dict = clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath)
params_dict = clt.load_json_new_dict(params_filepath)
config_dict = clt.load_json_new_dict(config_filepath)

config_dict["transition_type"] = "binomial_deterministic"
config_dict["timesteps_per_day"] = 1

bit_generator = np.random.MT19937(88888)

city = SIHR.SIHRSubpopModel(compartments_epi_metrics_dict,
                            params_dict,
                            config_dict,
                            np.random.Generator(bit_generator))

city.simulate_until_day(100)

clt.plot_subpop_basic_compartment_history(city, savefig_filename="city_compartment_history.png")
