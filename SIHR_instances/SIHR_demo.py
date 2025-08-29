import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import clt_toolkit as clt
import SIHR_core as SIHR
import numpy as np

base_path = clt.utils.PROJECT_ROOT / "SIHR_instances" / "SIHR_input_files"

# Get filepaths for initial values of state variables, fixed parameters,
#   and simulation settings
compartments_epi_metrics_init_vals_filepath = base_path / "compartments_epi_metrics_init_vals.json"
params_filepath = base_path / "common_params.json"
simulation_settings_filepath = base_path / "simulation_settings.json"

compartments_epi_metrics_dict = clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath)
params_dict = clt.load_json_new_dict(params_filepath)
simulation_settings_dict = clt.load_json_new_dict(simulation_settings_filepath)

simulation_settings_dict["transition_type"] = "binom_deterministic"
simulation_settings_dict["timesteps_per_day"] = 1

bit_generator = np.random.MT19937(88888)

city = SIHR.SIHRSubpopModel(compartments_epi_metrics_dict,
                            params_dict,
                            simulation_settings_dict,
                            np.random.Generator(bit_generator),
                            "demo_city")

city.simulate_until_day(100)

clt.plot_subpop_basic_compartment_history(city, savefig_filename="../city_compartment_history.png")
