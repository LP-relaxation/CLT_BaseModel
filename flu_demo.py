import numpy as np
from flu_components import ImmunoSEIRSConstructor
from plotting import create_basic_compartment_history_plot

from pathlib import Path

base_path = Path(__file__).parent / "instance1_1age_1risk_test"

immunoseirs_config_filepath = base_path / "config.json"
immunoseirs_epi_params_filepath = base_path / "epi_params.json"
immunoseirs_epi_compartments_state_vars_init_vals_filepath = base_path / "epi_compartments_state_vars_init_vals.json"

immunoseirs_constructor = ImmunoSEIRSConstructor(immunoseirs_config_filepath,
                                                 immunoseirs_epi_params_filepath,
                                                 immunoseirs_epi_compartments_state_vars_init_vals_filepath)

immunoseirs_model = immunoseirs_constructor.create_transmission_model(np.random.SeedSequence())

immunoseirs_model.simulate_until_time_period(365)

create_basic_compartment_history_plot(immunoseirs_model)

breakpoint()