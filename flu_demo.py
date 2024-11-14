import numpy as np
from flu_components import FluModelConstructor
from plotting import create_basic_compartment_history_plot
from pathlib import Path

base_path = Path(__file__).parent / "flu_demo_input_files"

immunoseirs_config_filepath = base_path / "config.json"
immunoseirs_fixed_params_filepath = base_path / "fixed_params.json"
immunoseirs_epi_compartments_state_vars_init_vals_filepath = base_path / "state_variables_init_vals.json"

immunoseirs_constructor = FluModelConstructor(immunoseirs_config_filepath,
                                              immunoseirs_fixed_params_filepath,
                                              immunoseirs_epi_compartments_state_vars_init_vals_filepath)

immunoseirs_model = immunoseirs_constructor.create_transmission_model(np.random.SeedSequence())

immunoseirs_model.simulate_until_time_period(200)

create_basic_compartment_history_plot(immunoseirs_model)

breakpoint()