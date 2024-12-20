import numpy as np
from pathlib import Path
import time

from flu_components import FluModelConstructor
from plotting import create_basic_compartment_history_plot
import matplotlib.pyplot as plt

# Obtain path to folder with JSON input files
base_path = Path(__file__).parent / "flu_demo_input_files"

# Get filepaths for configuration, fixed parameter values, and
#   initial values of state variables
config_filepath = base_path / "config.json"
fixed_params_filepath = base_path / "fixed_params.json"
state_vars_init_vals_filepath = base_path / "state_variables_init_vals.json"

# Create a constructor using these filepaths
flu_demo_constructor = \
    FluModelConstructor(config_filepath,
                        fixed_params_filepath,
                        state_vars_init_vals_filepath)

# Create TransmissionModel instance from the constructor,
#   using a random number generator with starting seed 888888
#   to generate random variables
flu_demo_model = flu_demo_constructor.create_transmission_model(888888)


start = time.time()
# Simulate 300 days
flu_demo_model.simulate_until_time_period(100)
end = time.time()
print(end - start)
#print(flu_demo_model.epi_metrics)
#for compartment in flu_demo_model.epi_metrics:
#    print(compartment.name)

#print(flu_demo_model.epi_metrics[2].history_vals_list)
print(flu_demo_model.lookup_by_name["wastewater"].history_vals_list)
#ww = flu_demo_model.epi_metrics[2].history_vals_list
ww = flu_demo_model.lookup_by_name["wastewater"].history_vals_list
print(ww)
plt.plot(ww)
plt.grid(True)
plt.show()
#print(flu_demo_model.epi_metric_lookup["wastewater"].history_vals_list)

# Plot
#create_basic_compartment_history_plot(flu_demo_model,
#                                      "flu_demo_model_ww.png")