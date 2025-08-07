# DRAFT VERSION -- VERY MESSY AND INTERFACES
#   WILL BE STREAMLINED AND CHANGED -- BE WARNED

# TODO: add vaccination time series as a Schedule --
#   for now it is constant, given in params

#######################################
############# IMPORTS #################
#######################################

import torch
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import clt_base as clt
import flu_core as flu

import copy

from dataclasses import fields
import torch

from flu_core import flu_torch_det_components as flu_torch

############################################
############# FILE LOADING #################
############################################

params_path = clt.utils.PROJECT_ROOT / "flu_instances" / "texas_input_files"
init_vals_path = clt.utils.PROJECT_ROOT / "flu_instances" / "explore_two_betas_input_files"

younger_subpop_init_vals_filepath = init_vals_path / "younger_subpop_init_vals.json"
older_subpop_init_vals_filepath = init_vals_path / "older_subpop_init_vals.json"

common_subpop_params_filepath = params_path / "common_subpop_params.json"
mixing_params_filepath = params_path / "mixing_params.json"
config_filepath = params_path / "config.json"

calendar_filepath = params_path / "school_work_calendar.csv"
humidity_filepath = params_path / "humidity_austin_2023_2024.csv"

younger_subpop_init_vals_dict = clt.load_json_new_dict(younger_subpop_init_vals_filepath)
older_subpop_init_vals_dict = clt.load_json_new_dict(older_subpop_init_vals_filepath)

common_subpop_params_dict = clt.load_json_new_dict(common_subpop_params_filepath)
mixing_params_dict = clt.load_json_new_dict(mixing_params_filepath)
config_dict = clt.load_json_new_dict(config_filepath)

calendar_df = pd.read_csv(calendar_filepath, index_col=0)

#################################################
############# UTILITY FUNCTIONS #################
#################################################

# TODO: NEED TO ORGANIZE!

# Turn this into a utility function and put it somewhere...
#   that is not here :P
def updated_dict(base_dict, updates):
    return {**base_dict, **updates}


# Same with this...
def enable_grad(container: flu.FluFullMetapopParamsTensors):
    for f in fields(container):
        val = getattr(container, f.name)

        if isinstance(val, torch.Tensor):
            if not val.requires_grad:
                val.requires_grad_()


####################################################
############# CREATE METAPOP MODEL #################
####################################################

# Thanks for your patience on this section...
# The torch implementation requires L x A x R tensors (ouch) --
#   to streamline the interface... the user inputs A x R arrays
#   (more readable, less redundant) -- and UNDER THE HOOD
#   these are converted to L x A x R tensors
# Overall, the goal is to have the object-oriented version
#   and the torch version be as standardized as possible
#   and have the same input types -- this is under construction...
#   future updates will make it easier to set up the
#   torch simulation.

bit_generator = np.random.MT19937(88888)
jumped_bit_generator = bit_generator.jumped(1)

younger_subpop_params_dict = updated_dict(common_subpop_params_dict, {"beta_baseline": 1.5})
older_subpop_params_dict = updated_dict(common_subpop_params_dict, {"beta_baseline": 2.5})

younger = flu.FluSubpopModel(younger_subpop_init_vals_dict,
                             younger_subpop_params_dict,
                             config_dict,
                             calendar_df,
                             np.random.Generator(bit_generator),
                             humidity_filepath,
                             name="younger")

older = flu.FluSubpopModel(older_subpop_init_vals_dict,
                           older_subpop_params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(jumped_bit_generator),
                           humidity_filepath,
                           name="older")

# Combine two subpopulations into one metapopulation model (travel model)
flu_demo_model = flu.FluMetapopModel([younger, older],
                                     mixing_params_dict)

flu_demo_model.update_full_metapop_state_tensors()
flu_demo_model.update_full_metapop_params_tensors()

state = flu_demo_model.full_metapop_state_tensors
params = flu_demo_model.full_metapop_params_tensors
precomputed = flu_demo_model.precomputed

# Need to add time-varying contact matrix and
#   make the syntax same as OOP version... but this is fine for now
# In OOP version, flu contact matrix is a schedules,
#   so it gets updated as a tensor based on information in each
#   subpopulation's state
# In current torch version (in progress...), we don't have
#   schedules, and have to figure out what to do here to keep
#   everything consistent
state.flu_contact_matrix = torch.tensor(np.stack([younger.params.total_contact_matrix] * 2, axis=0))
state.flu_contact_matrix = torch.tensor(np.stack([older.params.total_contact_matrix] * 2, axis=0))

# Save the initial state!
init_state = copy.deepcopy(state)

# Need fresh copies of the state and params to pass to the optimization
opt_state = copy.deepcopy(state)
opt_params = copy.deepcopy(params)

# `beta_baseline` is L x A x R -- but we only want there to be L=2 betas
#   (one for each subpopulation) -- so, we create a new variable
#   `beta_baseline_raw` to force the optimization to only optimize over L=2
#   parameters, not have full degrees of freedom and change L x A x R betas
# There's definitely room for improvement/clarity here...
# WE MUST TELL TORCH TO TRACK THE GRADIENT ON THE PARAMETERS WE WANT TO
#   OPTIMIZE! see `requires_grad = True`
opt_params.beta_baseline_raw = torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True)
opt_params.beta_baseline = torch.tensor(opt_params.beta_baseline_raw.view(2, 1, 1).expand(2, 5, 1),
                                         dtype=torch.float32,
                                         requires_grad=True)

# Generate "true" history
true_admits_history = flu_torch.simulate_hospital_admits(state, params, precomputed, 100, 2).clone().detach()

############################################
############# OPTIMIZATION #################
############################################

# Could potentially wrap this in something nice...

optimizer = torch.optim.Adam([opt_params.beta_baseline_raw], lr=0.01)

beta_baseline_opt_history = []
loss_history = []
fitting_start_time = time.time()

for i in range(1000):
    optimizer.zero_grad()
    opt_params.beta_baseline = opt_params.beta_baseline_raw.view(2, 1, 1).expand(2, 5, 1)
    sim_result = flu_torch.simulate_hospital_admits(init_state, opt_params, precomputed, 100, 2)
    loss = torch.nn.functional.mse_loss(sim_result, true_admits_history)
    loss_history.append(loss)
    loss.backward()
    optimizer.step()
    # beta_baseline_opt_history.append(opt_params.beta_baseline_raw.clone().detach())
    if i % 10 == 0:
        print("Loss function: " + str(loss))
        print("Estimated betas: " + str(opt_params.beta_baseline_raw.clone().detach()))
    if loss < 1e-2:
        break

print(time.time() - fitting_start_time)

print(opt_params.beta_baseline)

# Optional -- can simulate with fitted parameters and plot corresponding output
# Commented out for now but can un-comment
# fitted_admits_history = flu_torch.simulate_hospital_admits(init_state, opt_params, precomputed, 100, 2)

# plt.clf()
# plt.plot(torch.sum(true_admits_history, dim=(1, 2)), label="True hospital admits")
# plt.plot(torch.sum(fitted_admits_history.clone().detach(), dim=(1, 2)), label="Fitted hospital admits")
# plt.legend()
# plt.savefig("hospital_admits_plot.png", dpi=1200)
# plt.show()
