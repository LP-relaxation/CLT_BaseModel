#######################################
############# IMPORTS #################
#######################################

import torch
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import clt_toolkit as clt
import flu_core as flu

import copy

from dataclasses import fields
import torch

from flu_core import flu_torch_det_components as flu_torch

############################################
############# FILE LOADING #################
############################################

texas_files_path = clt.utils.PROJECT_ROOT / "flu_instances" / "texas_input_files"
calibration_files_path = clt.utils.PROJECT_ROOT / "flu_instances" / "calibration_research_input_files"

subpopA_init_vals_filepath = calibration_files_path / "subpopA_init_vals.json"
subpopB_init_vals_filepath = calibration_files_path / "subpopB_init_vals.json"

common_subpop_params_filepath = texas_files_path / "common_subpop_params.json"
mixing_params_filepath = calibration_files_path / "AB_mixing_params.json"
simulation_settings_filepath = texas_files_path / "simulation_settings.json"

calendar_df = pd.read_csv(texas_files_path / "school_work_calendar.csv", index_col=0)
humidity_df = pd.read_csv(texas_files_path / "absolute_humidity_austin_2023_2024.csv", index_col=0)
vaccines_df = pd.read_csv(texas_files_path / "daily_vaccines_constant.csv", index_col=0)

schedules_info = {}
schedules_info["flu_contact_matrix"] = calendar_df
schedules_info["daily_vaccines"] = vaccines_df
schedules_info["absolute_humidity"] = humidity_df

subpopA_init_vals_dict = clt.load_json_new_dict(subpopA_init_vals_filepath)
subpopB_init_vals_dict = clt.load_json_new_dict(subpopB_init_vals_filepath)

common_subpop_params_dict = clt.load_json_new_dict(common_subpop_params_filepath)
mixing_params_dict = clt.load_json_new_dict(mixing_params_filepath)
simulation_settings_dict = clt.load_json_new_dict(simulation_settings_filepath)

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

bit_generator = np.random.MT19937(88888)
jumped_bit_generator = bit_generator.jumped(1)

subpopA_params_dict = updated_dict(common_subpop_params_dict, {"beta_baseline": 1.5})
subpopB_params_dict = updated_dict(common_subpop_params_dict, {"beta_baseline": 2.5})

subpopA = flu.FluSubpopModel(subpopA_init_vals_dict,
                             subpopA_params_dict,
                             simulation_settings_dict,
                             np.random.Generator(bit_generator),
                             schedules_info,
                             name="subpopA")

subpopB = flu.FluSubpopModel(subpopB_init_vals_dict,
                           subpopB_params_dict,
                           simulation_settings_dict,
                           np.random.Generator(jumped_bit_generator),
                           schedules_info,
                           name="subpopB")

# Combine two subpopulations into one metapopulation model (travel model)
flu_demo_model = flu.FluMetapopModel([subpopA, subpopB],
                                     mixing_params_dict)

d = flu_demo_model.get_flu_torch_inputs()

state = d["state_tensors"]
params = d["params_tensors"]
schedules = d["schedule_tensors"]
precomputed = d["precomputed"]

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
true_admits_history = flu_torch.simulate_hospital_admits(state,
                                                         params,
                                                         precomputed,
                                                         schedules,
                                                         100,
                                                         2).clone().detach()

############################################
############# OPTIMIZATION #################
############################################

# Could potentially wrap this in something nice...

optimizer = torch.optim.Adam([opt_params.beta_baseline_raw], lr=0.01)

beta_baseline_opt_history = []
loss_history = []
fitting_start_time = time.time()

for i in range(int(5e3)):
    optimizer.zero_grad()
    opt_params.beta_baseline = opt_params.beta_baseline_raw.view(2, 1, 1).expand(2, 5, 1)
    sim_result = flu_torch.simulate_hospital_admits(init_state, opt_params, precomputed, schedules, 100, 2)
    loss = torch.nn.functional.mse_loss(sim_result, true_admits_history)
    loss_history.append(loss)
    loss.backward()
    optimizer.step()
    # beta_baseline_opt_history.append(opt_params.beta_baseline_raw.clone().detach())
    if i % 50 == 0:
        print("Loss function: " + str(loss))
        print("Estimated betas: " + str(opt_params.beta_baseline_raw.clone().detach()))
    if loss < 1e-2:
        break

print(time.time() - fitting_start_time)

print(opt_params.beta_baseline)

np.savetxt("caseAB_beta.csv", [t.detach().numpy() for t in beta_baseline_opt_history], delimiter=",")
np.savetxt("caseAB_mse.csv", [t.detach().numpy() for t in loss_history], delimiter=",")

breakpoint()

# Optional -- can simulate with fitted parameters and plot corresponding output
# Commented out for now but can un-comment
# fitted_admits_history = flu_torch.simulate_hospital_admits(init_state, opt_params, precomputed, schedules, 100, 2)

# plt.clf()
# plt.plot(torch.sum(true_admits_history, dim=(1, 2)), label="True hospital admits")
# plt.plot(torch.sum(fitted_admits_history.clone().detach(), dim=(1, 2)), label="Fitted hospital admits")
# plt.legend()
# plt.savefig("hospital_admits_plot.png", dpi=1200)
# plt.show()
