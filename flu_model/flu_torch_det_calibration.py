#############################################################################
######################## S-E-IP-IS-IA-H-R-D-M-Mv Model ######################
#############################################################################

# This code has Remy's humidity (seasonal forcing) functionality
# Implements linear dM/dt (Anass's new proposal)

# Note that we should probably avoid using the term "linear saturation"
#   as this is unclear and could cause confusion -- "saturation"
#   usually implies a FLATTENING response and saturation is nonlinear by definition

# TODO: add vaccination time series as a Schedule --
#   for now it is constant, given in params

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time

from collections import defaultdict
import matplotlib.pyplot as plt
from dataclasses import dataclass, fields

import flu_torch_det_components as flu_torch

base_path = Path(__file__).parent / "flu_torch_input_files"

# The update rule for immunity is
#   - dM/dt = (R_to_S_rate * R / N) * (1 - inf_induced_saturation * M - vax_induced_saturation * M_v)
#   - dMv/dt = (new vaccinations at time t - delta)/ N - vax_induced_immune_wane


states_path = base_path / "init_vals.json"
with states_path.open("r") as f:
    states_data = json.load(f)
state = flu_torch.State(**flu_torch.create_dict_of_tensors(states_data, True))

states_indices_path = base_path / "init_vals_indices.json"
with states_indices_path.open("r") as f:
    states_indices = json.load(f)

params_path = base_path / "params.json"
with params_path.open("r") as f:
    params_data = json.load(f)
params = flu_torch.Params(**flu_torch.create_dict_of_tensors(params_data, True))

params_indices_path = base_path / "params_indices.json"
with params_indices_path.open("r") as f:
    params_indices = json.load(f)

flu_torch.standardize_shapes(state,
                             states_indices,
                             params,
                             params_indices)

# state.save_current_vals_as_init_vals()

# params.beta_baseline = torch.tensor(1.0538, dtype=torch.float32, requires_grad=True)
# params.inf_induced_hosp_risk_reduce = torch.tensor(0.9880, dtype=torch.float32, requires_grad=True)
# params.vax_induced_hosp_risk_reduce = torch.tensor(0.9235, dtype=torch.float32, requires_grad=True)

true_H_history = flu_torch.simulate(state, params, 100).clone().detach()

# full_history = flu_torch.simulate_full_history(state, params, 100)
#
# full_history_arrays = {}
#
# for key, history_list in full_history.items():
#     full_history_arrays[key] = torch.tensor([t.sum().item() for t in history_list])
#
# print(np.average(1.0538/(full_history_arrays["Mv"]*0.9880 + full_history_arrays["M"]*0.9235)))
#
# state.reset_to_init_vals()
#
# params.beta_baseline = torch.tensor(1.5, dtype=torch.float32, requires_grad=True)
# params.inf_induced_hosp_risk_reduce = torch.tensor(1, dtype=torch.float32, requires_grad=True)
# params.vax_induced_hosp_risk_reduce = torch.tensor(1, dtype=torch.float32, requires_grad=True)
#
# full_history = flu_torch.simulate_full_history(state, params, 100)
#
# full_history_arrays = {}
#
# for key, history_list in full_history.items():
#     full_history_arrays[key] = torch.tensor([t.sum().item() for t in history_list])
#
# print(np.average(1.5/(full_history_arrays["Mv"]*1 + full_history_arrays["M"]*1)))

# plt.clf()
# plt.plot(full_history_arrays["H"], label="H")
# plt.plot(full_history_arrays["R"], label="R")
# plt.legend()
# plt.show()
#
# plt.clf()
# plt.plot(full_history_arrays["Mv"], label="Mv")
# plt.plot(full_history_arrays["M"], label="M")
# plt.legend()
# plt.show()

breakpoint()

# state.reset_to_init_vals()

state = flu_torch.State(**flu_torch.create_dict_of_tensors(states_data, True))

flu_torch.standardize_shapes(state,
                             states_indices,
                             params,
                             params_indices)

print(params.beta_baseline)
print(params.inf_induced_hosp_risk_reduce)
print(params.vax_induced_hosp_risk_reduce)

params.beta_baseline = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
params.inf_induced_hosp_risk_reduce = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
params.vax_induced_hosp_risk_reduce = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)

optimizer = torch.optim.Adam([params.beta_baseline,
                              params.inf_induced_hosp_risk_reduce,
                              params.vax_induced_hosp_risk_reduce], lr=0.01)

beta_baseline_opt_history = []
inf_induced_hosp_risk_reduce_opt_history = []
vax_induced_hosp_risk_reduce_opt_history = []

fitting_start_time = time.time()

for i in range(100):
    optimizer.zero_grad()
    sim_result = flu_torch.simulate(state, params, 100)
    loss = torch.nn.functional.mse_loss(sim_result, true_H_history)
    print(loss)
    loss.backward()
    optimizer.step()
    beta_baseline_opt_history.append(params.beta_baseline.clone().detach())
    inf_induced_hosp_risk_reduce_opt_history.append(params.inf_induced_hosp_risk_reduce.clone().detach())
    vax_induced_hosp_risk_reduce_opt_history.append(params.vax_induced_hosp_risk_reduce.clone().detach())

print(time.time() - fitting_start_time)

state.reset_to_init_vals()
params.beta_baseline = torch.tensor(0.6560, dtype=torch.float32, requires_grad=True)
params.inf_induced_hosp_risk_reduce = torch.tensor(0.8790, dtype=torch.float32, requires_grad=True)
params.vax_induced_hosp_risk_reduce = torch.tensor(0.9016, dtype=torch.float32, requires_grad=True)

fitted_H_history = flu_torch.simulate(state, params, 200)

breakpoint()

plt.clf()
plt.plot(beta_baseline_opt_history, label="Estimated beta over time")
plt.axhline(y = 1.5, color = 'b', label = 'True beta')
plt.legend()
plt.savefig("beta_baseline_plot.png", dpi=1200)
plt.show()

breakpoint()

plt.clf()
plt.plot(inf_induced_hosp_risk_reduce_opt_history, label="Estimated inf_induced_hosp_risk_reduce")
plt.axhline(y = 1, color = 'b', label = 'True value')
plt.legend()
plt.savefig("inf_induced_hosp_risk_reduce_plot.png", dpi=1200)
plt.show()

breakpoint()

plt.clf()
plt.plot(vax_induced_hosp_risk_reduce_opt_history, label="Estimated vax_induced_hosp_risk_reduce")
plt.axhline(y = 4, color = 'b', label = 'True value')
plt.legend()
plt.savefig("vax_induced_hosp_risk_reduce_plot.png", dpi=1200)
plt.show()

breakpoint()

plt.clf()
plt.plot(torch.sum(true_H_history, dim=(1,2)), label="True H")
plt.plot(torch.sum(fitted_H_history.clone().detach(), dim=(1,2)), label="Fitted H")
plt.legend()
plt.savefig("H_plot.png", dpi=1200)
plt.show()

breakpoint()

