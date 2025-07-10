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

import flu_1subpop_torch_components as flu_torch

base_path = Path(__file__).parent / "flu_1subpop_torch_input_files"

# The update rule for immunity is
#   - dM/dt = (R_to_S_rate * R / N) * (1 - inf_induced_saturation * M - vax_induced_saturation * M_v)
#   - dMv/dt = (new vaccinations at time t - delta)/ N - vax_induced_immune_wane


state_path = base_path / "compartments_epi_metrics_init_vals.json"
with state_path.open("r") as f:
    state_data = json.load(f)
state = flu_torch.State(**flu_torch.auto_tensor_dict(state_data))

params_path = base_path / "params.json"
with params_path.open("r") as f:
    params_data = json.load(f)
params = flu_torch.Params(**flu_torch.auto_tensor_dict(params_data))

params.dt = 0.1

true_H_history = flu_torch.simulate(state, params, 200).clone().detach()

print(params.beta_baseline)
print(params.inf_induced_hosp_risk_constant)
print(params.vax_induced_hosp_risk_constant)

state = flu_torch.State(**flu_torch.auto_tensor_dict(state_data))
params = flu_torch.Params(**flu_torch.auto_tensor_dict(params_data))

params.dt = 0.1
params.beta_baseline = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
params.inf_induced_hosp_risk_constant = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
params.vax_induced_hosp_risk_constant = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)

optimizer = torch.optim.Adam([params.beta_baseline,
                              params.inf_induced_hosp_risk_constant,
                              params.vax_induced_hosp_risk_constant], lr=0.01)

beta_baseline_opt_history = []
inf_induced_hosp_risk_constant_opt_history = []
vax_induced_hosp_risk_constant_opt_history = []

fitting_start_time = time.time()

for i in range(500):
    optimizer.zero_grad()
    sim_result = flu_torch.simulate(state, params, 200)
    loss = torch.nn.functional.mse_loss(sim_result, true_H_history)
    print(loss)
    loss.backward()
    optimizer.step()
    beta_baseline_opt_history.append(params.beta_baseline.clone().detach())
    inf_induced_hosp_risk_constant_opt_history.append(params.inf_induced_hosp_risk_constant.clone().detach())
    vax_induced_hosp_risk_constant_opt_history.append(params.vax_induced_hosp_risk_constant.clone().detach())

print(time.time() - fitting_start_time)

state = flu_torch.State(**flu_torch.auto_tensor_dict(state_data))

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
plt.plot(beta_baseline_opt_history, label="Estimated inf_induced_hosp_risk_constant")
plt.axhline(y = 1, color = 'b', label = 'True value')
plt.legend()
plt.savefig("inf_induced_hosp_risk_constant_plot.png", dpi=1200)
plt.show()

breakpoint()

plt.clf()
plt.plot(beta_baseline_opt_history, label="Estimated vax_induced_hosp_risk_constant")
plt.axhline(y = 1, color = 'b', label = 'True value')
plt.legend()
plt.savefig("vax_induced_hosp_risk_constant_plot.png", dpi=1200)
plt.show()

breakpoint()

plt.clf()
plt.plot(torch.sum(true_H_history, dim=1), label="True H")
plt.plot(torch.sum(fitted_H_history.clone().detach(), dim=1), label="Fitted H")
plt.legend()
plt.savefig("H_plot.png", dpi=1200)
plt.show()

breakpoint()

