# TODO: add vaccination time series as a Schedule --
#   for now it is constant, given in params

import torch
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import clt_base as clt
import flu_core as flu

import copy

from dataclasses import fields, is_dataclass
import torch

from flu_core import flu_torch_det_components as flu_torch

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

bit_generator = np.random.MT19937(88888)
jumped_bit_generator = bit_generator.jumped(1)


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


###########################################################
############# CREATE SUBPOPULATION MODELS #################
###########################################################

younger_subpop_params_dict = updated_dict(common_subpop_params_dict, {"beta_baseline": 5})
older_subpop_params_dict = updated_dict(common_subpop_params_dict, {"beta_baseline": 2})

# Create two subpopulation models, one for the north
#   side of the city and one for the south side of the city
# In this case, these two (toy) subpopulations have the
#   same demographics, initial compartment and epi metric values,
#   fixed parameters, and school-work calendar.
# If we wanted the "north" subpopulation and "south"
#   subpopulation to have different aforementioned values,
#   we could read in two separate sets of files -- one
#   for each subpopulation
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

init_state = copy.deepcopy(state)
init_params = copy.deepcopy(params)

init_params.beta_baseline_raw = torch.tensor([3.0, 3.0], dtype=torch.float32, requires_grad=True)
init_params.beta_baseline = torch.tensor(init_params.beta_baseline_raw.view(2, 1, 1).expand(2, 5, 1),
                                         dtype=torch.float32,
                                         requires_grad=True)

true_H_history = flu_torch.simulate(state, params, precomputed, 100, 10).clone().detach()

print(max(true_H_history.sum(axis=(1,2,3))))

optimizer = torch.optim.Adam([init_params.beta_baseline_raw], lr=0.01)

beta_baseline_opt_history = []
loss_history = []
fitting_start_time = time.time()

for i in range(100):
    optimizer.zero_grad()
    init_params.beta_baseline = init_params.beta_baseline_raw.view(2, 1, 1).expand(2, 5, 1)
    sim_result = flu_torch.simulate(init_state, init_params, precomputed, 100, 10)
    loss = torch.nn.functional.mse_loss(sim_result, true_H_history)
    loss_history.append(loss)
    loss.backward()
    optimizer.step()
    # print(init_params.beta_baseline[0, 0, 0], init_params.beta_baseline[1, 0, 0])
    beta_baseline_opt_history.append(init_params.beta_baseline.clone().detach())

print(time.time() - fitting_start_time)

breakpoint()

fitted_H_history = flu_torch.simulate(state, params, 100)

breakpoint()

plt.clf()
plt.plot(beta_baseline_opt_history, label="Estimated beta over time")
plt.axhline(y=1.5, color='b', label='True beta')
plt.legend()
plt.savefig("beta_baseline_plot.png", dpi=1200)
plt.show()

breakpoint()

plt.clf()
plt.plot(torch.sum(true_H_history, dim=(1, 2)), label="True H")
plt.plot(torch.sum(fitted_H_history.clone().detach(), dim=(1, 2)), label="Fitted H")
plt.legend()
plt.savefig("H_plot.png", dpi=1200)
plt.show()

breakpoint()
