###################################################################################
######################## MetroFluSim: pytorch implementation ######################
###################################################################################

# Important notes
# ---------------
# This is the full metapopulation implementation: includes subpopulations
# L = number of locations (subpopulations)
# A = number of age groups
# R = number of risk groups
# Suffixes with some combination of the letters "L", "A", "R"
#   can be found after some function and variable names --
#   this is to make the dimensions/indices explicit to help with
#   the tensor computations

# TODO: add vaccination time series as a Schedule --
#   for now it is constant, given in params

import torch

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Union
import numbers

import time
import matplotlib.pyplot as plt

from collections import defaultdict
from dataclasses import dataclass, fields, field

from .flu_data_structures import FluMetapopStateTensors, FluMetapopParamsTensors, FluPrecomputedTensors
from .flu_travel_functions import compute_travel_wtd_infectious

base_path = Path(__file__).parent / "texas_input_files"


def to_tensor(x: np.ndarray,
              requires_grad: bool) -> torch.Tensor:
    if x is None:
        return None
    return torch.tensor(x, dtype=torch.float64, requires_grad=requires_grad)


def create_dict_of_tensors(d: dict,
                           requires_grad: bool = True) -> dict:
    """
    Converts dictionary entries to `tensor` (of type `torch.float32`)
        and turns on gradient tracking for each entry -- returns new dictionary.
    """

    def to_tensor(k, v):
        if v is None:
            return None
        else:
            return torch.tensor(v, dtype=torch.float64, requires_grad=requires_grad)

    return {k: to_tensor(k, v) for k, v in d.items()}


humidity_df = pd.read_csv(base_path / "humidity_austin_2023_2024.csv")
humidity_df["date"] = pd.to_datetime(humidity_df["date"], format="%m/%d/%y").dt.date


def compute_beta_adjusted(_state: FluMetapopStateTensors,
                          params: FluMetapopParamsTensors,
                          day_counter: int) -> torch.Tensor:
    absolute_humidity = \
        humidity_df.iloc[day_counter]["humidity"]
    beta_adjusted = params.beta_baseline * \
                    (1 + params.humidity_impact * np.exp(-180 * absolute_humidity))
    return beta_adjusted


def compute_S_to_E(state: FluMetapopStateTensors,
                   params: FluMetapopParamsTensors,
                   precomputed: FluPrecomputedTensors,
                   day_counter: int) -> torch.Tensor:

    beta_adjusted = compute_beta_adjusted(state, params, day_counter)

    travel_wtd_infectious = compute_travel_wtd_infectious(state, params, precomputed)

    if travel_wtd_infectious.size() != torch.Size([precomputed.L,
                                                precomputed.A,
                                                precomputed.R]):
        raise Exception("force_of_infection must be L x A x R corresponding \n"
                        "to number of locations (subpopulations), age groups, \n"
                        "and risk groups.")

    # print("FOI", force_of_infection.sum())

    S_to_E = state.S * beta_adjusted * travel_wtd_infectious / \
             (precomputed.total_pop_LAR * (1 + params.inf_induced_inf_risk_reduce * state.M +
                                           params.vax_induced_inf_risk_reduce * state.Mv))

    return S_to_E


def compute_E_to_IP(state: FluMetapopStateTensors, params: FluMetapopParamsTensors) -> torch.Tensor:
    E_to_IP = state.E * params.E_to_I_rate * (1 - params.E_to_IA_prop)

    return E_to_IP


def compute_E_to_IA(state: FluMetapopStateTensors, params: FluMetapopParamsTensors) -> torch.Tensor:
    E_to_IA = state.E * params.E_to_I_rate * params.E_to_IA_prop

    return E_to_IA


def compute_IP_to_IS(state: FluMetapopStateTensors, params: FluMetapopParamsTensors) -> torch.Tensor:
    IP_to_IS = state.IP * params.IP_to_IS_rate

    return IP_to_IS


def compute_IS_to_R(state: FluMetapopStateTensors, params: FluMetapopParamsTensors) -> torch.Tensor:
    IS_to_R = state.IS * params.IS_to_R_rate * (1 - params.IS_to_H_adjusted_prop)

    return IS_to_R


def compute_IS_to_H(state: FluMetapopStateTensors, params: FluMetapopParamsTensors) -> torch.Tensor:
    IS_to_H = state.IS * params.IS_to_H_rate * params.IS_to_H_adjusted_prop / \
              (1 + params.inf_induced_hosp_risk_reduce * state.M +
               params.vax_induced_hosp_risk_reduce * state.Mv)

    return IS_to_H


def compute_IA_to_R(state: FluMetapopStateTensors, params: FluMetapopParamsTensors) -> torch.Tensor:
    IA_to_R = state.IA * params.IA_to_R_rate

    return IA_to_R


def compute_H_to_R(state: FluMetapopStateTensors, params: FluMetapopParamsTensors) -> torch.Tensor:
    H_to_R = state.H * params.H_to_R_rate * (1 - params.H_to_D_adjusted_prop)

    return H_to_R


def compute_H_to_D(state: FluMetapopStateTensors, params: FluMetapopParamsTensors) -> torch.Tensor:
    H_to_D = state.H * params.H_to_D_rate * params.H_to_D_adjusted_prop / \
             (1 + params.inf_induced_death_risk_reduce * state.M +
              params.vax_induced_death_risk_reduce * state.Mv)

    return H_to_D


def compute_R_to_S(state: FluMetapopStateTensors, params: FluMetapopParamsTensors) -> torch.Tensor:
    R_to_S = state.R * params.R_to_S_rate

    return R_to_S


# The update rule for immunity is
#   - dM/dt = (R_to_S_rate * R / N) * (1 - inf_induced_saturation * M - vax_induced_saturation * M_v)
#                   - inf_induced_immune_wane * state.M
#   - dMv/dt = (new vaccinations at time t - delta)/ N - vax_induced_immune_wane


def compute_M_change(state: FluMetapopStateTensors, params: FluMetapopParamsTensors, precomputed: FluPrecomputedTensors) -> torch.Tensor:
    M_change = (params.R_to_S_rate * state.R / precomputed.total_pop_LAR) * \
               (1 - params.inf_induced_saturation * state.M - params.vax_induced_saturation * state.Mv) - \
               params.inf_induced_immune_wane * state.M

    return M_change


def compute_Mv_change(state: FluMetapopStateTensors, params: FluMetapopParamsTensors, precomputed: FluPrecomputedTensors) -> torch.Tensor:
    Mv_change = params.daily_vaccines / precomputed.total_pop_LAR - \
                params.vax_induced_immune_wane * state.Mv

    return Mv_change


def step(state: FluMetapopStateTensors,
         params: FluMetapopParamsTensors,
         precomputed: FluPrecomputedTensors,
         day_counter: int,
         dt: float):
    # WARNING: do NOT use in-place operations such as +=
    #   on leaf tensors with requires_grad = True --
    #   this breaks the computational graph --
    #   I think we should be okay with the `force_of_infection`
    #   update because `force_of_infection` is an intermediate
    #   computation and not a leaf tensor (the model's params are
    #   leaf tensors), but here we do non-in-place operations
    #   just in case

    S_to_E = compute_S_to_E(state, params, precomputed, day_counter) * dt

    E_to_IP = compute_E_to_IP(state, params) * dt

    E_to_IA = compute_E_to_IA(state, params) * dt

    IP_to_IS = compute_IP_to_IS(state, params) * dt

    IS_to_R = compute_IS_to_R(state, params) * dt

    IS_to_H = compute_IS_to_H(state, params) * dt

    IA_to_R = compute_IA_to_R(state, params) * dt

    H_to_R = compute_H_to_R(state, params) * dt

    H_to_D = compute_H_to_D(state, params) * dt

    R_to_S = compute_R_to_S(state, params) * dt

    M_change = compute_M_change(state, params, precomputed) * dt

    Mv_change = compute_Mv_change(state, params, precomputed) * dt

    S_new = state.S + R_to_S - S_to_E

    E_new = state.E + S_to_E - E_to_IP - E_to_IA

    IP_new = state.IP + E_to_IP - IP_to_IS

    IS_new = state.IS + IP_to_IS - IS_to_R - IS_to_H

    IA_new = state.IA + E_to_IA - IA_to_R

    H_new = state.H + IS_to_H - H_to_R - H_to_D

    R_new = state.R + IS_to_R + IA_to_R + H_to_R - R_to_S

    D_new = state.D + H_to_D

    M_new = state.M + M_change

    Mv_new = state.Mv + Mv_change

    return FluMetapopStateTensors(S=S_new, E=E_new, IP=IP_new, IS=IS_new, IA=IA_new, H=H_new, R=R_new, D=D_new, M=M_new, Mv=Mv_new)


def simulate_full_history(state: FluMetapopStateTensors, params: FluMetapopParamsTensors, num_timesteps: int) -> dict:
    """
    Not autodiff compatible
    """
    history_dict = defaultdict(list)

    precomputed = FluPrecomputedTensors(state, params)

    for timestep in range(num_timesteps):
        state = step(state, params, precomputed, timestep)

        for field in fields(state):
            if field.name == "init_vals":
                continue
            history_dict[str(field.name)].append(getattr(state, field.name).clone())

    return history_dict


def simulate(state: FluMetapopStateTensors,
             params: FluMetapopParamsTensors,
             num_days: int,
             timesteps_per_day: int) -> torch.Tensor:
    history = []

    precomputed = FluPrecomputedTensors(state, params)

    dt = 1/float(timesteps_per_day)

    for day in range(num_days):
        for timestep in range(timesteps_per_day):
            state = step(state, params, precomputed, day, dt)
            history.append(state.H.clone())

    return torch.stack(history)


states_path = base_path / "init_vals.json"
with states_path.open("r") as f:
    states_data = json.load(f)
state = FluMetapopStateTensors(**create_dict_of_tensors(states_data, False))

params_path = base_path / "common_params.json"
with params_path.open("r") as f:
    params_data = json.load(f)
params = FluMetapopParamsTensors(**create_dict_of_tensors(params_data, True))

start = time.time()
true_H_history = simulate(state, params, 10, 1).clone().detach()
print(time.time() - start)

breakpoint()
