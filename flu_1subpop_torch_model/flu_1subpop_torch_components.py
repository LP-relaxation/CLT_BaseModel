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
import matplotlib.pyplot as plt

from collections import defaultdict
from dataclasses import dataclass, fields

base_path = Path(__file__).parent / "flu_1subpop_torch_input_files"

# The update rule for immunity is
#   - dM/dt = (R_to_S_rate * R / N) * (1 - inf_induced_saturation * M - vax_induced_saturation * M_v)
#   - dMv/dt = (new vaccinations at time t - delta)/ N - vax_induced_immune_wane


def to_tensor(x, requires_grad):
    if x is None:
        return None
    return torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad)


def auto_tensor_dict(d: dict,
                     requires_grad: bool=True) -> dict:
    """
    Converts dictionary entries to `tensor` (of type `torch.float32`)
        and turns on gradient tracking for each entry -- returns new dictionary.
    """

    def to_tensor(v):
        if v is None:
            return None
        return torch.tensor(v, dtype=torch.float32, requires_grad=requires_grad)
    return {k: to_tensor(v) for k, v in d.items()}


@dataclass
class Params:

    dt: torch.Tensor = None
    num_age_groups: int = 1
    num_risk_groups: int = 1
    total_pop: torch.Tensor = None

    beta_baseline: torch.Tensor = None
    humidity_impact: torch.Tensor = None

    R_to_S_rate: torch.Tensor = None
    E_to_I_rate: torch.Tensor = None
    IP_to_IS_rate: torch.Tensor = None
    IS_to_R_rate: torch.Tensor = None
    IA_to_R_rate: torch.Tensor = None
    IS_to_H_rate: torch.Tensor = None
    H_to_R_rate: torch.Tensor = None
    H_to_D_rate: torch.Tensor = None

    E_to_IA_prop: torch.Tensor = None
    H_to_D_adjusted_prop: torch.Tensor = None
    IS_to_H_adjusted_prop: torch.Tensor = None

    inf_induced_saturation: torch.Tensor = None
    inf_induced_immune_wane: torch.Tensor = None
    inf_induced_inf_risk_constant: torch.Tensor = None
    inf_induced_hosp_risk_constant: torch.Tensor = None
    inf_induced_death_risk_constant: torch.Tensor = None

    vax_induced_saturation: torch.Tensor = None
    vax_induced_immune_wane: torch.Tensor = None
    vax_induced_inf_risk_constant: torch.Tensor = None
    vax_induced_hosp_risk_constant: torch.Tensor = None
    vax_induced_death_risk_constant: torch.Tensor = None

    vaccines_per_day: torch.Tensor = None

    total_contact_matrix: torch.Tensor = None

    IP_relative_inf: torch.Tensor = None
    IA_relative_inf: torch.Tensor = None


@dataclass
class State:

    S: torch.Tensor
    E: torch.Tensor
    IP: torch.Tensor
    IS: torch.Tensor
    IA: torch.Tensor
    H: torch.Tensor
    R: torch.Tensor
    D: torch.Tensor
    M: torch.Tensor
    Mv: torch.Tensor


humidity_df = pd.read_csv(base_path / "humidity_austin_2023_2024.csv")
humidity_df["date"] = pd.to_datetime(humidity_df["date"], format='%m/%d/%y').dt.date


def compute_beta_adjusted(params: Params, timestep_counter: int) -> torch.tensor:

    absolute_humidity = humidity_df.iloc[int(np.floor(timestep_counter * params.dt))]["humidity"]

    beta_adjusted = params.beta_baseline * (1 + params.humidity_impact * np.exp(-180 * absolute_humidity))

    return beta_adjusted


def compute_wtd_infected(state: State, params: Params) -> torch.tensor:

    return params.IA_relative_inf * state.IA + params.IP_relative_inf + state.IP + state.IS


def compute_S_to_E(state: State, params: Params, timestep_counter: int) -> torch.tensor:

    beta_adjusted = compute_beta_adjusted(params, timestep_counter)

    wtd_infected = compute_wtd_infected(state, params)

    S_to_E = params.dt * state.S * beta_adjusted * \
             torch.matmul(params.total_contact_matrix, torch.sum(wtd_infected, dim=1)) / \
             (params.total_pop * (1 + params.inf_induced_inf_risk_constant * state.M +
                                  params.vax_induced_inf_risk_constant * state.Mv))

    return S_to_E


def compute_E_to_IP(state: State, params: Params) -> torch.tensor:

    E_to_IP = params.dt * state.E * params.E_to_I_rate * (1 - params.E_to_IA_prop)

    return E_to_IP


def compute_E_to_IA(state: State, params: Params) -> torch.tensor:

    E_to_IA = params.dt * state.E * params.E_to_I_rate * params.E_to_IA_prop

    return E_to_IA


def compute_IP_to_IS(state: State, params: Params) -> torch.tensor:

    IP_to_IS = params.dt * state.IP * params.IP_to_IS_rate

    return IP_to_IS


def compute_IS_to_R(state: State, params: Params) -> torch.tensor:

    IS_to_R = params.dt * state.IS * params.IS_to_R_rate * (1 - params.IS_to_H_adjusted_prop)

    return IS_to_R


def compute_IS_to_H(state: State, params: Params) -> torch.tensor:

    IS_to_H = params.dt * state.IS * params.IS_to_H_rate * params.IS_to_H_adjusted_prop / \
              (1 + params.inf_induced_hosp_risk_constant * state.M +
               params.vax_induced_hosp_risk_constant * state.Mv)

    return IS_to_H


def compute_IA_to_R(state: State, params: Params) -> torch.tensor:

    IA_to_R = params.dt * state.IA * params.IA_to_R_rate

    return IA_to_R


def compute_H_to_R(state: State, params: Params) -> torch.tensor:

    H_to_R = params.dt * state.H * params.H_to_R_rate * (1 - params.H_to_D_adjusted_prop)

    return H_to_R


def compute_H_to_D(state: State, params: Params) -> torch.tensor:

    H_to_D = params.dt * state.H * params.H_to_D_rate * params.H_to_D_adjusted_prop / \
             (1 + params.inf_induced_death_risk_constant * state.M +
              params.vax_induced_death_risk_constant * state.Mv)

    return H_to_D


def compute_R_to_S(state: State, params: Params) -> torch.tensor:

    R_to_S = params.dt * state.R * params.R_to_S_rate

    return R_to_S


def compute_M_change(state: State, params: Params) -> torch.tensor:

    M_change = ((state.R * params.dt * params.R_to_S_rate / params.total_pop) * \
                (1 - params.inf_induced_saturation * state.M - params.vax_induced_saturation * state.Mv) - \
                params.inf_induced_immune_wane * state.M) * params.dt

    return M_change


def compute_Mv_change(state: State, params: Params) -> torch.tensor:

    Mv_change = (params.vaccines_per_day / params.total_pop - \
                 params.vax_induced_immune_wane * state.Mv) * params.dt

    return Mv_change


def step(state: State, params: Params, timestep_counter: int):

    S_to_E = compute_S_to_E(state, params, timestep_counter)

    E_to_IP = compute_E_to_IP(state, params)

    E_to_IA = compute_E_to_IA(state, params)

    IP_to_IS = compute_IP_to_IS(state, params)

    IS_to_R = compute_IS_to_R(state, params)

    IS_to_H = compute_IS_to_H(state, params)

    IA_to_R = compute_IA_to_R(state, params)

    H_to_R = compute_H_to_R(state, params)

    H_to_D = compute_H_to_D(state, params)

    R_to_S = compute_R_to_S(state, params)

    M_change = compute_M_change(state, params)

    Mv_change = compute_Mv_change(state, params)

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

    return State(S=S_new, E=E_new, IP=IP_new, IS=IS_new, IA=IA_new, H=H_new, R=R_new, D=D_new, M=M_new, Mv=Mv_new)


def simulate_full_history(init_state: State, params: Params, num_timesteps: int) -> dict:

    """
    Not autodiff compatible
    """
    history_dict = defaultdict(list)

    for timestep in range(num_timesteps):
        state = step(state, params, timestep)

        for field in fields(state):
            history_dict[str(field.name)].append(state.H.clone())

    return history_dict


def simulate(init_state: State, params: Params, num_timesteps: int) -> torch.Tensor:

    history = []

    state = init_state

    for timestep in range(num_timesteps):
        state = step(state, params, timestep)
        history.append(state.H.clone())

    return torch.stack(history)