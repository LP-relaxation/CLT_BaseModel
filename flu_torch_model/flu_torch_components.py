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

import time
import matplotlib.pyplot as plt

from collections import defaultdict
from dataclasses import dataclass, fields

base_path = Path(__file__).parent / "flu_torch_input_files"

# The update rule for immunity is
#   - dM/dt = (R_to_S_rate * R / N) * (1 - inf_induced_saturation * M - vax_induced_saturation * M_v)
#   - dMv/dt = (new vaccinations at time t - delta)/ N - vax_induced_immune_wane


def to_tensor(x, requires_grad):
    if x is None:
        return None
    return torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad)


def auto_tensor_dict(d: dict,
                     requires_grad: bool = True,
                     non_tensor_keys: list = []) -> dict:
    """
    Converts dictionary entries to `tensor` (of type `torch.float32`)
        and turns on gradient tracking for each entry -- returns new dictionary.
    """

    def to_tensor(k, v):
        if v is None:
            return None
        if k not in non_tensor_keys:
            return torch.tensor(v, dtype=torch.float32, requires_grad=requires_grad)
        else:
            return v

    return {k: to_tensor(k, v) for k, v in d.items()}


@dataclass
class Params:
    dt: torch.Tensor = None
    num_locations: int = None
    num_age_groups: int = None
    num_risk_groups: int = None
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
    school_contact_matrix: torch.Tensor = None
    work_contact_matrix: torch.Tensor = None

    IP_relative_inf: torch.Tensor = None
    IA_relative_inf: torch.Tensor = None

    relative_suscept_by_age: torch.Tensor = None
    prop_time_away_by_age: torch.Tensor = None
    travel_proportions_array: torch.Tensor = None


@dataclass
class State:
    S: torch.Tensor = None
    E: torch.Tensor = None
    IP: torch.Tensor = None
    IS: torch.Tensor = None
    IA: torch.Tensor = None
    H: torch.Tensor = None
    R: torch.Tensor = None
    D: torch.Tensor = None
    M: torch.Tensor = None
    Mv: torch.Tensor = None


class Precomputed:
    """
    Stores precomputed quantities that are repeatedly
    used, for computational efficiency.
    """

    def __init__(self, params: Params, state: State) -> None:

        self.total_pop_LAR = torch.tensor(state.S +
                                          state.E +
                                          state.IP +
                                          state.IS +
                                          state.IA +
                                          state.H +
                                          state.R +
                                          state.D)

        self.total_pop_LA = torch.sum(self.total_pop_LAR, dim=2)

        # Remove the diagonal!
        self.nonlocal_travel_prop = params.travel_proportions_array.clone().fill_diagonal_(0.0)

        # We don't need einsum for residents traveling
        #   -- Dave and Remy helped me check this
        # \sum_{k \not = \ell} v^{\ell \rightarrow k}
        # Note we already have k \not = \ell because we set the diagonal of
        #   nonlocal_travel_prop to 0
        self.sum_residents_nonlocal_travel_prop = self.nonlocal_travel_prop.sum(dim=1)


humidity_df = pd.read_csv(base_path / "humidity_austin_2023_2024.csv")
humidity_df["date"] = pd.to_datetime(humidity_df["date"], format="%m/%d/%y").dt.date


def compute_beta_adjusted(params: Params, timestep_counter: int) -> torch.tensor:

    absolute_humidity = humidity_df.iloc[int(np.floor(timestep_counter * params.dt))]["humidity"]

    beta_adjusted = params.beta_baseline * (1 + params.humidity_impact * np.exp(-180 * absolute_humidity))

    return beta_adjusted


def compute_wtd_infected(state: State, params: Params) -> torch.tensor:

    return params.IA_relative_inf * state.IA + params.IP_relative_inf + state.IP + state.IS


def compute_S_to_E(state: State,
                   params: Params,
                   timestep_counter: int,
                   force_of_infection: torch.tensor) -> torch.tensor:

    if force_of_infection.size() != torch.Size([params.num_locations,
                                                params.num_age_groups,
                                                params.num_risk_groups]):
        raise Exception("force_of_infection must be L x A x R corresponding \n"
                        "to number of locations (subpopulations), age groups, \n"
                        "and risk groups.")

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


def compute_wtd_infectious_LA(state: State,
                              params: Params) -> torch.tensor:
    """
    Returns L x A array -- summed over risk groups
    """

    IS = torch.einsum("lar->la", state.IS)
    wtd_IP = \
        params.IP_relative_inf * torch.einsum("lar->la", state.IP)
    wtd_IA = \
        params.IA_relative_inf * torch.einsum("lar->la", state.IA)

    return IS + wtd_IP + wtd_IA


def compute_active_pop_LAR(state: State,
                           _params: Params,
                           precomputed: Precomputed) -> torch.tensor:
    """
    Active population refers to those who are not
        symptomatic infectious or hospitalized (i.e. to
        represent people who are healthy enough to be
        out and about)
    """

    # _params is not used now -- but this is included for
    #   function signature consistency with other
    #   similar computation functions

    return precomputed.total_pop_LAR - state.IS - state.H


def compute_effective_pop_LA(state: State,
                             params: Params,
                             precomputed: Precomputed) -> torch.tensor:

    active_pop_LAR = compute_active_pop_LAR(state, params, precomputed)

    # Nonlocal travel proportions is L x L
    # Active population LAR is L x A x R
    outside_visitors_LAR = torch.einsum("kl,kar->lar",
                                        precomputed.nonlocal_travel_prop,
                                        active_pop_LAR)

    # In computation, broadcast sum_residents_nonlocal_travel_prop to be L x 1 x 1
    traveling_residents_LAR = precomputed.sum_residents_nonlocal_travel_prop[:, None, None] * \
                              active_pop_LAR

    effective_pop_LA = precomputed.total_pop_LA.unsqueeze(-1) + \
        params.relative_suscept_by_age.unsqueeze(0) * \
                       torch.sum(outside_visitors_LAR + traveling_residents_LAR, dim=2).unsqueeze(-1)

    return effective_pop_LA.squeeze(2)


def compute_wtd_infectious_ratio_LLA(state: State,
                                     params: Params,
                                     precomputed: Precomputed) -> torch.tensor:
    """
    Returns L x L x A array -- element i,j,a corresponds to
        ratio of weighted infectious people in location i, age group a
        (summed over risk groups) to the effective population in location j
        (summed over risk groups) -- where, again, "effective" population
        means that the population accounts for travelers and symptomatic
        and hospitalized individuals
    """

    wtd_infectious_LA = compute_wtd_infectious_LA(state, params)

    effective_pop_LA = compute_effective_pop_LA(state, params, precomputed)

    prop_wtd_infectious = torch.einsum("ka,la->kla",
                                       wtd_infectious_LA,
                                       effective_pop_LA)

    return prop_wtd_infectious


def compute_raw_local_to_local_foi(params: Params,
                                   state: State,
                                   precomputed: Precomputed,
                                   location_ix: int) -> torch.tensor:
    """
    Raw means that this is unnormalized by `relative_suscept_by_age`
    """

    return (1 - params.prop_time_away_by_age *
            precomputed.sum_residents_nonlocal_travel_prop) * \
           torch.matmul(params.total_contact_matrix[location_ix, :, :],
                        precomputed.wtd_infectious_ratio_LLA[location_ix, location_ix, :])


def compute_raw_outside_visitors_foi(params: Params,
                                     state: State,
                                     precomputed: Precomputed,
                                     local_ix: int,
                                     visitors_ix: int) -> torch.tensor:
    """
    Computes raw (unnormalized by `relative_suscept_by_age`) force
        of infection to local_ix, due to outside visitors from
        visitors_ix
    """

    # In location dest_ix, we are looking at the visitors from
    #   origin_ix who come to dest_ix (and infect folks in dest_ix)
    return params.travel_proportions_array[visitors_ix, local_ix] * \
           torch.matmul(params.prop_day_spend_travel_by_age * params.total_contact_matrix[local_ix, :, :],
                        precomputed.wtd_infectious_ratio_LLA[visitors_ix, local_ix, :])


def compute_raw_residents_traveling_foi(params: Params,
                                        state: State,
                                        precomputed: Precomputed,
                                        local_ix: int,
                                        dest_ix: int) -> torch.tensor:
    """
    Computes raw (unnormalized by `relative_suscept_by_age`) force
        of infection to local_ix, due to residents of local_ix
        traveling to dest_ix and getting infected in dest_ix
    """

    return params.prop_day_spend_travel_by_age * \
           params.travel_proportions_array[local_ix, dest_ix] * \
           torch.matmul(params.total_contact_matrix[local_ix, :, :],
                        precomputed.wtd_infectious_ratio_LLA[dest_ix, dest_ix, :])


def step(state: State,
         params: Params,
         precomputed: Precomputed,
         timestep_counter: int):

    # WARNING: do NOT use in-place operations such as +=
    #   on leaf tensors with requires_grad = True --
    #   this breaks the computational graph --
    #   I think we should be okay with the `force_of_infection`
    #   update because `force_of_infection` is an intermediate
    #   computation and not a leaf tensor (the model's params are
    #   leaf tensors), but here we do non-in-place operations
    #   just in case

    L = params.num_locations
    A = params.num_age_groups
    R = params.num_risk_groups

    force_of_infection = torch.tensor(np.zeros((L, A, R)))

    precomputed.wtd_infectious_ratio_LLA = \
        compute_wtd_infectious_ratio_LLA(state, params, precomputed)

    for l in np.arange(L):

        raw_foi = torch.tensor(np.zeros((A, 1)))

        # local-to-local force of infection
        raw_foi = raw_foi + compute_raw_local_to_local_foi(params, state, precomputed, l)

        for k in np.arange(L):

            raw_foi = raw_foi + compute_raw_outside_visitors_foi(params, state, precomputed, l, k)

            raw_foi = raw_foi + compute_raw_residents_traveling_foi(params, state, precomputed, l, k)

        normalized_foi = params.relative_suscept_by_age * raw_foi

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
        state = step(state, params, precomputed, timestep)

        for field in fields(state):
            history_dict[str(field.name)].append(state.H.clone())

    return history_dict


def simulate(init_state: State, params: Params, num_timesteps: int) -> torch.Tensor:
    history = []

    state = init_state

    for timestep in range(num_timesteps):
        state = step(state, params, precomputed, timestep)
        history.append(state.H.clone())

    return torch.stack(history)

breakpoint()

state_path = base_path / "compartments_epi_metrics_init_vals.json"
with state_path.open("r") as f:
    state_data = json.load(f)
state = State(**auto_tensor_dict(state_data, False))

params_path = base_path / "params.json"
with params_path.open("r") as f:
    params_data = json.load(f)
params = Params(**auto_tensor_dict(params_data, True, ["num_locations", "num_age_groups", "num_risk_groups"]))

params.dt = 0.1

precomputed = Precomputed(params, state)

true_H_history = simulate(state, params, 200).clone().detach()

breakpoint()

