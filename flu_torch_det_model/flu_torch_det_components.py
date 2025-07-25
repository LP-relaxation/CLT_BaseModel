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

base_path = Path(__file__).parent / "flu_torch_input_files"


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


@dataclass
class Params:
    dt: float = None
    num_locations: int = None
    num_age_groups: int = None
    num_risk_groups: int = None

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

    daily_vaccines: torch.Tensor = None

    total_contact_matrix: torch.Tensor = None
    school_contact_matrix: torch.Tensor = None
    work_contact_matrix: torch.Tensor = None

    IP_relative_inf: torch.Tensor = None
    IA_relative_inf: torch.Tensor = None

    relative_suscept: torch.Tensor = None
    prop_time_away: torch.Tensor = None
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

    init_vals: dict = field(default_factory=dict)

    # Note: `init_vals: dict = {}` does NOT work --
    #   gives "mutable default" argument

    def save_current_vals_as_init_vals(self):

        for field in fields(self):
            if field.name == "init_vals":
                continue
            self.init_vals[field.name] = getattr(self, field.name).clone()

    def reset_to_init_vals(self):

        for name, val in self.init_vals.items():
            setattr(self, name, val.clone())


def standardize_shapes(state: State,
                       states_indices: dict,
                       params: Params,
                       params_indices: dict) -> None:
    """
    For all fields in `input`, if field is not a scalar or L x A x R,
        or is not a special variable listed below, then apply dimension
        expansion so that fields are L x A x R for tensor multiplication.

    Special variables that are exempted:
        - `total_contact_matrix`, `school_contact_matrix`, `work_contact_matrix`:
            all of these must be dimension A x A
        - `travel_proportions_array`: this must be L x L

    Valid values for the `indices_dict` are: "age", "age_risk",
        "location", and "location_age" -- other combinations
        are not considered because they do not make sense --
        we assume that we only have risk IF we have age, for example
    """

    L = int(params.num_locations.item())
    A = int(params.num_age_groups.item())
    R = int(params.num_risk_groups.item())

    error_str = " size does not match index specification in \n" \
                "indices dictionary -- please check files and inputs, \n" \
                "then try again."

    for dc, indices_dict in zip([state, params], [states_indices, params_indices]):
        for name, value in vars(dc).items():

            # Ignore the field that corresponds to a dictionary
            if name == "init_vals":
                continue

            # Contact matrices should be A x A
            # This includes:
            #   "school_contact_matrix",
            #   "work_contact_matrix",
            #   "travel_proportions_array"
            elif "contact_matrix" in name:
                # Need nested if-statements because user may
                #   have already converted contact matrix to L x A x A
                if value.size() != torch.Size([L, A, A]):
                    if value.size() != torch.Size([A, A]):
                        raise Exception(str(name) + error_str)
                    setattr(dc, name, value.view(1, A, A).expand(L, A, A))

            elif name == "travel_proportions_array":
                if value.size() != torch.Size([L, L]):
                    raise Exception(str(name) + error_str)

            # If scalar or already L x A x R, do not need to adjust
            #   dimensions
            elif value.size() == torch.Size([]):
                continue

            elif value.size() == torch.Size([L, A, R]):
                continue

            elif indices_dict[name] == "age":
                if value.size() != torch.Size([A]):
                    raise Exception(str(name) + error_str)
                else:
                    setattr(dc, name, value.view(1, A, 1).expand(L, A, R))

            elif getattr(indices_dict, name) == "age_risk":
                if value.size() != torch.Size([A, R]):
                    raise Exception(str(name) + error_str)
                else:
                    setattr(dc, name, value.view(1, A, R).expand(L, A, R))

            # We probably won't use this, but just in case...
            elif getattr(indices_dict, name) == "location":
                if value.size() != torch.Size([L]):
                    raise Exception(str(name) + error_str)
                else:
                    setattr(dc, name, value.view(L, 1, 1).expand(L, A, R))

            elif getattr(indices_dict, name) == "location_age":
                if value.size() != torch.Size([L, A]):
                    raise Exception(str(name) + error_str)
                else:
                    setattr(dc, name, value.view(L, A, 1).expand(L, A, R))


class Precomputed:
    """
    Stores precomputed quantities that are repeatedly
    used, for computational efficiency.
    """

    def __init__(self, state: State, params: Params) -> None:
        self.total_pop_LAR = torch.tensor(state.S +
                                          state.E +
                                          state.IP +
                                          state.IS +
                                          state.IA +
                                          state.H +
                                          state.R +
                                          state.D)

        self.L = int(params.num_locations.item())
        self.A = int(params.num_age_groups.item())
        self.R = int(params.num_risk_groups.item())

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


def compute_beta_adjusted(state: State, params: Params, timestep_counter: int) -> torch.Tensor:
    absolute_humidity = humidity_df.iloc[int(np.floor(timestep_counter * params.dt.item()))]["humidity"]

    beta_adjusted = params.beta_baseline * (1 + params.humidity_impact * np.exp(-180 * absolute_humidity))

    return beta_adjusted


def compute_wtd_infected(state: State, params: Params) -> torch.Tensor:
    return params.IA_relative_inf * state.IA + params.IP_relative_inf + state.IP + state.IS


def compute_S_to_E(state: State,
                   params: Params,
                   precomputed: Precomputed,
                   timestep_counter: int) -> torch.Tensor:
    force_of_infection = compute_total_foi(state, params, precomputed, timestep_counter)

    if force_of_infection.size() != torch.Size([precomputed.L,
                                                precomputed.A,
                                                precomputed.R]):
        raise Exception("force_of_infection must be L x A x R corresponding \n"
                        "to number of locations (subpopulations), age groups, \n"
                        "and risk groups.")

    # print("FOI", force_of_infection.sum())

    S_to_E = params.dt * state.S * force_of_infection / \
             (precomputed.total_pop_LAR * (1 + params.inf_induced_inf_risk_constant * state.M +
                                           params.vax_induced_inf_risk_constant * state.Mv))

    return S_to_E


def compute_E_to_IP(state: State, params: Params) -> torch.Tensor:
    E_to_IP = params.dt * state.E * params.E_to_I_rate * (1 - params.E_to_IA_prop)

    return E_to_IP


def compute_E_to_IA(state: State, params: Params) -> torch.Tensor:
    E_to_IA = params.dt * state.E * params.E_to_I_rate * params.E_to_IA_prop

    return E_to_IA


def compute_IP_to_IS(state: State, params: Params) -> torch.Tensor:
    IP_to_IS = params.dt * state.IP * params.IP_to_IS_rate

    return IP_to_IS


def compute_IS_to_R(state: State, params: Params) -> torch.Tensor:
    IS_to_R = params.dt * state.IS * params.IS_to_R_rate * (1 - params.IS_to_H_adjusted_prop)

    return IS_to_R


def compute_IS_to_H(state: State, params: Params) -> torch.Tensor:
    IS_to_H = params.dt * state.IS * params.IS_to_H_rate * params.IS_to_H_adjusted_prop / \
              (1 + params.inf_induced_hosp_risk_constant * state.M +
               params.vax_induced_hosp_risk_constant * state.Mv)

    return IS_to_H


def compute_IA_to_R(state: State, params: Params) -> torch.Tensor:
    IA_to_R = params.dt * state.IA * params.IA_to_R_rate

    return IA_to_R


def compute_H_to_R(state: State, params: Params) -> torch.Tensor:
    H_to_R = params.dt * state.H * params.H_to_R_rate * (1 - params.H_to_D_adjusted_prop)

    return H_to_R


def compute_H_to_D(state: State, params: Params) -> torch.Tensor:
    H_to_D = params.dt * state.H * params.H_to_D_rate * params.H_to_D_adjusted_prop / \
             (1 + params.inf_induced_death_risk_constant * state.M +
              params.vax_induced_death_risk_constant * state.Mv)

    return H_to_D


def compute_R_to_S(state: State, params: Params) -> torch.Tensor:
    R_to_S = params.dt * state.R * params.R_to_S_rate

    return R_to_S


# The update rule for immunity is
#   - dM/dt = (R_to_S_rate * R / N) * (1 - inf_induced_saturation * M - vax_induced_saturation * M_v)
#                   - inf_induced_immune_wane * state.M
#   - dMv/dt = (new vaccinations at time t - delta)/ N - vax_induced_immune_wane


def compute_M_change(state: State, params: Params, precomputed: Precomputed) -> torch.Tensor:
    M_change = (params.R_to_S_rate * state.R / precomputed.total_pop_LAR) * \
               (1 - params.inf_induced_saturation * state.M - params.vax_induced_saturation * state.Mv) - \
               params.inf_induced_immune_wane * state.M

    return M_change * params.dt


def compute_Mv_change(state: State, params: Params, precomputed: Precomputed) -> torch.Tensor:
    Mv_change = params.daily_vaccines / precomputed.total_pop_LAR - \
                params.vax_induced_immune_wane * state.Mv

    return Mv_change * params.dt


def compute_wtd_infectious_LA(state: State,
                              params: Params) -> torch.Tensor:
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
                           precomputed: Precomputed) -> torch.Tensor:
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
                             precomputed: Precomputed) -> torch.Tensor:
    L, A = precomputed.L, precomputed.A

    active_pop_LAR = compute_active_pop_LAR(state, params, precomputed)

    # Nonlocal travel proportions is L x L
    # Active population LAR is L x A x R
    outside_visitors_LAR = torch.einsum("kl,kar->lar",
                                        precomputed.nonlocal_travel_prop,
                                        active_pop_LAR)

    # In computation, broadcast sum_residents_nonlocal_travel_prop to be L x 1 x 1
    traveling_residents_LAR = precomputed.sum_residents_nonlocal_travel_prop[:, None, None] * \
                              active_pop_LAR

    effective_pop_LA = precomputed.total_pop_LA + \
                       params.prop_time_away[0, :, 0] * \
                       torch.sum(outside_visitors_LAR + traveling_residents_LAR, dim=2)

    assert effective_pop_LA.size() == torch.Size([L, A])

    return effective_pop_LA


def compute_wtd_infectious_ratio_LLA(state: State,
                                     params: Params,
                                     precomputed: Precomputed) -> torch.Tensor:
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
                                       1 / effective_pop_LA)

    return prop_wtd_infectious


# @torch.jit.script
def compute_raw_local_to_local_foi(prop_time_away: torch.Tensor,
                                   total_contact_matrix: torch.Tensor,
                                   sum_residents_nonlocal_travel_prop: torch.Tensor,
                                   wtd_infectious_ratio_LLA: torch.Tensor,
                                   location_ix: int) -> torch.Tensor:
    """
    Raw means that this is unnormalized by `relative_suscept`

    Excludes beta -- that is factored in later
    """

    result = (1 - prop_time_away[0, :, 0] * sum_residents_nonlocal_travel_prop[location_ix]) * \
             torch.matmul(total_contact_matrix[location_ix, :, :].double(),
                          wtd_infectious_ratio_LLA[location_ix, location_ix, :].double())

    return result


# @torch.jit.script
def compute_raw_outside_visitors_foi(prop_time_away: torch.Tensor,
                                     total_contact_matrix: torch.Tensor,
                                     travel_proportions_array: torch.Tensor,
                                     wtd_infectious_ratio_LLA: torch.Tensor,
                                     local_ix: int,
                                     visitors_ix: int) -> torch.Tensor:
    """
    Computes raw (unnormalized by `relative_suscept`) force
        of infection to local_ix, due to outside visitors from
        visitors_ix

    Excludes beta -- that is factored in later

    Output should be size A
    """

    # In location dest_ix, we are looking at the visitors from
    #   origin_ix who come to dest_ix (and infect folks in dest_ix)

    result = travel_proportions_array[visitors_ix, local_ix] * \
             torch.matmul(prop_time_away[0, :, 0] * total_contact_matrix[local_ix, :, :],
                          wtd_infectious_ratio_LLA[visitors_ix, local_ix, :])

    return result


def compute_raw_residents_traveling_foi(prop_time_away: torch.Tensor,
                                        total_contact_matrix: torch.Tensor,
                                        travel_proportions_array: torch.Tensor,
                                        wtd_infectious_ratio_LLA: torch.Tensor,
                                        local_ix: int,
                                        dest_ix: int) -> torch.Tensor:
    """
    Computes raw (unnormalized by `relative_suscept`) force
        of infection to local_ix, due to residents of local_ix
        traveling to dest_ix and getting infected in dest_ix

    Excludes beta -- that is factored in later

    Output should be size A
    """

    result = prop_time_away[0, :, 0] * travel_proportions_array[local_ix, dest_ix] * \
             torch.matmul(total_contact_matrix[local_ix, :, :],
                          wtd_infectious_ratio_LLA[dest_ix, dest_ix, :])

    return result


def compute_total_foi(state: State,
                      params: Params,
                      precomputed: Precomputed,
                      timestep_counter: int) -> torch.Tensor:
    """
    Compute total force of infection! Includes beta
    """

    L, A, R = precomputed.L, precomputed.A, precomputed.R

    prop_time_away = params.prop_time_away
    total_contact_matrix = params.total_contact_matrix
    travel_proportions_array = params.travel_proportions_array
    sum_residents_nonlocal_travel_prop = precomputed.sum_residents_nonlocal_travel_prop
    wtd_infectious_ratio_LLA = compute_wtd_infectious_ratio_LLA(state, params, precomputed)

    relative_suscept_by_age = params.relative_suscept[0, :, 0]

    foi = torch.tensor(np.zeros((L, A, R)))

    for l in np.arange(L):

        raw_foi = torch.tensor(np.zeros(A))

        # local-to-local force of infection
        raw_foi = raw_foi + compute_raw_local_to_local_foi(prop_time_away,
                                                           total_contact_matrix,
                                                           sum_residents_nonlocal_travel_prop,
                                                           wtd_infectious_ratio_LLA,
                                                           l)

        for k in np.arange(L):
            raw_foi = raw_foi + compute_raw_outside_visitors_foi(prop_time_away,
                                                                 total_contact_matrix,
                                                                 travel_proportions_array,
                                                                 wtd_infectious_ratio_LLA,
                                                                 l,
                                                                 k)

            raw_foi = raw_foi + compute_raw_residents_traveling_foi(prop_time_away,
                                                                    total_contact_matrix,
                                                                    travel_proportions_array,
                                                                    wtd_infectious_ratio_LLA,
                                                                    l,
                                                                    k)

            # print(raw_foi)

            # breakpoint()

        normalized_foi = relative_suscept_by_age * raw_foi

        foi[l, :, :] = normalized_foi.view(A, 1).expand((A, R))

    beta_adjusted = compute_beta_adjusted(state, params, timestep_counter)

    return beta_adjusted * foi


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

    S_to_E = compute_S_to_E(state, params, precomputed, timestep_counter)

    E_to_IP = compute_E_to_IP(state, params)

    E_to_IA = compute_E_to_IA(state, params)

    IP_to_IS = compute_IP_to_IS(state, params)

    IS_to_R = compute_IS_to_R(state, params)

    IS_to_H = compute_IS_to_H(state, params)

    IA_to_R = compute_IA_to_R(state, params)

    H_to_R = compute_H_to_R(state, params)

    H_to_D = compute_H_to_D(state, params)

    R_to_S = compute_R_to_S(state, params)

    M_change = compute_M_change(state, params, precomputed)

    Mv_change = compute_Mv_change(state, params, precomputed)

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


def simulate_full_history(state: State, params: Params, num_timesteps: int) -> dict:
    """
    Not autodiff compatible
    """
    history_dict = defaultdict(list)

    precomputed = Precomputed(state, params)

    for timestep in range(num_timesteps):
        state = step(state, params, precomputed, timestep)

        for field in fields(state):
            if field.name == "init_vals":
                continue
            history_dict[str(field.name)].append(getattr(state, field.name).clone())

    return history_dict


def simulate(state: State,
             params: Params,
             num_timesteps: int) -> torch.Tensor:
    history = []

    precomputed = Precomputed(state, params)

    for timestep in range(num_timesteps):
        state = step(state, params, precomputed, timestep)
        history.append(state.H.clone())

    return torch.stack(history)


# states_path = base_path / "init_vals.json"
# with states_path.open("r") as f:
#     states_data = json.load(f)
# state = State(**create_dict_of_tensors(states_data, False))
#
# states_indices_path = base_path / "init_vals_indices.json"
# with states_indices_path.open("r") as f:
#     states_indices = json.load(f)
#
# params_path = base_path / "params.json"
# with params_path.open("r") as f:
#     params_data = json.load(f)
# params = Params(**create_dict_of_tensors(params_data, True))
#
# params_indices_path = base_path / "params_indices.json"
# with params_indices_path.open("r") as f:
#     params_indices = json.load(f)
#
# standardize_shapes(state,
#                    states_indices,
#                    params,
#                    params_indices)
#
# start = time.time()
# true_H_history = simulate(state, params, 200).clone().detach()
# print(time.time() - start)

# breakpoint()
