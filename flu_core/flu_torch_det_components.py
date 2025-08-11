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

import torch
import numpy as np
import pandas as pd
import clt_base as clt

import datetime

from collections import defaultdict
from dataclasses import dataclass, fields, field

from .flu_data_structures import FluFullMetapopStateTensors, FluFullMetapopParamsTensors, FluPrecomputedTensors
from .flu_travel_functions import compute_total_mixing_exposure

base_path = clt.utils.PROJECT_ROOT / "flu_instances" / "texas_input_files"


def torch_approx_binomial_probability_from_rate(rate, dt):
    return 1 - torch.exp(-rate * dt)


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


humidity_df = pd.read_csv(base_path / "absolute_humidity_austin_2023_2024.csv")
humidity_df["date"] = pd.to_datetime(humidity_df["date"], format="%m/%d/%y").dt.date


def compute_beta_adjusted(_state: FluFullMetapopStateTensors,
                          params: FluFullMetapopParamsTensors,
                          schedules: dict[torch.tensor],
                          day_counter: int) -> torch.Tensor:
    absolute_humidity = \
        humidity_df.iloc[day_counter]["humidity"]
    beta_adjusted = params.beta_baseline * \
                    (1 + params.humidity_impact * np.exp(-180 * absolute_humidity))

    return beta_adjusted


def compute_S_to_E(state: FluFullMetapopStateTensors,
                   params: FluFullMetapopParamsTensors,
                   precomputed: FluPrecomputedTensors,
                   schedules: dict[torch.tensor],
                   day_counter: int,
                   dt: float) -> torch.Tensor:
    beta_adjusted = compute_beta_adjusted(state, params, schedules, day_counter)

    total_mixing_exposure = compute_total_mixing_exposure(state, params, precomputed, schedules)

    if total_mixing_exposure.size() != torch.Size([precomputed.L,
                                                   precomputed.A,
                                                   precomputed.R]):
        raise Exception("force_of_infection must be L x A x R corresponding \n"
                        "to number of locations (subpopulations), age groups, \n"
                        "and risk groups.")

    # print("FOI", force_of_infection.sum())

    rate = beta_adjusted * total_mixing_exposure / \
           (1 + params.inf_induced_inf_risk_reduce * state.M +
            params.vax_induced_inf_risk_reduce * state.Mv)

    S_to_E = state.S * torch_approx_binomial_probability_from_rate(rate, dt)

    return S_to_E


def compute_E_to_IP(state: FluFullMetapopStateTensors,
                    params: FluFullMetapopParamsTensors,
                    dt: float) -> torch.Tensor:
    rate = params.E_to_I_rate * (1 - params.E_to_IA_prop)

    E_to_IP = state.E * torch_approx_binomial_probability_from_rate(rate, dt)

    return E_to_IP


def compute_E_to_IA(state: FluFullMetapopStateTensors,
                    params: FluFullMetapopParamsTensors,
                    dt: float) -> torch.Tensor:
    rate = params.E_to_I_rate * params.E_to_IA_prop

    E_to_IA = state.E * torch_approx_binomial_probability_from_rate(rate, dt)

    return E_to_IA


def compute_IP_to_IS(state: FluFullMetapopStateTensors,
                     params: FluFullMetapopParamsTensors,
                     dt: float) -> torch.Tensor:
    rate = params.IP_to_IS_rate

    IP_to_IS = state.IP * torch_approx_binomial_probability_from_rate(rate, dt)

    return IP_to_IS


def compute_IS_to_R(state: FluFullMetapopStateTensors,
                    params: FluFullMetapopParamsTensors,
                    dt: float) -> torch.Tensor:
    immunity_force = (1 + params.inf_induced_hosp_risk_reduce * state.M +
                      params.vax_induced_hosp_risk_reduce * state.Mv)

    rate = params.IS_to_R_rate * (1 - params.IS_to_H_adjusted_prop / immunity_force)

    IS_to_R = state.IS * torch_approx_binomial_probability_from_rate(rate, dt)

    return IS_to_R


def compute_IS_to_H(state: FluFullMetapopStateTensors,
                    params: FluFullMetapopParamsTensors,
                    dt: float) -> torch.Tensor:
    immunity_force = (1 + params.inf_induced_hosp_risk_reduce * state.M +
                      params.vax_induced_hosp_risk_reduce * state.Mv)

    rate = params.IS_to_H_rate * params.IS_to_H_adjusted_prop / immunity_force

    IS_to_H = state.IS * torch_approx_binomial_probability_from_rate(rate, dt)

    return IS_to_H


def compute_IA_to_R(state: FluFullMetapopStateTensors,
                    params: FluFullMetapopParamsTensors,
                    dt: float) -> torch.Tensor:
    rate = params.IA_to_R_rate

    IA_to_R = state.IA * torch_approx_binomial_probability_from_rate(rate, dt)

    return IA_to_R


def compute_H_to_R(state: FluFullMetapopStateTensors,
                   params: FluFullMetapopParamsTensors,
                   dt: float) -> torch.Tensor:
    immunity_force = (1 + params.inf_induced_death_risk_reduce * state.M +
                      params.vax_induced_death_risk_reduce * state.Mv)

    rate = params.H_to_R_rate * (1 - params.H_to_D_adjusted_prop / immunity_force)

    H_to_R = state.H * torch_approx_binomial_probability_from_rate(rate, dt)

    return H_to_R


def compute_H_to_D(state: FluFullMetapopStateTensors,
                   params: FluFullMetapopParamsTensors,
                   dt: float) -> torch.Tensor:
    immunity_force = (1 + params.inf_induced_death_risk_reduce * state.M +
                      params.vax_induced_death_risk_reduce * state.Mv)

    rate = params.H_to_D_rate * params.H_to_D_adjusted_prop / immunity_force

    H_to_D = state.H * torch_approx_binomial_probability_from_rate(rate, dt)

    return H_to_D


def compute_R_to_S(state: FluFullMetapopStateTensors,
                   params: FluFullMetapopParamsTensors,
                   dt: float) -> torch.Tensor:
    rate = params.R_to_S_rate

    R_to_S = state.R * torch_approx_binomial_probability_from_rate(rate, dt)

    return R_to_S


# The update rule for immunity is
#   - dM/dt = (R_to_S_rate * R / N) * (1 - inf_induced_saturation * M - vax_induced_saturation * M_v)
#                   - inf_induced_immune_wane * state.M
#   - dMv/dt = (new vaccinations at time t - delta)/ N - vax_induced_immune_wane


def compute_M_change(state: FluFullMetapopStateTensors, params: FluFullMetapopParamsTensors,
                     precomputed: FluPrecomputedTensors,
                     dt: float) -> torch.Tensor:
    M_change = (params.R_to_S_rate * state.R / precomputed.total_pop_LAR) * \
               (1 - params.inf_induced_saturation * state.M - params.vax_induced_saturation * state.Mv) - \
               params.inf_induced_immune_wane * state.M

    return M_change * dt


def compute_Mv_change(state: FluFullMetapopStateTensors,
                      params: FluFullMetapopParamsTensors,
                      precomputed: FluPrecomputedTensors,
                      dt: float) -> torch.Tensor:
    Mv_change = params.daily_vaccines / precomputed.total_pop_LAR - \
                params.vax_induced_immune_wane * state.Mv

    return Mv_change * dt


# """
# Parameters:
#     start_real_date (datetime.date):
#         actual date that aligns with the beginning of the simulation
#     schedules_spec (dict[pd.DataFrame]):
#         `dict` of `DataFrame` objects
#         keys must be these strings:
#             "absolute_humidity",
#             "flu_contact_matrix",
#             "daily_vaccines"
#         (keys correspond to fields in `FluSubpopState`
#         associated with `Schedule` instances)
#         dataframe associated with "absolute_humidity" must
#             have columns "date" and "absolute_humidity" -- "date" entries must
#             correspond to consecutive calendar days and must either
#             be strings with `"YYYY-MM-DD"` format or `datetime.date`
#             objects -- "value" entries correspond to absolute humidity
#             on those days
#         dataframe associated with "flu_contact_matrix" must
#             have columns "date", "is_school_day", and "is_work_day" --
#             "date" entries must correspond to consecutive calendar days
#             and must either be strings with `"YYYY-MM-DD"` format or
#             `datetime.date` object and "is_school_day" and "is_work_day"
#             entries are Booleans indicating if that date is a school
#             day or work day
#         dataframe associated with "daily_vaccines" must have
#             columns "date" and "daily_vaccines" -- "date" entries must
#             correspond to consecutive calendar days and must either
#             be strings with `"YYYY-MM-DD"` format or `datetime.date`
#             objects -- "value" entries correspond to historical
#             number vaccinated on those days
# """


def step(state: FluFullMetapopStateTensors,
         params: FluFullMetapopParamsTensors,
         schedules: dict[torch.tensor],
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

    S_to_E = compute_S_to_E(state, params, precomputed, schedules, day_counter, dt)

    E_to_IP = compute_E_to_IP(state, params, dt)

    E_to_IA = compute_E_to_IA(state, params, dt)

    IP_to_IS = compute_IP_to_IS(state, params, dt)

    IS_to_R = compute_IS_to_R(state, params, dt)

    IS_to_H = compute_IS_to_H(state, params, dt)

    IA_to_R = compute_IA_to_R(state, params, dt)

    H_to_R = compute_H_to_R(state, params, dt)

    H_to_D = compute_H_to_D(state, params, dt)

    R_to_S = compute_R_to_S(state, params, dt)

    M_change = compute_M_change(state, params, precomputed, dt)

    Mv_change = compute_Mv_change(state, params, precomputed, schedules, dt)

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

    state_new = FluFullMetapopStateTensors(S=S_new,
                                           E=E_new,
                                           IP=IP_new,
                                           IS=IS_new,
                                           IA=IA_new,
                                           H=H_new,
                                           R=R_new,
                                           D=D_new,
                                           M=M_new,
                                           Mv=Mv_new,
                                           flu_contact_matrix=state.flu_contact_matrix)

    calibration_targets = {}
    calibration_targets["IS_to_H"] = IS_to_H

    return state_new, calibration_targets


def simulate_full_history(state: FluFullMetapopStateTensors,
                          params: FluFullMetapopParamsTensors,
                          precomputed: FluPrecomputedTensors,
                          schedules: torch.tensor,
                          num_timesteps: int) -> dict:
    history_dict = defaultdict(list)

    for timestep in range(num_timesteps):
        state, calibration_targets = step(state, params, precomputed, schedules, timestep)

        for field in fields(state):
            if field.name == "init_vals":
                continue
            history_dict[str(field.name)].append(getattr(state, field.name).clone())

        for key, value in calibration_targets.items():
            history_dict[key] = value

    return history_dict


def simulate_hospital_admits(state: FluFullMetapopStateTensors,
                             params: FluFullMetapopParamsTensors,
                             precomputed: FluPrecomputedTensors,
                             schedules: dict[torch.tensor],
                             num_days: int,
                             timesteps_per_day: int) -> torch.Tensor:
    hospital_admits_history = []

    dt = 1 / float(timesteps_per_day)

    for day in range(num_days):
        for timestep in range(timesteps_per_day):
            state, calibration_targets = step(state, params, precomputed, schedules, day, dt)
        hospital_admits_history.append(calibration_targets["IS_to_H"].clone())

    return torch.stack(hospital_admits_history)
