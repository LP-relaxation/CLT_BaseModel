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
import clt_toolkit as clt

import datetime

from collections import defaultdict
from dataclasses import dataclass, fields, field

from .flu_data_structures import FluFullMetapopStateTensors, \
    FluFullMetapopParamsTensors, FluPrecomputedTensors, \
    FluFullMetapopScheduleTensors
from .flu_travel_functions import compute_total_mixing_exposure

base_path = clt.utils.PROJECT_ROOT / "flu_instances" / "texas_input_files"


def torch_approx_binom_probability_from_rate(rate, dt):
    return 1 - torch.exp(-rate * dt)


def to_tensor(x: np.ndarray,
              requires_grad: bool) -> torch.Tensor:
    if x is None:
        return None
    return torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad)


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
            return torch.tensor(v, dtype=torch.float32, requires_grad=requires_grad)

    return {k: to_tensor(k, v) for k, v in d.items()}


def compute_beta_adjusted(_state: FluFullMetapopStateTensors,
                          params: FluFullMetapopParamsTensors,
                          schedules: FluFullMetapopScheduleTensors,
                          day_counter: int) -> torch.Tensor:
    absolute_humidity = schedules.absolute_humidity[day_counter]
    beta_adjusted = params.beta_baseline * (1 + params.humidity_impact * np.exp(-180 * absolute_humidity))

    return beta_adjusted


def compute_flu_contact_matrix(state: FluFullMetapopStateTensors,
                               params: FluFullMetapopParamsTensors,
                               precomputed: FluPrecomputedTensors,
                               schedules: FluFullMetapopScheduleTensors,
                               day_counter: int):
    # Here, using schedules.is_school_day[day_counter][:,:,0] and similarly for
    #   is_work_day because each contact matrix (as a metapop tensor) is L x A x A --
    #   we don't use risk -- assume here that we do not have a different school/work-day
    #   schedule based on risk, so just grab the first risk group
    # But then we have to take (1 - schedules.is_school_day[day_counter][:, :, 0]), which is
    #   L x A, and then make it L x A x 1 (unsqueeze the last dimension) to make the
    #   broadcasting work (because this gets element-wise multiplied by params.school_contact_matrix)
    flu_contact_matrix = \
        params.total_contact_matrix - \
        params.school_contact_matrix * (1 - schedules.is_school_day[day_counter][:, :, 0]).unsqueeze(dim=2) - \
        params.work_contact_matrix * (1 - schedules.is_work_day[day_counter][:, :, 0]).unsqueeze(dim=2)

    return flu_contact_matrix


def compute_S_to_E(state: FluFullMetapopStateTensors,
                   params: FluFullMetapopParamsTensors,
                   precomputed: FluPrecomputedTensors,
                   schedules: FluFullMetapopScheduleTensors,
                   day_counter: int,
                   dt: float) -> torch.Tensor:
    # Needs flu_contact_matrix to be in state for this
    total_mixing_exposure = compute_total_mixing_exposure(state, params, precomputed)

    if total_mixing_exposure.size() != torch.Size([precomputed.L,
                                                   precomputed.A,
                                                   precomputed.R]):
        raise Exception("force_of_infection must be L x A x R corresponding \n"
                        "to number of locations (subpopulations), age groups, \n"
                        "and risk groups.")

    beta_adjusted = compute_beta_adjusted(state, params, schedules, day_counter)

    inf_induced_inf_risk_reduce = params.inf_induced_inf_risk_reduce
    inf_induced_proportional_risk_reduce = inf_induced_inf_risk_reduce / (1 - inf_induced_inf_risk_reduce)

    vax_induced_inf_risk_reduce = params.vax_induced_inf_risk_reduce
    vax_induced_proportional_risk_reduce = vax_induced_inf_risk_reduce / (1 - vax_induced_inf_risk_reduce)

    immune_force = (1 + inf_induced_proportional_risk_reduce * state.M +
                    vax_induced_proportional_risk_reduce * state.MV)

    rate = beta_adjusted * total_mixing_exposure / immune_force

    S_to_E = state.S * torch_approx_binom_probability_from_rate(rate, dt)

    return S_to_E


def compute_E_to_IP_rate(state: FluFullMetapopStateTensors,
                         params: FluFullMetapopParamsTensors) -> torch.Tensor:
    return params.E_to_I_rate * (1 - params.E_to_IA_prop)


def compute_E_to_IA_rate(state: FluFullMetapopStateTensors,
                         params: FluFullMetapopParamsTensors) -> torch.Tensor:
    return params.E_to_I_rate * params.E_to_IA_prop


def compute_IP_to_IS(state: FluFullMetapopStateTensors,
                     params: FluFullMetapopParamsTensors,
                     dt: float) -> torch.Tensor:
    rate = params.IP_to_IS_rate

    IP_to_IS = state.IP * torch_approx_binom_probability_from_rate(rate, dt)

    return IP_to_IS


def compute_IS_to_R_rate(state: FluFullMetapopStateTensors,
                         params: FluFullMetapopParamsTensors) -> torch.Tensor:
    inf_induced_hosp_risk_reduce = params.inf_induced_hosp_risk_reduce
    inf_induced_proportional_risk_reduce = inf_induced_hosp_risk_reduce / (1 - inf_induced_hosp_risk_reduce)

    vax_induced_hosp_risk_reduce = params.vax_induced_hosp_risk_reduce
    vax_induced_proportional_risk_reduce = vax_induced_hosp_risk_reduce / (1 - vax_induced_hosp_risk_reduce)

    immunity_force = (1 + inf_induced_proportional_risk_reduce * state.M +
                      vax_induced_proportional_risk_reduce * state.MV)

    rate = params.IS_to_R_rate * (1 - params.IS_to_H_adjusted_prop / immunity_force)

    return rate


def compute_IS_to_H_rate(state: FluFullMetapopStateTensors,
                         params: FluFullMetapopParamsTensors) -> torch.Tensor:
    inf_induced_hosp_risk_reduce = params.inf_induced_hosp_risk_reduce
    inf_induced_proportional_risk_reduce = inf_induced_hosp_risk_reduce / (1 - inf_induced_hosp_risk_reduce)

    vax_induced_hosp_risk_reduce = params.vax_induced_hosp_risk_reduce
    vax_induced_proportional_risk_reduce = vax_induced_hosp_risk_reduce / (1 - vax_induced_hosp_risk_reduce)

    immunity_force = (1 + inf_induced_proportional_risk_reduce * state.M +
                      vax_induced_proportional_risk_reduce * state.MV)

    rate = params.IS_to_H_rate * params.IS_to_H_adjusted_prop / immunity_force

    return rate


def compute_IA_to_R(state: FluFullMetapopStateTensors,
                    params: FluFullMetapopParamsTensors,
                    dt: float) -> torch.Tensor:
    rate = params.IA_to_R_rate

    IA_to_R = state.IA * torch_approx_binom_probability_from_rate(rate, dt)

    return IA_to_R


def compute_H_to_R_rate(state: FluFullMetapopStateTensors,
                        params: FluFullMetapopParamsTensors) -> torch.Tensor:
    inf_induced_death_risk_reduce = params.inf_induced_death_risk_reduce
    vax_induced_death_risk_reduce = params.vax_induced_death_risk_reduce

    inf_induced_proportional_risk_reduce = \
        inf_induced_death_risk_reduce / (1 - inf_induced_death_risk_reduce)

    vax_induced_proportional_risk_reduce = \
        vax_induced_death_risk_reduce / (1 - vax_induced_death_risk_reduce)

    immunity_force = (1 + inf_induced_proportional_risk_reduce * state.M +
                      vax_induced_proportional_risk_reduce * state.MV)

    rate = params.H_to_R_rate * (1 - params.H_to_D_adjusted_prop / immunity_force)

    return rate


def compute_H_to_D_rate(state: FluFullMetapopStateTensors,
                        params: FluFullMetapopParamsTensors) -> torch.Tensor:
    inf_induced_death_risk_reduce = params.inf_induced_death_risk_reduce
    vax_induced_death_risk_reduce = params.vax_induced_death_risk_reduce

    inf_induced_proportional_risk_reduce = \
        inf_induced_death_risk_reduce / (1 - inf_induced_death_risk_reduce)

    vax_induced_proportional_risk_reduce = \
        vax_induced_death_risk_reduce / (1 - vax_induced_death_risk_reduce)

    immunity_force = (1 + inf_induced_proportional_risk_reduce * state.M +
                      vax_induced_proportional_risk_reduce * state.MV)

    rate = params.H_to_D_rate * params.H_to_D_adjusted_prop / immunity_force

    return rate


def compute_R_to_S(state: FluFullMetapopStateTensors,
                   params: FluFullMetapopParamsTensors,
                   dt: float) -> torch.Tensor:
    rate = params.R_to_S_rate

    R_to_S = state.R * torch_approx_binom_probability_from_rate(rate, dt)

    return R_to_S


# The update rule for immunity is
#   - dM/dt = (R_to_S_rate * R / N) * (1 - inf_induced_saturation * M - vax_induced_saturation * M_v)
#                   - inf_induced_immune_wane * state.M
#   - dMV/dt = (new vaccinations at time t - delta)/ N - vax_induced_immune_wane


def compute_M_change(state: FluFullMetapopStateTensors, params: FluFullMetapopParamsTensors,
                     precomputed: FluPrecomputedTensors,
                     dt: float) -> torch.Tensor:
    # Note: already includes dt
    R_to_S = state.R * torch_approx_binom_probability_from_rate(params.R_to_S_rate, dt)

    M_change = (R_to_S / precomputed.total_pop_LAR) * \
               (1 - params.inf_induced_saturation * state.M - params.vax_induced_saturation * state.MV) - \
               params.inf_induced_immune_wane * state.M * dt

    # Because R_to_S includes dt already, we do not return M_change * dt -- we only multiply
    #   the last term in the expression above by dt
    return M_change


def compute_MV_change(state: FluFullMetapopStateTensors,
                      params: FluFullMetapopParamsTensors,
                      precomputed: FluPrecomputedTensors,
                      schedules: FluFullMetapopScheduleTensors,
                      day_counter: int,
                      dt: float) -> torch.Tensor:
    MV_change = state.daily_vaccines / precomputed.total_pop_LAR - \
                params.vax_induced_immune_wane * state.MV

    return MV_change * dt


def update_state_with_schedules(state: FluFullMetapopStateTensors,
                                params: FluFullMetapopParamsTensors,
                                precomputed: FluPrecomputedTensors,
                                schedules: FluFullMetapopScheduleTensors,
                                day_counter: int):

    flu_contact_matrix = compute_flu_contact_matrix(state, params, precomputed, schedules, day_counter)
    absolute_humidity = schedules.absolute_humidity[day_counter]
    daily_vaccines = schedules.daily_vaccines[day_counter]

    state_new = FluFullMetapopStateTensors(
        S=state.S,
        E=state.E,
        IP=state.IP,
        IS=state.IS,
        IA=state.IA,
        H=state.H,
        R=state.R,
        D=state.D,
        M=state.M,
        MV=state.MV,
        absolute_humidity=absolute_humidity,
        daily_vaccines=daily_vaccines,
        flu_contact_matrix=flu_contact_matrix
    )

    return state_new


def advance_timestep(state: FluFullMetapopStateTensors,
                     params: FluFullMetapopParamsTensors,
                     precomputed: FluPrecomputedTensors,
                     schedules: FluFullMetapopScheduleTensors,
                     day_counter: int,
                     dt: float):

    S_to_E = compute_S_to_E(state, params, precomputed, schedules, day_counter, dt)

    # Deterministic multinomial implementation to match
    #   object-oriented version
    E_to_IP_rate = compute_E_to_IP_rate(state, params)
    E_to_IA_rate = compute_E_to_IA_rate(state, params)
    E_outgoing_total_rate = E_to_IP_rate + E_to_IA_rate
    E_to_IA = state.E * (E_to_IA_rate / E_outgoing_total_rate) * \
        torch_approx_binom_probability_from_rate(E_outgoing_total_rate, dt)
    E_to_IP = state.E * (E_to_IP_rate / E_outgoing_total_rate) * \
        torch_approx_binom_probability_from_rate(E_outgoing_total_rate, dt)

    IA_to_R = compute_IA_to_R(state, params, dt)

    IP_to_IS = compute_IP_to_IS(state, params, dt)

    # Deterministic multinomial implementation to match
    #   object-oriented version
    IS_to_R_rate = compute_IS_to_R_rate(state, params)
    IS_to_H_rate = compute_IS_to_H_rate(state, params)
    IS_outgoing_total_rate = IS_to_R_rate + IS_to_H_rate
    IS_to_R = state.IS * (IS_to_R_rate / IS_outgoing_total_rate) * \
              torch_approx_binom_probability_from_rate(IS_outgoing_total_rate, dt)
    IS_to_H = state.IS * (IS_to_H_rate / IS_outgoing_total_rate) * \
              torch_approx_binom_probability_from_rate(IS_outgoing_total_rate, dt)

    # Deterministic multinomial implementation to match
    #   object-oriented version
    H_to_R_rate = compute_H_to_R_rate(state, params)
    H_to_D_rate = compute_H_to_D_rate(state, params)
    H_outgoing_total_rate = H_to_R_rate + H_to_D_rate
    H_to_R = state.H * (H_to_R_rate / H_outgoing_total_rate) * torch_approx_binom_probability_from_rate(
        H_outgoing_total_rate, dt)
    H_to_D = state.H * (H_to_D_rate / H_outgoing_total_rate) * torch_approx_binom_probability_from_rate(
        H_outgoing_total_rate, dt)

    R_to_S = compute_R_to_S(state, params, dt)

    M_change = compute_M_change(state, params, precomputed, dt)

    MV_change = compute_MV_change(state, params, precomputed, schedules, day_counter, dt)

    S_new = torch.nn.functional.softplus(state.S + R_to_S - S_to_E)

    E_new = torch.nn.functional.softplus(state.E + S_to_E - E_to_IP - E_to_IA)

    IP_new = torch.nn.functional.softplus(state.IP + E_to_IP - IP_to_IS)

    IS_new = torch.nn.functional.softplus(state.IS + IP_to_IS - IS_to_R - IS_to_H)

    if (IS_to_H < 0).any():
        breakpoint()

    IA_new = torch.nn.functional.softplus(state.IA + E_to_IA - IA_to_R)

    H_new = torch.nn.functional.softplus(state.H + IS_to_H - H_to_R - H_to_D)

    R_new = torch.nn.functional.softplus(state.R + IS_to_R + IA_to_R + H_to_R - R_to_S)

    D_new = torch.nn.functional.softplus(state.D + H_to_D)

    M_new = state.M + M_change

    MV_new = state.MV + MV_change

    state_new = FluFullMetapopStateTensors(S=S_new,
                                           E=E_new,
                                           IP=IP_new,
                                           IS=IS_new,
                                           IA=IA_new,
                                           H=H_new,
                                           R=R_new,
                                           D=D_new,
                                           M=M_new,
                                           MV=MV_new,
                                           absolute_humidity=state.absolute_humidity,
                                           daily_vaccines=state.daily_vaccines,
                                           flu_contact_matrix=state.flu_contact_matrix)

    calibration_targets = {}
    calibration_targets["IS_to_H"] = IS_to_H

    transition_variables = {}
    transition_variables["S_to_E"] = S_to_E
    transition_variables["E_to_IP"] = E_to_IP
    transition_variables["E_to_IA"] = E_to_IA
    transition_variables["IA_to_R"] = IA_to_R
    transition_variables["IP_to_IS"] = IP_to_IS
    transition_variables["IS_to_R"] = IS_to_R
    transition_variables["IS_to_H"] = IS_to_H
    transition_variables["H_to_R"] = H_to_R
    transition_variables["H_to_D"] = H_to_D
    transition_variables["R_to_S"] = R_to_S
    transition_variables["M_change"] = M_change
    transition_variables["MV_change"] = MV_change

    return state_new, calibration_targets, transition_variables


def torch_simulate_full_history(state: FluFullMetapopStateTensors,
                                params: FluFullMetapopParamsTensors,
                                precomputed: FluPrecomputedTensors,
                                schedules: FluFullMetapopScheduleTensors,
                                num_days: int,
                                timesteps_per_day: int) -> dict:
    dt = 1 / float(timesteps_per_day)

    state_history_dict = defaultdict(list)
    tvar_history_dict = defaultdict(list)

    for day in range(num_days):
        state = update_state_with_schedules(state, params, precomputed, schedules, day)
        for timestep in range(timesteps_per_day):
            state, calibration_targets, tvar_history = advance_timestep(state, params, precomputed, schedules, day, dt)
            for key in tvar_history:
                tvar_history_dict[key].append(tvar_history[key])
        for field in fields(state):
            if field.name == "init_vals":
                continue
            state_history_dict[str(field.name)].append(getattr(state, field.name).clone())

    return state_history_dict, tvar_history_dict


def torch_simulation_hospital_admits(state: FluFullMetapopStateTensors,
                                     params: FluFullMetapopParamsTensors,
                                     precomputed: FluPrecomputedTensors,
                                     schedules: FluFullMetapopScheduleTensors,
                                     num_days: int,
                                     timesteps_per_day: int) -> torch.Tensor:
    hospital_admits_history = []

    dt = 1 / float(timesteps_per_day)

    for day in range(num_days):
        state = update_state_with_schedules(state, params, precomputed, schedules, day)
        for timestep in range(timesteps_per_day):
            state, calibration_targets, _ = advance_timestep(state, params, precomputed, schedules, day, dt)
        hospital_admits_history.append(calibration_targets["IS_to_H"].clone())

    return torch.stack(hospital_admits_history)
