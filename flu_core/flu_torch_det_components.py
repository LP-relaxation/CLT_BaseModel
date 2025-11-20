###################################################################################
######################## MetroFluSim: pytorch implementation ######################
###################################################################################

# Dimensions
#   L (int):
#       number of locations/subpopulations
#   A (int):
#       number of age groups
#   R (int):
#       number of risk groups

import torch
import numpy as np
import pandas as pd
import clt_toolkit as clt

import datetime

from typing import Tuple

from collections import defaultdict
from dataclasses import dataclass, fields, field

from .flu_data_structures import FluFullMetapopStateTensors, \
    FluFullMetapopParamsTensors, FluPrecomputedTensors, \
    FluFullMetapopScheduleTensors
from .flu_travel_functions import compute_total_mixing_exposure

base_path = clt.utils.PROJECT_ROOT / "flu_instances" / "texas_input_files"


def torch_approx_binom_probability_from_rate(rate, dt):
    """
    Torch-compatible implementation of converting a
    rate into a probability. See analogous numpy implementation
    `base_components/approx_binom_probability_from_rate()` docstring
    for details.
    """

    return 1 - torch.exp(-rate * dt)


def create_dict_of_tensors(d: dict,
                           requires_grad: bool = True) -> dict:
    """
    Converts dictionary entries to `tensor` (of type `torch.float32`)
    and if `requires_grad` is `True`, turns on gradient tracking for
    each entry -- returns new dictionary.
    """

    def to_tensor(k, v):
        if v is None:
            return None
        else:
            return torch.tensor(v, dtype=torch.float32, requires_grad=requires_grad)

    return {k: to_tensor(k, v) for k, v in d.items()}


def compute_beta_adjusted(state: FluFullMetapopStateTensors,
                          params: FluFullMetapopParamsTensors) -> torch.Tensor:
    """
    Computes beta-adjusted humidity.

    Returns:
        (torch.Tensor of size (L, A, R))
    """

    absolute_humidity = state.absolute_humidity
    beta_adjusted = params.beta_baseline * (1 + params.humidity_impact * np.exp(-180 * absolute_humidity))

    return beta_adjusted


def compute_flu_contact_matrix(params: FluFullMetapopParamsTensors,
                               schedules: FluFullMetapopScheduleTensors,
                               day_counter: int) -> torch.Tensor:
    """
    Computes flu model contact matrix in tensor format -- makes
    adjustments based on whether day is school day or work day.

    Returns:
        (torch.Tensor of size (L, A, A))
    """

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
                   dt: float) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """

    # Needs flu_contact_matrix to be in state for this
    total_mixing_exposure = compute_total_mixing_exposure(state, params, precomputed)

    if total_mixing_exposure.size() != torch.Size([precomputed.L,
                                                   precomputed.A,
                                                   precomputed.R]):
        raise Exception("force_of_infection must be L x A x R corresponding \n"
                        "to number of locations (subpopulations), age groups, \n"
                        "and risk groups.")

    beta_adjusted = compute_beta_adjusted(state, params)

    inf_induced_inf_risk_reduce = params.inf_induced_inf_risk_reduce
    inf_induced_proportional_risk_reduce = inf_induced_inf_risk_reduce / (1 - inf_induced_inf_risk_reduce)

    vax_induced_inf_risk_reduce = params.vax_induced_inf_risk_reduce
    vax_induced_proportional_risk_reduce = vax_induced_inf_risk_reduce / (1 - vax_induced_inf_risk_reduce)

    immune_force = (1 + inf_induced_proportional_risk_reduce * state.M +
                    vax_induced_proportional_risk_reduce * state.MV)

    rate = beta_adjusted * total_mixing_exposure / immune_force

    S_to_E = state.S * torch_approx_binom_probability_from_rate(rate, dt)

    return S_to_E


def compute_E_to_IP_rate(params: FluFullMetapopParamsTensors) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """

    return params.E_to_I_rate * (1 - params.E_to_IA_prop)


def compute_E_to_IA_rate(params: FluFullMetapopParamsTensors) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """

    return params.E_to_I_rate * params.E_to_IA_prop


def compute_IP_to_ISR_rate(state: FluFullMetapopStateTensors,
                           params: FluFullMetapopParamsTensors) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """
    
    inf_induced_hosp_risk_reduce = params.inf_induced_hosp_risk_reduce
    inf_induced_proportional_risk_reduce = inf_induced_hosp_risk_reduce / (1 - inf_induced_hosp_risk_reduce)

    vax_induced_hosp_risk_reduce = params.vax_induced_hosp_risk_reduce
    vax_induced_proportional_risk_reduce = vax_induced_hosp_risk_reduce / (1 - vax_induced_hosp_risk_reduce)

    immunity_force = (1 + inf_induced_proportional_risk_reduce * state.M +
                      vax_induced_proportional_risk_reduce * state.MV)

    rate = params.IP_to_IS_rate * (1 - params.IP_to_ISH_prop / immunity_force)

    return rate


def compute_IP_to_ISH_rate(state: FluFullMetapopStateTensors,
                           params: FluFullMetapopParamsTensors) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """

    inf_induced_hosp_risk_reduce = params.inf_induced_hosp_risk_reduce
    inf_induced_proportional_risk_reduce = inf_induced_hosp_risk_reduce / (1 - inf_induced_hosp_risk_reduce)

    vax_induced_hosp_risk_reduce = params.vax_induced_hosp_risk_reduce
    vax_induced_proportional_risk_reduce = vax_induced_hosp_risk_reduce / (1 - vax_induced_hosp_risk_reduce)

    immunity_force = (1 + inf_induced_proportional_risk_reduce * state.M +
                      vax_induced_proportional_risk_reduce * state.MV)

    rate = params.IP_to_IS_rate * (params.IP_to_ISH_prop / immunity_force)

    return rate


def compute_IA_to_R(state: FluFullMetapopStateTensors,
                    params: FluFullMetapopParamsTensors,
                    dt: float) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """

    rate = params.IA_to_R_rate

    IA_to_R = state.IA * torch_approx_binom_probability_from_rate(rate, dt)

    return IA_to_R


def compute_ISR_to_R(state: FluFullMetapopStateTensors,
                     params: FluFullMetapopParamsTensors,
                     dt: float) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """

    rate = params.ISR_to_R_rate

    ISR_to_R = state.ISR * torch_approx_binom_probability_from_rate(rate, dt)

    return ISR_to_R


def compute_ISH_to_HR_rate(state: FluFullMetapopStateTensors,
                           params: FluFullMetapopParamsTensors) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """

    inf_induced_death_risk_reduce = params.inf_induced_death_risk_reduce
    vax_induced_death_risk_reduce = params.vax_induced_death_risk_reduce

    inf_induced_proportional_risk_reduce = \
        inf_induced_death_risk_reduce / (1 - inf_induced_death_risk_reduce)

    vax_induced_proportional_risk_reduce = \
        vax_induced_death_risk_reduce / (1 - vax_induced_death_risk_reduce)

    immunity_force = (1 + inf_induced_proportional_risk_reduce * state.M +
                      vax_induced_proportional_risk_reduce * state.MV)

    rate = (1 - params.ISH_to_HD_prop / immunity_force) * params.ISH_to_H_rate

    return rate


def compute_ISH_to_HD_rate(state: FluFullMetapopStateTensors,
                           params: FluFullMetapopParamsTensors) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """

    inf_induced_death_risk_reduce = params.inf_induced_death_risk_reduce
    vax_induced_death_risk_reduce = params.vax_induced_death_risk_reduce

    inf_induced_proportional_risk_reduce = \
        inf_induced_death_risk_reduce / (1 - inf_induced_death_risk_reduce)

    vax_induced_proportional_risk_reduce = \
        vax_induced_death_risk_reduce / (1 - vax_induced_death_risk_reduce)

    immunity_force = (1 + inf_induced_proportional_risk_reduce * state.M +
                      vax_induced_proportional_risk_reduce * state.MV)

    rate = params.ISH_to_HD_prop / immunity_force * params.ISH_to_H_rate

    return rate


def compute_HR_to_R(state: FluFullMetapopStateTensors,
                    params: FluFullMetapopParamsTensors,
                    dt: float) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """
    
    rate = params.HR_to_R_rate

    HR_to_R = state.HR * torch_approx_binom_probability_from_rate(rate, dt)

    return HR_to_R


def compute_HD_to_D(state: FluFullMetapopStateTensors,
                    params: FluFullMetapopParamsTensors,
                    dt: float) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """
    
    rate = params.HD_to_D_rate

    HD_to_D = state.HD * torch_approx_binom_probability_from_rate(rate, dt)

    return HD_to_D


def compute_R_to_S(state: FluFullMetapopStateTensors,
                   params: FluFullMetapopParamsTensors,
                   dt: float) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """

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
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """

    # Note: already includes dt
    R_to_S = state.R * torch_approx_binom_probability_from_rate(params.R_to_S_rate, dt)

    M_change = (R_to_S / precomputed.total_pop_LAR_tensor) * \
               (1 - params.inf_induced_saturation * state.M - params.vax_induced_saturation * state.MV) - \
               params.inf_induced_immune_wane * state.M * dt

    # Because R_to_S includes dt already, we do not return M_change * dt -- we only multiply
    #   the last term in the expression above by dt
    return M_change


def compute_MV_change(state: FluFullMetapopStateTensors,
                      params: FluFullMetapopParamsTensors,
                      precomputed: FluPrecomputedTensors,
                      dt: float) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """

    MV_change = state.daily_vaccines - \
                params.vax_induced_immune_wane * state.MV

    return MV_change * dt


def update_state_with_schedules(state: FluFullMetapopStateTensors,
                                params: FluFullMetapopParamsTensors,
                                schedules: FluFullMetapopScheduleTensors,
                                day_counter: int) -> FluFullMetapopStateTensors:
    """
    Returns new dataclass formed by copying the current `state`
    and updating specific values according to `schedules` and
    the simulation's current `day_counter`.

    Returns:
        (FluFullMetapopStateTensors):
            New state with updated schedule-related values:
              - `flu_contact_matrix`
              - `absolute_humidity`
              - `daily_vaccines`
            All other fields remain unchanged from the input `state`.
    """

    flu_contact_matrix = compute_flu_contact_matrix(params, schedules, day_counter)
    absolute_humidity = schedules.absolute_humidity[day_counter]
    daily_vaccines = schedules.daily_vaccines[day_counter]

    state_new = FluFullMetapopStateTensors(
        S=state.S,
        E=state.E,
        IP=state.IP,
        ISR=state.ISR,
        ISH=state.ISH,
        IA=state.IA,
        HR=state.HR,
        HD=state.HD,
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
                     dt: float,
                     save_calibration_targets: bool=False,
                     save_tvar_history: bool=False) -> Tuple[FluFullMetapopStateTensors, dict, dict]:
    """
    Advance the simulation one timestep, with length `dt`.
    Updates state corresponding to compartments and
    epidemiological metrics after computing transition variables
    and metric changes.

    Note that in this torch "mean" deterministic implementation...
    - We compute rates in the same way as the
        `get_binom_deterministic_no_round`
        transition type in the OOP code -- see
        `TransitionVariables` class in
        `clt_toolkit / base_components` for more details.
    - We also implement a "mean" deterministic analog
        of the multinomial distribution to handle
        multiple outflows from the same compartment
    - We do not round the transition variables
    - We also use `softplus`, a smooth approximation to the
        ReLU function, to ensure that compartments are
        nonnegative (which is not guaranteed using
        the mean of a binomial/multinomial random variable
        rather than sampling from those distributions).

    Returns:
        (Tuple[FluFullMetapopStateTensors, dict, dict]):
            New `FluFullMetapopStateTensors` with updated state,
            `dict` of calibration targets corresponding to state
            values or transition variable values used for calibration,
            and `dict` of transition variable values to save this
            history. If `save_calibration_targets` is `False`,
            then the corresponding `dict` is empty, and similarly with
            `save_tvar_history`.
    """ 
    
    S_to_E = compute_S_to_E(state, params, precomputed, dt)

    # Deterministic multinomial implementation to match
    #   object-oriented version
    E_to_IP_rate = compute_E_to_IP_rate(params)
    E_to_IA_rate = compute_E_to_IA_rate(params)
    E_outgoing_total_rate = E_to_IP_rate + E_to_IA_rate
    E_outgoing_total = state.E * \
        torch_approx_binom_probability_from_rate(E_outgoing_total_rate, dt)
    E_to_IA = E_outgoing_total * (E_to_IA_rate / E_outgoing_total_rate)              
    E_to_IP = E_outgoing_total * (E_to_IP_rate / E_outgoing_total_rate)

    IA_to_R = compute_IA_to_R(state, params, dt)
    
    # Deterministic multinomial implementation to match
    #   object-oriented version
    IP_to_ISR_rate = compute_IP_to_ISR_rate(state, params)
    IP_to_ISH_rate = compute_IP_to_ISH_rate(state, params)
    IP_outgoing_total_rate = IP_to_ISR_rate + IP_to_ISH_rate
    IP_outgoing_total = state.IP * \
        torch_approx_binom_probability_from_rate(IP_outgoing_total_rate, dt)
    IP_to_ISR = IP_outgoing_total * (IP_to_ISR_rate / IP_outgoing_total_rate)
    IP_to_ISH = IP_outgoing_total * (IP_to_ISH_rate / IP_outgoing_total_rate)
                
    ISR_to_R = compute_ISR_to_R(state, params, dt)
    
    # Deterministic multinomial implementation to match
    #   object-oriented version
    ISH_to_HR_rate = compute_ISH_to_HR_rate(state, params)
    ISH_to_HD_rate = compute_ISH_to_HD_rate(state, params)
    ISH_outgoing_total_rate = ISH_to_HR_rate + ISH_to_HD_rate
    ISH_outgoing_total = state.ISH * \
        torch_approx_binom_probability_from_rate(ISH_outgoing_total_rate, dt)
    ISH_to_HR = ISH_outgoing_total * (ISH_to_HR_rate / ISH_outgoing_total_rate)
    ISH_to_HD = ISH_outgoing_total * (ISH_to_HD_rate / ISH_outgoing_total_rate)

    # Deterministic multinomial implementation to match
    #   object-oriented version
    HR_to_R = compute_HR_to_R(state, params, dt)
    HD_to_D = compute_HD_to_D(state, params, dt)

    R_to_S = compute_R_to_S(state, params, dt)
    
    
    # Make sure compartments are nonnegative
    S_new = torch.nn.functional.softplus(state.S + R_to_S - S_to_E)
    E_new = torch.nn.functional.softplus(state.E + S_to_E - E_to_IP - E_to_IA)
    IP_new = torch.nn.functional.softplus(state.IP + E_to_IP - IP_to_ISR - IP_to_ISH)
    ISR_new = torch.nn.functional.softplus(state.ISR + IP_to_ISR - ISR_to_R)
    ISH_new = torch.nn.functional.softplus(state.ISH + IP_to_ISH - ISH_to_HR - ISH_to_HD)
    IA_new = torch.nn.functional.softplus(state.IA + E_to_IA - IA_to_R)
    HR_new = torch.nn.functional.softplus(state.HR + ISH_to_HR - HR_to_R)
    HD_new = torch.nn.functional.softplus(state.HD + ISH_to_HD - HD_to_D)
    R_new = torch.nn.functional.softplus(state.R + ISR_to_R + IA_to_R + HR_to_R - R_to_S)
    D_new = torch.nn.functional.softplus(state.D + HD_to_D)

    # Update immunity variables
    M_change = compute_M_change(state, params, precomputed, dt)
    MV_change = compute_MV_change(state, params, precomputed, dt)
    M_new = state.M + M_change
    MV_new = state.MV + MV_change

    state_new = FluFullMetapopStateTensors(S=S_new,
                                           E=E_new,
                                           IP=IP_new,
                                           ISR=ISR_new,
                                           ISH=ISH_new,
                                           IA=IA_new,
                                           HR=HR_new,
                                           HD=HD_new,
                                           R=R_new,
                                           D=D_new,
                                           M=M_new,
                                           MV=MV_new,
                                           absolute_humidity=state.absolute_humidity,
                                           daily_vaccines=state.daily_vaccines,
                                           flu_contact_matrix=state.flu_contact_matrix)

    calibration_targets = {}
    if save_calibration_targets:
        calibration_targets["ISH_to_H"] = ISH_to_HR + ISH_to_HD

    transition_variables = {}
    if save_tvar_history:
        transition_variables["S_to_E"] = S_to_E
        transition_variables["E_to_IP"] = E_to_IP
        transition_variables["E_to_IA"] = E_to_IA
        transition_variables["IA_to_R"] = IA_to_R
        transition_variables["IP_to_ISR"] = IP_to_ISR
        transition_variables["IP_to_ISH"] = IP_to_ISH
        transition_variables["ISR_to_R"] = ISR_to_R
        transition_variables["ISH_to_HR"] = ISH_to_HR
        transition_variables["ISH_to_HD"] = ISH_to_HD
        transition_variables["HR_to_R"] = HR_to_R
        transition_variables["HD_to_D"] = HD_to_D
        transition_variables["R_to_S"] = R_to_S
        transition_variables["M_change"] = M_change
        transition_variables["MV_change"] = MV_change

    return state_new, calibration_targets, transition_variables


def torch_simulate_full_history(state: FluFullMetapopStateTensors,
                                params: FluFullMetapopParamsTensors,
                                precomputed: FluPrecomputedTensors,
                                schedules: FluFullMetapopScheduleTensors,
                                num_days: int,
                                timesteps_per_day: int) -> Tuple[dict, dict]:
    """
    Simulates the flu model with a differentiable torch implementation
    that carries out `binom_deterministic_no_round` transition types --
    returns hospital admits for calibration use.

    See subroutine `advance_timestep` for additional details.

    Returns:
        (Tuple[dict, dict]):
            Returns compartment states and transition variables
            for day, location, age, risk, in tensor format.
    """

    dt = 1 / float(timesteps_per_day)

    state_history_dict = defaultdict(list)
    tvar_history_dict = defaultdict(list)

    # This could probably be written better so we don't have
    #   unused variables "_" that grab `advance_timestep` output?

    for day in range(num_days):
        state = update_state_with_schedules(state, params, schedules, day)

        for timestep in range(timesteps_per_day):
            # TODO double check whether this split makes sense
            #   to get the total transition variables we should need to save values
            #   at each timestep when there are several steps per day
            #   (these variables may not be used anywhere right now)
            if timestep == timesteps_per_day-1:
                state, _, tvar_history = \
                    advance_timestep(state, params, precomputed, dt, save_tvar_history=True)
                for key in tvar_history:
                    tvar_history_dict[key].append(tvar_history[key])
            else:
                state, _, _ = \
                    advance_timestep(state, params, precomputed, dt, save_tvar_history=False)

        for field in fields(state):
            if field.name == "init_vals":
                continue
            state_history_dict[str(field.name)].append(getattr(state, field.name).clone())

    return state_history_dict, tvar_history_dict


def torch_simulate_hospital_admits(state: FluFullMetapopStateTensors,
                                     params: FluFullMetapopParamsTensors,
                                     precomputed: FluPrecomputedTensors,
                                     schedules: FluFullMetapopScheduleTensors,
                                     num_days: int,
                                     timesteps_per_day: int) -> torch.Tensor:
    """
    Analogous to `torch_simulate_full_history` but only saves and
    returns hospital admits for calibration use.

    Returns:
        (torch.Tensor of size (num_days, L, A, R)):
            Returns hospital admits (the ISH to HR and HD 
            transition variable values) for day, location,
            age, risk, in tensor format.
    """

    hospital_admits_history = []

    dt = 1 / float(timesteps_per_day)

    for day in range(num_days):
        state = update_state_with_schedules(state, params, schedules, day)
        for timestep in range(timesteps_per_day):
            state, calibration_targets, _ = \
                advance_timestep(state, params, precomputed, dt, save_calibration_targets=True)
        hospital_admits_history.append(calibration_targets["ISH_to_H"].clone())

    return torch.stack(hospital_admits_history)
