import datetime
import copy

import numpy as np
import pandas as pd
import sciris as sc

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import clt_base as clt

base_path = Path(__file__).parent.parent / "flu_demo_input_files"


# Note: for dataclasses, Optional is used to help with static type checking
# -- it means that an attribute can either hold a value with the specified
# datatype or it can be None


@dataclass
class FluSubpopParams(clt.SubpopParams):
    """
    Data container for pre-specified and fixed epidemiological
    parameters in FluModel flu model.

    Each field of datatype np.ndarray must be |A| x |R|,
    where |A| is the number of age groups and |R| is the number of
    risk groups. Note: this means all arrays should be 2D.
    See FluSubpopState docstring for important formatting note
    on 2D arrays.

    TODO:
        when adding multiple strains, need to add subscripts
        to math of attributes and add strain-specific description

    Attributes:
        num_age_groups (positive int):
            number of age groups.
        num_risk_groups (positive int):
            number of risk groups.
        beta_baseline (positive float): transmission rate.
        total_pop_age_risk (np.ndarray of positive ints):
            total number in population, summed across all
            age-risk groups.
        humidity_impact (positive float):
            coefficient that determines how much absolute
            humidity affects beta_baseline.
        hosp_immune_gain (positive float):
            factor by which population-level immunity
            against hospitalization grows after each
            case that recovers.
        inf_immune_gain (positive float):
            factor by which population-level immunity
            against infection grows after each case
                that recovers.
        immune_saturation (np.ndarray of positive floats):
            constant(s) modeling saturation of antibody
            production of individuals.
        hosp_immune_wane (positive float):
            rate at which infection-induced immunity
            against hospitalization wanes.
        inf_immune_wane (positive float):
            rate at which infection-induced immunity
            against infection wanes.
        hosp_risk_reduce (positive float in [0,1]):
            reduction in hospitalization risk from
            infection-induced immunity.
        inf_risk_reduce (positive float in [0,1]):
            reduction in infection risk
            from infection-induced immunity.
        death_risk_reduce (positive float in [0,1]):
            reduction in death risk from infection-induced immunity.
        R_to_S_rate (positive float):
            rate at which people in R move to S.
        E_to_I_rate (positive float):
            rate at which people in E move to I (both
            IP and IA, infected pre-symptomatic and infected
            asymptomatic)
        IP_to_IS_rate (positive float):
            rate a which people in IP (infected pre-symptomatic)
            move to IS (infected symptomatic)
        IS_to_R_rate (positive float):
            rate at which people in IS (infected symptomatic)
            move to R.
        IA_to_R_rate (positive float):
            rate at which people in IA (infected asymptomatic)
            move to R
        IS_to_H_rate (positive float):
            rate at which people in IS (infected symptomatic)
            move to H.
        H_to_R_rate (positive float):
            rate at which people in H move to R.
        H_to_D_rate (positive float):
            rate at which people in H move to D.
        E_to_IA_prop (np.ndarray of positive floats in [0,1]):
            proportion exposed who are asymptomatic based on
            age-risk groups.
        IS_to_H_adjusted_prop (np.ndarray of positive floats in [0,1]):
            rate-adjusted proportion infected who are hospitalized
            based on age-risk groups.
        H_to_D_adjusted_prop (np.ndarray of positive floats in [0,1]):
            rate-adjusted proportion hospitalized who die based on
            age-risk groups.
        IP_relative_inf (positive float):
            relative infectiousness of pre-symptomatic to symptomatic
            people (IP to IS compartment).
        IA_relative_inf (positive float):
            relative infectiousness of asymptomatic to symptomatic
            people (IA to IS compartment).
        viral_shed_peak (positive float):
            the peak time of an individual's viral shedding.
        viral_shed_magnitude (positive float):
            magnitude of the viral shedding.
        viral_shed_duration (positive float):
            duration of the viral shedding,
            must be larger than viral_shed_peak
        viral_shed_feces_mass (positive float):
            average mass of feces (gram)
        relative_suscept_by_age (np.ndarray of positive floats in [0,1]):
            relative susceptibility to infection by age group
        prop_time_away_by_age (np.ndarray of positive floats in [0,1]):
            total proportion of time spent away from home by age group
        contact_mult_travel (positive float in [0,1]):
            multiplier to reduce contact rate of traveling individuals
        contact_mult_symp (positive float in [0,1]):
            multiplier to reduce contact rate of symptomatic individuals
        total_contact_matrix (np.ndarray of positive floats):
            |A| x |A| contact matrix (where |A| is the number
            of age groups), where element i,j is the average
            contacts from age group j that an individual in
            age group i has
        school_contact_matrix (np.ndarray of positive floats):
            |A| x |A| contact matrix (where |A| is the number
            of age groups), where element i,j is the average
            contacts from age group j that an individual in
            age group i has at school -- this matrix plus the
            work_contact_matrix must be less than the
            total_contact_matrix, element-wise
        work_contact_matrix (np.ndarray of positive floats):
            |A| x |A| contact matrix (where |A| is the number
            of age groups), where element i,j is the average
            contacts from age group j that an individual in
            age group i has at work -- this matrix plus the
            work_contact_matrix must be less than the
            total_contact_matrix, element-wise
    """

    num_age_groups: Optional[int] = None
    num_risk_groups: Optional[int] = None
    beta_baseline: Optional[float] = None
    total_pop_age_risk: Optional[np.ndarray] = None
    humidity_impact: Optional[float] = None
    hosp_immune_gain: Optional[float] = None
    inf_immune_gain: Optional[float] = None
    immune_saturation: Optional[np.ndarray] = None
    hosp_immune_wane: Optional[float] = None
    inf_immune_wane: Optional[float] = None
    hosp_risk_reduce: Optional[float] = None
    inf_risk_reduce: Optional[float] = None
    death_risk_reduce: Optional[float] = None
    R_to_S_rate: Optional[float] = None
    E_to_I_rate: Optional[float] = None
    IP_to_IS_rate: Optional[float] = None
    IS_to_R_rate: Optional[float] = None
    IA_to_R_rate: Optional[float] = None
    IS_to_H_rate: Optional[float] = None
    H_to_R_rate: Optional[float] = None
    H_to_D_rate: Optional[float] = None
    E_to_IA_prop: Optional[np.ndarray] = None
    IS_to_H_adjusted_prop: Optional[np.ndarray] = None
    H_to_D_adjusted_prop: Optional[np.ndarray] = None
    IP_relative_inf: Optional[float] = None
    IA_relative_inf: Optional[float] = None
    viral_shed_peak: Optional[float] = None  # viral shedding parameters
    viral_shed_magnitude: Optional[float] = None  # viral shedding parameters
    viral_shed_duration: Optional[float] = None  # viral shedding parameters
    viral_shed_feces_mass: Optional[float] = None  # viral shedding parameters
    relative_suscept_by_age: Optional[np.ndarray] = None
    prop_time_away_by_age: Optional[np.ndarray] = None
    contact_mult_travel: Optional[float] = None
    contact_mult_symp: Optional[float] = None
    total_contact_matrix: Optional[np.ndarray] = None
    school_contact_matrix: Optional[np.ndarray] = None
    work_contact_matrix: Optional[np.ndarray] = None


@dataclass
class FluSubpopState(clt.SubpopState):
    """
    Data container for pre-specified and fixed set of
    Compartment initial values and EpiMetric initial values
    in FluModel flu model.

    Each field below should be |A| x |R| np.ndarray, where
    |A| is the number of age groups and |R| is the number of risk groups.
    Note: this means all arrays should be 2D. Even if there is
    1 age group and 1 risk group (no group stratification),
    each array should be 1x1, which is two-dimensional.
    For example, np.array([[100]]) is correct --
    np.array([100]) is wrong.

    Attributes:
        S (np.ndarray of positive floats):
            susceptible compartment for age-risk groups --
            (holds current_val of Compartment "S").
        E (np.ndarray of positive floats):
            exposed compartment for age-risk groups --
            (holds current_val of Compartment "E").
        IP (np.ndarray of positive floats):
            infected pre-symptomatic compartment for age-risk groups
            (holds current_val of Compartment "IP").
        IS (np.ndarray of positive floats):
            infected symptomatic compartment for age-risk groups
            (holds current_val of Compartment "IS").
        IA (np.ndarray of positive floats):
            infected asymptomatic compartment for age-risk groups
            (holds current_val of Compartment "IA").
        H (np.ndarray of positive floats):
            hospital compartment for age-risk groups
            (holds current_val of Compartment "H").
        R (np.ndarray of positive floats):
            recovered compartment for age-risk groups
            (holds current_val of Compartment "R").
        D (np.ndarray of positive floats):
            dead compartment for age-risk groups
            (holds current_val of Compartment "D").
        pop_immunity_hosp (np.ndarray of positive floats):
            infection-induced population-level immunity against
            hospitalization, for age-risk groups (holds current_val
            of EpiMetric "pop_immunity_hosp").
        pop_immunity_inf (np.ndarray of positive floats):
            infection-induced population-level immunity against
            infection, for age-risk groups (holds current_val
            of EpiMetric "pop_immunity_inf").
        absolute_humidity (positive float):
            grams of water vapor per cubic meter g/m^3,
            used as seasonality parameter that influences
            transmission rate beta_baseline.
        flu_contact_matrix (np.ndarray of positive floats):
            |A| x |A| array, where |A| is the number of age
            groups -- element (a, a') corresponds to the number 
            of contacts that a person in age group a
            has with people in age-risk group a'.
        beta_reduct (float in [0,1]):
            starting value of DynamicVal "beta_reduct" on
            starting day of simulation -- this DynamicVal
            emulates a simple staged-alert policy
        wastewater (np.ndarray of positive floats):
            wastewater viral load
        force_of_infection (np.ndarray of positive floats):
            total force of infection from movement within
            home location, travel to other locations,
            and visitors from other locations
        hospital_admits (np.ndarray of int):
            tracks IS to H
    """

    S: Optional[np.ndarray] = None
    E: Optional[np.ndarray] = None
    IP: Optional[np.ndarray] = None
    IS: Optional[np.ndarray] = None
    IA: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    D: Optional[np.ndarray] = None
    pop_immunity_hosp: Optional[np.ndarray] = None
    pop_immunity_inf: Optional[np.ndarray] = None
    absolute_humidity: Optional[float] = None
    flu_contact_matrix: Optional[np.ndarray] = None
    beta_reduct: Optional[float] = 0.0
    wastewater: Optional[np.ndarray] = None  # wastewater viral load
    force_of_infection: Optional[np.ndarray] = None
    hospital_admits: Optional[np.ndarray] = None


class SusceptibleToExposed(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "S" to "E" compartment. The functional form is the same across
    subpopulations.

    The rate depends on the corresponding subpopulation's
    contact matrix, transmission rate beta, number
    infected (symptomatic, asymptomatic, and pre-symptomatic),
    and population-level immunity against infection,
    among other parameters.

    This is the most complicated transition variable in the
    flu model. If using metapopulation model (travel model), then
    the rate depends on an ForceOfInfection instance that is
    a function of other subpopulations' states and parameters,
    and travel between subpopulations.

    If there is no metapopulation model, then there is no
    ForceOfInfection InteractionTerm instance, and the rate
    is much simpler.
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:

        # If there is no ForceOfInfection InteractionTerm instance,
        #   then there is no travel model -- so, simulate
        #   this subpopulation entirely independently and
        #   use the simplified transition rate that does not
        #   depend on travel dynamics

        if state.force_of_infection is not None:
            return state.force_of_infection
        else:
            wtd_presymp_asymp = compute_wtd_presymp_asymp(state, params)

            return compute_common_coeff_force_of_infection(state, params) * \
                   np.matmul(state.flu_contact_matrix,
                             np.divide(np.reshape(np.sum(state.IS, axis=1), (2,1)) + wtd_presymp_asymp,
                                       compute_pop_by_age(params)))


class RecoveredToSusceptible(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "R" to "S" compartment. The functional form is the same across
    subpopulations.
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.R_to_S_rate)


class ExposedToAsymp(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "E" to "IA" compartment. The functional form is the same across
    subpopulations.

    Each ExposedToAsymp instance forms a TransitionVariableGroup with
    a corresponding ExposedToPresymp instance (these two
    transition variables are jointly distributed).
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.E_to_I_rate * params.E_to_IA_prop)


class ExposedToPresymp(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "E" to "IP" compartment. The functional form is the same across
    subpopulations.

    Each ExposedToPresymp instance forms a TransitionVariableGroup with
    a corresponding ExposedToAsymp instance (these two
    transition variables are jointly distributed).
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.E_to_I_rate * (1 - params.E_to_IA_prop))


class PresympToSymp(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "IP" to "IS" compartment. The functional form is the same across
    subpopulations.
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.IP_to_IS_rate)


class SympToRecovered(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "IS" to "R" compartment. The functional form is the same across
    subpopulations.

    Each SympToRecovered instance forms a TransitionVariableGroup with
    a corresponding SympToHosp instance (these two
    transition variables are jointly distributed).
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        return np.full((params.num_age_groups, params.num_risk_groups),
                       (1 - params.IS_to_H_adjusted_prop) * params.IS_to_R_rate)


class AsympToRecovered(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "IA" to "R" compartment. The functional form is the same across
    subpopulations.
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.IA_to_R_rate)


class HospToRecovered(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "H" to "R" compartment. The functional form is the same across
    subpopulations.

    Each HospToRecovered instance forms a TransitionVariableGroup with
    a corresponding HospToDead instance (these two
    transition variables are jointly distributed).
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        return np.full((params.num_age_groups, params.num_risk_groups),
                       (1 - params.H_to_D_adjusted_prop) * params.H_to_R_rate)


class SympToHosp(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "IS" to "H" compartment. The functional form is the same across
    subpopulations.

    Each SympToHosp instance forms a TransitionVariableGroup with
    a corresponding SympToRecovered instance (these two
    transition variables are jointly distributed).

    The rate of SympToHosp decreases as population-level immunity
    against hospitalization increases.
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:

        hosp_risk_reduce = params.hosp_risk_reduce

        if (np.asarray(hosp_risk_reduce) != 1).all():
            proportional_risk_reduction = hosp_risk_reduce / (1 - hosp_risk_reduce)
        else:
            proportional_risk_reduction = 1

        return np.asarray(params.IS_to_H_rate * params.IS_to_H_adjusted_prop /
                          (1 + proportional_risk_reduction * state.pop_immunity_hosp))


class HospToDead(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "H" to "D" compartment. The functional form is the same across
    subpopulations.

    Each HospToDead instance forms a TransitionVariableGroup with
    a corresponding HospToRecovered instance (these two
    transition variables are jointly distributed).

    The rate of HospToDead decreases as population-level immunity
    against hospitalization increases.
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:

        death_risk_reduce = params.death_risk_reduce

        if (np.asarray(death_risk_reduce) != 1).all():
            proportional_risk_reduction = death_risk_reduce / (1 - death_risk_reduce)
        else:
            proportional_risk_reduction = 1

        return np.asarray(params.H_to_D_adjusted_prop * params.H_to_D_rate /
                          (1 + proportional_risk_reduction * state.pop_immunity_hosp))


class PopulationImmunityHosp(clt.EpiMetric):
    """
    EpiMetric-derived class for population-level immunity
    from hospitalization.
    
    Population-level immunity increases as people move
    from "R" to "S" -- this is a design choice intended
    to avoid "double-counting." People in "R" cannot be
    infected at all. People who move from "R" to "S" 
    are susceptible again, but these recently-recovered people 
    should have partial immunity. To handle this phenomenon,
    the population-level immunity epi metric increases as
    people move from "R" to "S." 

    Params:
        R_to_S (RecoveredToSusceptible):
            RecoveredToSusceptible TransitionVariable
            in the SubpopModel -- it is an attribute
            because the population-level immunity
            increases as people move from "R" to "S".

    See parent class docstring for other attributes.
    """

    def __init__(self, init_val, R_to_S):
        super().__init__(init_val)
        self.R_to_S = R_to_S

    def get_change_in_current_val(self,
                                  state: FluSubpopState,
                                  params: FluSubpopParams,
                                  num_timesteps: int) -> np.ndarray:

        pop_immunity_hosp = state.pop_immunity_hosp

        immunity_gain_numerator = params.hosp_immune_gain * self.R_to_S.current_val
        immunity_gain_denominator = params.total_pop_age_risk * \
                                    (1 + params.immune_saturation * pop_immunity_hosp)

        immunity_gain = immunity_gain_numerator / immunity_gain_denominator
        immunity_loss = params.hosp_immune_wane * pop_immunity_hosp

        final_change = immunity_gain - immunity_loss / num_timesteps

        return np.asarray(final_change, dtype=np.float64)


class PopulationImmunityInf(clt.EpiMetric):
    """
    EpiMetric-derived class for population-level immunity
    from infection.

    Analogous to PopulationImmunityHosp -- see that class's
    docstring for more information. Update formula is the 
    same except for inf_immune_gain and
    inf_immune_wane. 

    Params:
        R_to_S (RecoveredToSusceptible):
            RecoveredToSusceptible TransitionVariable
            in the SubpopModel -- it is an attribute
            because the population-level immunity
            increases as people move from "R" to "S".

    See parent class docstring for other attributes.
    """

    def __init__(self, init_val, R_to_S):
        super().__init__(init_val)
        self.R_to_S = R_to_S

    def get_change_in_current_val(self,
                                  state: FluSubpopState,
                                  params: FluSubpopParams,
                                  num_timesteps: int):

        pop_immunity_inf = np.float64(state.pop_immunity_inf)

        immunity_gain_numerator = params.inf_immune_gain * self.R_to_S.current_val
        immunity_gain_denominator = params.total_pop_age_risk * \
                                    (1 + params.immune_saturation * pop_immunity_inf)

        immunity_gain = immunity_gain_numerator / immunity_gain_denominator
        immunity_loss = params.inf_immune_wane * pop_immunity_inf

        final_change = immunity_gain - immunity_loss / num_timesteps

        return np.asarray(final_change, dtype=np.float64)


# test on the wastewater viral load simulation
class Wastewater(clt.EpiMetric):

    def __init__(self, init_val, S_to_E):
        super().__init__(init_val)
        self.S_to_E = S_to_E
        # preprocess
        self.flag_preprocessed = False
        self.viral_shedding = []
        self.viral_shed_duration = None
        self.viral_shed_magnitude = None
        self.viral_shed_peak = None
        self.viral_shed_feces_mass = None
        self.S_to_E_len = 5000  # preset to match the simulation time horizon
        self.S_to_E_history = np.zeros(self.S_to_E_len)
        self.cur_time_stamp = -1
        self.num_timesteps = None
        self.val_list_len = None
        self.current_val_list = None
        self.cur_idx_timestep = -1

    def get_change_in_current_val(self,
                                  state: FluSubpopState,
                                  params: FluSubpopParams,
                                  num_timesteps: int):
        if not self.flag_preprocessed:  # preprocess the viral shedding function if not done yet
            self.val_list_len = num_timesteps
            self.current_val_list = np.zeros(self.val_list_len)
            self.preprocess(params, num_timesteps)
        return 0

    def update_current_val(self) -> None:
        """
        Adds change_in_current_val attribute to current_val attribute
            in-place.
        """
        # record number of exposed people per day
        self.cur_time_stamp += 1
        self.S_to_E_history[self.cur_time_stamp] = np.sum(self.S_to_E.current_val)
        current_val = 0

        # attribute access shortcut
        cur_time_stamp = self.cur_time_stamp

        # discrete convolution
        len_duration = self.viral_shed_duration * self.num_timesteps

        if self.cur_time_stamp >= len_duration - 1:
            current_val = self.S_to_E_history[
                          (cur_time_stamp - len_duration + 1):(cur_time_stamp + 1)] @ self.viral_shedding
        else:
            current_val = self.S_to_E_history[
                          :(cur_time_stamp + 1)] @ self.viral_shedding[-(cur_time_stamp + 1):]

        self.current_val = current_val
        self.cur_idx_timestep += 1
        self.current_val_list[self.cur_idx_timestep] = current_val

    def preprocess(self,
                   params: FluSubpopParams,
                   num_timesteps: int):
        # store the parameters locally
        self.viral_shed_duration = copy.deepcopy(params.viral_shed_duration)
        self.viral_shed_magnitude = copy.deepcopy(params.viral_shed_magnitude)
        self.viral_shed_peak = copy.deepcopy(params.viral_shed_peak)
        self.viral_shed_feces_mass = copy.deepcopy(params.viral_shed_feces_mass)
        self.num_timesteps = copy.deepcopy(num_timesteps)
        num_timesteps = np.float64(num_timesteps)
        self.viral_shedding = []
        # trapezoidal integral
        for time_idx in range(int(params.viral_shed_duration * self.num_timesteps)):
            cur_time_point = time_idx / num_timesteps
            next_time_point = (time_idx + 1) / num_timesteps
            next_time_log_viral_shedding = params.viral_shed_magnitude * next_time_point / \
                                           (params.viral_shed_peak ** 2 + next_time_point ** 2)
            if time_idx == 0:
                interval_viral_shedding = params.viral_shed_feces_mass * 0.5 * (
                        10 ** next_time_log_viral_shedding) / num_timesteps
            else:
                cur_time_log_viral_shedding = params.viral_shed_magnitude * cur_time_point / \
                                              (params.viral_shed_peak ** 2 + cur_time_point ** 2)
                interval_viral_shedding = params.viral_shed_feces_mass * 0.5 \
                                          * (
                                                  10 ** cur_time_log_viral_shedding + 10 ** next_time_log_viral_shedding) / num_timesteps
            self.viral_shedding.append(interval_viral_shedding)
        self.viral_shedding.reverse()
        self.viral_shedding = np.array(self.viral_shedding)

    def save_history(self) -> None:
        """
        Saves daily viral load (accumulated during one day) to history by appending current_val attribute
            to history_vals_list in place

        """
        daily_viral_load = np.sum(self.current_val_list)
        self.history_vals_list.append(daily_viral_load)
        # reset the index of current_val_list
        self.cur_idx_timestep = -1

    def reset(self) -> None:
        """
        Resets history_vals_list attribute to empty list.
        """
        self.flag_preprocessed = False
        self.viral_shedding = []
        self.viral_shed_duration = None
        self.viral_shed_magnitude = None
        self.viral_shed_peak = None
        self.viral_shed_feces_mass = None
        self.S_to_E_len = 5000  # preset to match the simulation time horizon
        self.S_to_E_history = np.zeros(self.S_to_E_len)
        self.cur_time_stamp = -1
        self.num_timesteps = None
        self.val_list_len = None
        self.current_val_list = None
        self.cur_idx_timestep = -1


class BetaReduct(clt.DynamicVal):
    """
    "Toy" function representing staged-alert policy
        that reduces transmission by 50% when more than 5%
        of the total population is infected. Note: the
        numbers are completely made up :)
    The "permanent_lockdown" toggle is to avoid "bang-bang"
        behavior where the staged-alert policy gets triggered
        one day and then is off the next, and then is on the
        day after, and so on... but as the name suggests,
        it IS permanent.
    TODO: replace with realistic function.
    """

    def __init__(self, init_val, is_enabled):
        super().__init__(init_val, is_enabled)
        self.permanent_lockdown = False

    def update_current_val(self, state, params):
        if np.sum(state.IS) / np.sum(params.total_pop_age_risk) > 0.05:
            self.current_val = .5
            self.permanent_lockdown = True
        else:
            if not self.permanent_lockdown:
                self.current_val = 0.0


class AbsoluteHumidity(clt.Schedule):

    def __init__(self,
                 init_val: Optional[np.ndarray | float] = None,
                 filepath: Optional[str] = None):
        """
        Args:
            init_val (Optional[np.ndarray | float]):
                starting value(s) at the beginning of the simulation
            timeseries_df (Optional[pd.DataFrame] = None):
                has a "date" column with strings in format `"YYYY-MM-DD"`
                of consecutive calendar days, and other columns
                corresponding to values on those days
        """

        super().__init__(init_val)

        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"], format='%m/%d/%y').dt.date
        self.time_series_df = df

    def update_current_val(self, params, current_date: datetime.date) -> None:
        self.current_val = self.time_series_df.loc[
            self.time_series_df["date"] == current_date, "humidity"].values[0]


class FluContactMatrix(clt.Schedule):
    """
    Flu contact matrix.

    Attributes:
        timeseries_df (pd.DataFrame):
            has a "date" column with strings in format "YYYY-MM-DD"
            of consecutive calendar days, and other columns
            named "is_school_day" (bool) and "is_work_day" (bool)
            corresponding to type of day.

    See parent class docstring for other attributes.
    """

    def __init__(self,
                 init_val: Optional[np.ndarray | float] = None,
                 calendar_df: pd.DataFrame = None):

        super().__init__(init_val)

        self.calendar_df = calendar_df

    def update_current_val(self,
                           subpop_params: FluSubpopParams,
                           current_date: datetime.date) -> None:

        df = self.calendar_df

        try:
            current_row = df[df["date"] == current_date].iloc[0]
            self.current_val = subpop_params.total_contact_matrix - \
                               (1 - current_row["is_school_day"]) * subpop_params.school_contact_matrix - \
                               (1 - current_row["is_work_day"]) * subpop_params.work_contact_matrix
        except IndexError:
            # print(f"Error: {current_date} is not in the Calendar's calendar_df. Using total contact matrix.")
            self.current_val = subpop_params.total_contact_matrix


def compute_wtd_presymp_asymp(subpop_state: FluSubpopState,
                              subpop_params: FluSubpopParams) -> np.ndarray:
    """
    Returns weighted sum of IP and IA compartment for
        subpopulation with given state and parameters.
        IP and IA are weighted by their relative infectiousness
        respectively, and then summed over risk groups.

    Returns:
        np.ndarray:
            |A| x 1 array -- where |A| is the number of age
            groups -- the ith element corresponds to the
            weighted sum of presymptomatic and asymptomatic
            individuals, also summed across all risk groups,
            for age group i.
    """

    # sum over risk groups
    wtd_IP = \
        subpop_params.IP_relative_inf * np.sum(subpop_state.IP, axis=1, keepdims=True)
    wtd_IA = \
        subpop_params.IA_relative_inf * np.sum(subpop_state.IA, axis=1, keepdims=True)

    return wtd_IP + wtd_IA


def compute_immunity_force(subpop_state: FluSubpopState,
                           subpop_params: FluSubpopParams) -> np.ndarray:
    """
    Computes a denominator that shows up repeatedly in
    travel model calculations.

    Returns:
        |A| x |R| array -- where |A| is the number of age groups
        and |R| is the number of risk groups -- representing
        the force of population-level immunity against infection
        -- used in the denominator of many computations
    """

    inf_risk_reduce = subpop_params.inf_risk_reduce
    if (np.asarray(inf_risk_reduce) != 1).all():
        proportional_risk_reduction = inf_risk_reduce / (1 - inf_risk_reduce)
    else:
        proportional_risk_reduction = 1

    return 1 + (proportional_risk_reduction *
                subpop_state.pop_immunity_inf)


def compute_common_coeff_force_of_infection(subpop_state: FluSubpopState,
                                            subpop_params: FluSubpopParams) -> np.ndarray:
    """
    Computes a coefficient that shows up repeatedly in
    travel model calculations.

    Returns:
        np.ndarray:
            |A| x |R| array -- where |A| is the number of age groups
            and |R| is the number of risk groups -- representing
            a kind of baseline transmission rate, adjusted for
            population-level immunity -- used as the coefficient of
            many computations
    """

    beta = subpop_params.beta_baseline * (1 + subpop_params.humidity_impact *
                                          np.exp(-180 * subpop_state.absolute_humidity))
    relative_suscept_by_age = subpop_params.relative_suscept_by_age
    immunity_force = compute_immunity_force(subpop_state, subpop_params)

    return beta * np.divide(relative_suscept_by_age, immunity_force)


def compute_pop_by_age(subpop_params: FluSubpopParams) -> np.ndarray:
    """
    Returns:
        np.ndarray:
            |A| x 1 array -- where |A| is the number of age groups --
            where ith element corresponds to total population
            (across all compartments, including "D", and across all risk groups)
            in age group i
    """

    return np.sum(subpop_params.total_pop_age_risk, axis=1, keepdims=True)


class FluInterSubpopRepo(clt.InterSubpopRepo):
    """
    Holds collection of SubpopState instances, with
        actions to query and interact with them.

    Attributes:
        subpop_models (dict):
            dictionary where keys are SubpopModel names and values are the
            SubpopModel instances -- whole dictionary contains all SubpopModel
            instances that comprise the associated MetapopModel
        subpop_names_mapping (dict):
            keys are names of SubpopModel instances and values are integers
            0, 1, ..., |L|-1, where |L| is the nubmer of subpopulations
            (associated SubpopModel instances). Provides a mapping between
            the name of the subpopulation and the row/column position in
            travel_proportions_array (and other associated indices used for
            intermediate computation on this class).
        travel_proportions_array (np.ndarray):
            |L| x |L| array, where |L| is the number of subpopulations
            (associated SubpopModel instances). Element i,j corresponds to
            proportion of subpopulation i that travels to subpopulation j
            (elements must be in [0,1]). The mapping of subpopulations is given by
            subpop_names_mapping.
        sum_prop_residents_traveling_out_array (np.ndarray):
            |L| x 1 array, where |L| is the number of subpopulations (associated
            SubpopModel instances). Element l is the  sum of the proportion of
            residents in given subpopulation who travel to a destination subpopulation,
            summed over all destinations but excluding residents' within-subpopulation
            traveling. Note that this value may be greater than 1.
        force_of_infection_array (np.ndarray):
            |L| x |A| x |R| array, where |L| is the number of subpopulations 
            (associated SubpopModel instances), |A| is the number of age groups, 
            and |R| is the number of risk groups. Element l,a,r corresponds to the
            total infection rate to residents of subpopulation l. The mapping of indices
            to the subpopulation is given by the ordering of subpop names in the "subpop_name" 
            column of the travel_proportions dataframe -- so that the l,a,r element
            in force_of_infection_array corresponds to the subpopulation whose name is ith 
            in the "subpop_name" column of  travel_proportions.

    See parent class InterSubpopRepo's docstring for
        attributes and additional methods.
    """

    def __init__(self,
                 subpop_models: dict,
                 subpop_names_mapping: dict,
                 travel_proportions_array: np.ndarray):

        super().__init__(subpop_models)

        self.subpop_names_mapping = subpop_names_mapping
        self.travel_proportions_array = travel_proportions_array

        self.sum_prop_residents_traveling_out_array = \
            self.compute_sum_prop_residents_traveling_out()

        #   This attribute will be set to an array using method compute_shared_quantities()
        #   during the associated MetapopModel's simulate_until_day() method.
        self.force_of_infection_array = None

    def compute_shared_quantities(self):
        """
        Updates force_of_infection_array attribute in-place.
        """

        force_of_infection_array = []

        wtd_no_symp_by_age_cache = self.create_wtd_no_symp_by_age_cache()
        effective_pop_by_age_cache = self.create_effective_pop_by_age_cache()

        # Extract subpop names in correct order corresponding to their mapping
        subpop_names_ordered = sorted(self.subpop_names_mapping, key=self.subpop_names_mapping.get)

        for subpop_name in subpop_names_ordered:
            force_of_infection = \
                self.inf_from_home_region_movement(subpop_name,
                                                   wtd_no_symp_by_age_cache,
                                                   effective_pop_by_age_cache) + \
                self.inf_from_visitors(subpop_name,
                                       wtd_no_symp_by_age_cache,
                                       effective_pop_by_age_cache) + \
                self.inf_from_residents_traveling(subpop_name,
                                                  wtd_no_symp_by_age_cache,
                                                  effective_pop_by_age_cache)

            force_of_infection_array.append(force_of_infection)

        self.force_of_infection_array = np.asarray(force_of_infection_array)

    def prop_residents_traveling_pairwise(self,
                                          origin_subpop_name: str,
                                          dest_subpop_name: str) -> float:
        """
        Returns:
             (float):
                the proportion in [0,1] of residents in given origin subpopulation
                who travel to the given destination subpopulation.
        """

        subpop_names_mapping = self.subpop_names_mapping

        return self.travel_proportions_array[subpop_names_mapping[origin_subpop_name],
                                             subpop_names_mapping[dest_subpop_name]]

    def compute_sum_prop_residents_traveling_out(self) -> np.ndarray:
        """
        Returns |L| x 1 array, where |L| is the number of subpopulations,
        corresponding to the sum of the proportion of residents in given subpopulation
        who travel to a destination subpopulation, summed over all destinations
        but excluding residents' within-subpopulation traveling.

        Note that each element's value may be greater than 1!
        """

        travel_proportions_array = self.travel_proportions_array

        # For each subpopulation (row index), sum the travel proportions
        #   in that row but subtract the diagonal element (because
        #   we are excluding residents who travel within their home subpopulation).
        return np.sum(travel_proportions_array, axis=1, keepdims=True) - \
               np.diag(travel_proportions_array).reshape(-1, 1)

    def create_wtd_no_symp_by_age_cache(self) -> dict:
        """
        Creates cache (dictionary) of weighted sum of
        non-symptomatic infectious people (IP and IA),
        weighted by relative infectiousness, summed over risk
        groups and categorized by age, for a given subpopulation.

        Keys are the subpopulation name, values are the
        |A| x |1| array corresponding to the above weighted sum,
        where |A| is the number of age groups.
        """

        wtd_no_symp_by_age_cache = {}

        for subpop_model in self.subpop_models.values():
            wtd_no_symp_by_age_cache[subpop_model.name] = \
                compute_wtd_presymp_asymp(subpop_model.state,
                                          subpop_model.params)

        return wtd_no_symp_by_age_cache

    def create_pop_by_age_cache(self) -> dict:
        """
        Creates cache (dictionary) of total population
        for a given subpopulation, summed over risk groups
        and categorized by age.

        Keys are the subpopulation name, values are the
        |A| x |1| array corresponding to the above population sum,
        where |A| is the number of age groups.
        """

        pop_by_age_cache = {}

        for subpop_model in self.subpop_models.values():
            pop_by_age_cache[subpop_model.name] = \
                compute_pop_by_age(subpop_model.params)

        return pop_by_age_cache

    def create_pop_healthy_by_age(self) -> dict:
        """
        Creates cache (dictionary) of weighted sum of
        "healthy-presenting" people (those not in compartments
        IS and H). Corresponds to total population minus
        IS and H populations weighted by multiplier that reduces
        contact rates of sick individuals, summed over risk groups
        and categorized by age, for a given subpopulation.

        Keys are the subpopulation name, values are the
        |A| x |1| array corresponding to the above population sum,
        where |A| is the number of age groups.
        """

        pop_healthy_by_age = {}

        pop_by_age_cache = self.create_pop_by_age_cache()

        for subpop_model in self.subpop_models.values():
            subpop_name = subpop_model.name
            subpop_state = subpop_model.state

            pop_healthy_by_age[subpop_name] = \
                pop_by_age_cache[subpop_name] - \
                (1 - subpop_model.params.contact_mult_symp) * \
                np.sum(subpop_state.IS, axis=1, keepdims=True) - \
                np.sum(subpop_state.H, axis=1, keepdims=True)

        return pop_healthy_by_age

    def sum_wtd_visitors_by_age(self,
                                subpop_name: str,
                                pop_healthy_by_age: dict) -> np.ndarray:
        """
        Returns |A| x 1 array corresponding to weighted visitors
        to given subpopulation, where |A| is the number of age groups.
        """

        wtd_visitors_by_origin = []

        for origin_subpop_name in self.subpop_models.keys():
            if origin_subpop_name == subpop_name:
                continue
            else:
                wtd_visitors_by_origin.append(
                    self.prop_residents_traveling_pairwise(origin_subpop_name,
                                                           subpop_name) *
                    pop_healthy_by_age[origin_subpop_name])

        return np.sum(wtd_visitors_by_origin, axis=1)

    def create_effective_pop_by_age_cache(self):
        """
        Creates cache (dictionary) of effective population
        for each subpopulation, which corresponds to a population
        adjustment to account for non-traveling sick residents,
        traveling residents, and outside visitors.

        Keys are the subpopulation name, values are the
        |A| x |1| array corresponding to the above population sum,
        where |A| is the number of age groups.
        """

        pop_by_age_cache = self.create_pop_by_age_cache()
        pop_healthy_by_age = self.create_pop_healthy_by_age()

        effective_pop_by_age_cache = {}

        subpop_names_mapping = self.subpop_names_mapping

        for subpop_model in self.subpop_models.values():
            subpop_name = subpop_model.name
            subpop_params = subpop_model.params

            effective_pop_by_age_cache[subpop_name] = \
                pop_by_age_cache[subpop_name] + \
                subpop_params.prop_time_away_by_age * \
                (self.sum_wtd_visitors_by_age(subpop_name, pop_healthy_by_age) -
                 self.sum_prop_residents_traveling_out_array[subpop_names_mapping[subpop_name]] *
                 pop_healthy_by_age[subpop_name])

        return effective_pop_by_age_cache

    def inf_from_home_region_movement(self,
                                      subpop_name: str,
                                      wtd_no_symp_by_age_cache: dict,
                                      effective_pop_by_age_cache: dict) -> np.ndarray:
        """
        Returns |A| x 1 array corresponding to infection rate due to
        residents in given subpopulation traveling within their own
        subpopulation, where |A| is the number of age groups.
        """

        # Math reminder -- other than the summation over proportion of residents
        #   traveling, there is only ONE subpopulation index here.

        subpop_model = self.subpop_models[subpop_name]
        subpop_state = subpop_model.state
        subpop_params = subpop_model.params
        subpop_names_mapping = self.subpop_names_mapping

        prop_time_away_by_age = subpop_params.prop_time_away_by_age
        sum_prop_residents_traveling_out = \
            self.sum_prop_residents_traveling_out_array[subpop_names_mapping[subpop_name]]

        contact_matrix = subpop_state.flu_contact_matrix
        wtd_infected_by_age = \
            wtd_no_symp_by_age_cache[subpop_name] + \
            np.sum(subpop_state.IS, axis=1, keepdims=True)

        wtd_infected_to_pop_ratio = np.divide(wtd_infected_by_age,
                                              effective_pop_by_age_cache[subpop_name])

        common_coeff = compute_common_coeff_force_of_infection(subpop_state, subpop_params)

        return common_coeff * (1 - prop_time_away_by_age * sum_prop_residents_traveling_out) * \
               np.matmul(contact_matrix, wtd_infected_to_pop_ratio)

    def inf_from_visitors(self,
                          subpop_name: str,
                          wtd_no_symp_by_age_cache: dict,
                          effective_pop_by_age_cache: dict) -> np.ndarray:
        """
        Returns |L| x |A| x 1 array corresponding to infection rate to given
        subpopulation, due to outside visitors from other subpopulations,
        where |A| is the number of age groups.
        """

        subpop_model = self.subpop_models[subpop_name]
        subpop_state = subpop_model.state
        subpop_params = subpop_model.params

        all_subpop_models_names = self.subpop_models.keys()

        # Create a list to store all pairwise values
        # Each element corresponds to the infection rate due to
        #   visitors from another subpopulation
        # These elements are summed to obtain the overall infection rate
        #   from outside visitors from ALL other subpopulations
        inf_from_visitors_pairwise = []

        for visitors_subpop_name in all_subpop_models_names:

            # Do not include residential travel within same subpopulation
            #   -- this is handled with inf_from_home_region_movement
            if visitors_subpop_name == subpop_name:
                continue

            else:
                # Math reminder -- the weighted sum of infected people is for
                #   other subpopulations, but the other indices/values are for
                #   the input subpopulation.

                contact_mult_symp = subpop_params.contact_mult_symp

                prop_residents_traveling_pairwise = \
                    self.prop_residents_traveling_pairwise(visitors_subpop_name,
                                                           subpop_name)

                prop_time_away_by_age = subpop_params.prop_time_away_by_age

                contact_matrix = subpop_state.flu_contact_matrix

                visitors_subpop_model = self.subpop_models[visitors_subpop_name]
                visitors_subpop_state = visitors_subpop_model.state

                wtd_infected_visitors_by_age = \
                    wtd_no_symp_by_age_cache[visitors_subpop_name] + \
                    contact_mult_symp * np.sum(visitors_subpop_state.IS, axis=1, keepdims=True)

                wtd_infected_to_pop_ratio = \
                    np.divide(wtd_infected_visitors_by_age,
                              effective_pop_by_age_cache[subpop_name])

                inf_from_visitors_pairwise.append(
                    prop_residents_traveling_pairwise *
                    np.matmul(contact_matrix,
                              prop_time_away_by_age * wtd_infected_to_pop_ratio))

        common_coeff = compute_common_coeff_force_of_infection(subpop_state, subpop_params)

        return np.sum(common_coeff * subpop_params.contact_mult_travel *
                      np.asarray(inf_from_visitors_pairwise), axis=0)

    def inf_from_residents_traveling(self,
                                     subpop_name: str,
                                     wtd_no_symp_by_age_cache: dict,
                                     effective_pop_by_age_cache: dict) -> np.ndarray:
        """
        Returns |A| x 1 array corresponding to infection rate to given
        subpopulation, due to residents getting infected while visiting
        other subpopulations and then bringing the infections back home,
        where |A| is the number of age groups.
        """

        subpop_model = self.subpop_models[subpop_name]
        subpop_state = subpop_model.state
        subpop_params = subpop_model.params

        all_subpop_models_names = self.subpop_models.keys()

        # Create a list to store all pairwise values
        # Each element corresponds to the infection rate due to
        #   residents visiting another subpopulation
        # These elements are summed to obtain the overall infection rate
        #   from residents getting infected in ALL other subpopulations
        inf_from_residents_traveling_pairwise = []

        for dest_subpop_name in all_subpop_models_names:

            if dest_subpop_name == subpop_name:
                continue

            else:
                # Math reminder -- the weighted sum of infected people AND
                #   the effective population refers to the OTHER subpopulation
                #   -- because we are dealing with residents that get
                #   infected in OTHER subpopulations

                prop_residents_traveling_pairwise = self.prop_residents_traveling_pairwise(
                    subpop_name, dest_subpop_name
                )

                prop_time_away_by_age = subpop_params.prop_time_away_by_age

                contact_matrix = subpop_state.flu_contact_matrix

                dest_subpop_model = self.subpop_models[dest_subpop_name]
                dest_subpop_state = dest_subpop_model.state

                # Weighted infected at DESTINATION subpopulation
                wtd_infected_dest_by_age = \
                    wtd_no_symp_by_age_cache[dest_subpop_name] + \
                    np.sum(dest_subpop_state.IS, axis=1, keepdims=True)

                # Ratio of weighted infected (at destination) to
                #   effective population (at destination)
                wtd_infected_to_pop_dest_ratio = \
                    np.divide(wtd_infected_dest_by_age,
                              effective_pop_by_age_cache[dest_subpop_name])

                inf_from_residents_traveling_pairwise.append(
                    prop_residents_traveling_pairwise *
                    wtd_infected_to_pop_dest_ratio *
                    np.matmul(contact_matrix,
                              prop_time_away_by_age))

        common_coeff = compute_common_coeff_force_of_infection(subpop_state, subpop_params)

        return np.sum(common_coeff * subpop_params.contact_mult_travel *
                      inf_from_residents_traveling_pairwise, axis=0)


class ForceOfInfection(clt.InteractionTerm):
    """
    InteractionTerm-derived class for modeling S_to_E transition rate
        for a given subpopulation, which depends on the
        subpopulation's contact matrix, population-level immunity,
        travel dynamics across subpopulations, and also
        the states of other subpopulations.
    """

    def __init__(self,
                 subpop_name: str):
        super().__init__()
        self.subpop_name = subpop_name

    def update_current_val(self,
                           inter_subpop_repo: FluInterSubpopRepo,
                           subpop_params: FluSubpopParams) -> None:
        subpop_name = self.subpop_name
        subpop_names_mapping = inter_subpop_repo.subpop_names_mapping

        self.current_val = \
            inter_subpop_repo.force_of_infection_array[subpop_names_mapping[subpop_name]]


class FluSubpopModel(clt.SubpopModel):
    """
    Class for creating ImmunoSEIRS flu model with predetermined fixed
    structure -- initial values and epidemiological structure are
    populated by user-specified JSON files.

    Key method create_transmission_model returns a SubpopModel
    instance with S-E-I-H-R-D compartments and pop_immunity_inf
    and pop_immunity_hosp epi metrics. 
    
    The update structure is as follows:
        - S <- S + R_to_S - S_to_E
        - E <- E + S_to_E - E_to_IP - E_to_IA
        - IA <- IA + E_to_IA - IA_to_R 
        - IP <- IP + E_to_IP - IP_to_IS
        - IS <- IS + IP_to_IS - IS_to_R - IS_to_H
        - H <- H + IS_to_H - H_to_R - H_to_D
        - R <- R + IS_to_R + H_to_R - R_to_S
        - D <- D + H_to_D

    The following are TransitionVariable instances:
        - R_to_S is a RecoveredToSusceptible instance
        - S_to_E is a SusceptibleToExposed instance
        - IP_to_IS is a PresympToSymp instance
        - IS_to_H is a SympToHosp instance
        - IS_to_R is a SympToRecovered instance
        - H_to_R is a HospToRecovered instance
        - H_to_D is a HospToDead instance

    There are three TransitionVariableGroups:
        - E_out (handles E_to_IP and E_to_IA)
        - IS_out (handles IS_to_H and IS_to_R)
        - H_out (handles H_to_R and H_to_D)

    The following are EpiMetric instances:
        - pop_immunity_inf is a PopulationImmunityInf instance
        - pop_immunity_hosp is a PopulationImmunityHosp instance

    Transition rates and update formulas are specified in
        corresponding classes.

    See parent class SubpopModel's docstring for additional attributes.
    """

    def __init__(self,
                 compartments_epi_metrics: dict,
                 params: dict,
                 config: dict,
                 calendar_df: pd.DataFrame,
                 RNG: np.random.Generator,
                 absolute_humidity_filepath: str,
                 name: str = "",
                 wastewater_enabled: bool = False):
        """
        Args:
            compartments_epi_metrics (dict):
                holds current simulation state information,
                such as current values of epidemiological compartments
                and epi metrics -- keys and values respectively
                must match field names and format of FluSubpopState.
            params (dict):
                holds epidemiological parameter values -- keys and
                values respectively must match field names and
                format of FluSubpopParams.
            config (dict):
                holds configuration values -- keys and values
                respectively must match field names and format of
                Config.
            calendar_df (pd.DataFrame):
                DataFrame with columns "date", "is_school_day", and
                "is_work_day" -- "date" entries are either strings
                format with "YYYY-MM-DD" or datetime.date objects,
                and "is_school_day" and "is_work_day" entries are
                Booleans indicating if that date is a school
                day or work day
            RNG (np.random.Generator):
                numpy random generator object used to obtain
                random numbers.
            absolute_humidity_filepath (str):
                filepath (ending in ".csv") corresponding to
                absolute humidity data -- see `AbsoluteHumidity`
                class for CSV file specifications
            name (str):
                unique name of MetapopModel instance.
            wastewater_enabled (bool):
                if True, includes "wastewater" EpiMetric. Otherwise,
                excludes it.
        """

        # Assign config, params, and state to model-specific
        # types of dataclasses that have epidemiological parameter information
        # and sim state information

        self.wastewater_enabled = wastewater_enabled
        self.absolute_humidity_filepath = absolute_humidity_filepath

        if not all(isinstance(val, datetime.date) for val in calendar_df["date"]):
            try:
                calendar_df["date"] = pd.to_datetime(calendar_df["date"], format="%Y-%m-%d").dt.date
            except ValueError:
                print("Error: The date format should be YYYY-MM-DD.")

        self.calendar_df = calendar_df

        state = clt.make_dataclass_from_dict(FluSubpopState, compartments_epi_metrics)
        params = clt.make_dataclass_from_dict(FluSubpopParams, params)
        config = clt.make_dataclass_from_dict(clt.Config, config)

        # IMPORTANT NOTE: as always, we must be careful with mutable objects
        #   and generally use deep copies to avoid modification of the same
        #   object. But in this function call, using deep copies is unnecessary
        #   (redundant) because the parent class SubpopModel's __init__()
        #   creates deep copies.
        super().__init__(state, params, config, RNG, name)

    def create_interaction_terms(self) -> sc.objdict[str, clt.InteractionTerm]:

        # Create interaction terms
        # If there is no associated `MetapopModel`, then
        #   there is no travel model, so do not create `ForceOfInfection`
        #   instance -- there are no interaction terms for this `SubpopModel`.

        interaction_terms = sc.objdict()

        if self.metapop_model:
            interaction_terms["force_of_infection"] = ForceOfInfection(self.name)

        return interaction_terms

    def create_compartments(self) -> sc.objdict[str, clt.Compartment]:

        # Create `Compartment` instances S-E-IA-IP-IS-H-R-D (7 compartments total),
        #   save in sc.objdict, and return objdict

        compartments = sc.objdict()

        for name in ("S", "E", "IP", "IS", "IA", "H", "R", "D"):
            compartments[name] = clt.Compartment(getattr(self.state, name))

        return compartments

    def create_dynamic_vals(self) -> sc.objdict[str, clt.DynamicVal]:
        """
        Create all `DynamicVal` instances, save in sc.objdict, and return objdict
        """

        dynamic_vals = sc.objdict()

        dynamic_vals["beta_reduct"] = BetaReduct(init_val=0.0,
                                                 is_enabled=False)

        return dynamic_vals

    def create_schedules(self) -> sc.objdict[str, clt.Schedule]:
        """
        Create all `Schedule` instances, save in sc.objdict, and return objdict
        """

        schedules = sc.objdict()

        schedules["absolute_humidity"] = AbsoluteHumidity(filepath=self.absolute_humidity_filepath)
        schedules["flu_contact_matrix"] = FluContactMatrix(init_val=None,
                                                           calendar_df=self.calendar_df)

        return schedules

    def create_transition_variables(self,
                                    compartments: sc.objdict[str, clt.Compartment]) -> \
            sc.objdict[str, clt.TransitionVariable]:
        """
        Create all `TransitionVariable` instances,
            save in sc.objdict, and return objdict
        """

        # NOTE: see the parent class `SubpopModel`'s `__init__()` --
        #   `create_transition_variables` is called after
        #   `self.config` is assigned

        transition_type = self.config.transition_type

        transition_variables = sc.objdict()

        S = compartments.S
        E = compartments.E
        IP = compartments.IP
        IS = compartments.IS
        IA = compartments.IA
        H = compartments.H
        R = compartments.R
        D = compartments.D

        transition_variables.R_to_S = RecoveredToSusceptible(R, S, transition_type)
        transition_variables.S_to_E = SusceptibleToExposed(S, E, transition_type)
        transition_variables.IP_to_IS = PresympToSymp(IP, IS, transition_type)
        transition_variables.IA_to_R = AsympToRecovered(IA, R, transition_type)
        transition_variables.E_to_IP = ExposedToPresymp(E, IP, transition_type, True)
        transition_variables.E_to_IA = ExposedToAsymp(E, IA, transition_type, True)
        transition_variables.IS_to_R = SympToRecovered(IS, R, transition_type, True)
        transition_variables.IS_to_H = SympToHosp(IS, H, transition_type, True)
        transition_variables.H_to_R = HospToRecovered(H, R, transition_type, True)
        transition_variables.H_to_D = HospToDead(H, D, transition_type, True)

        return transition_variables

    def create_transition_variable_groups(
            self,
            compartments: sc.objdict[str, clt.Compartment],
            transition_variables: sc.objdict[str, clt.TransitionVariable]) \
            -> sc.objdict[str, clt.TransitionVariableGroup]:
        """
        Create all transition variable groups described in docstring (2 transition
        variable groups total), save in sc.objdict, return
        """

        # Shortcuts for attribute access
        # NOTE: see the parent class SubpopModel's __init__() --
        #   create_transition_variable_groups is called after
        #   self.config is assigned

        transition_type = self.config.transition_type

        transition_variable_groups = sc.objdict()

        transition_variable_groups.E_out = clt.TransitionVariableGroup(compartments.E,
                                                                       transition_type,
                                                                       (transition_variables.E_to_IP,
                                                                        transition_variables.E_to_IA))

        transition_variable_groups.IS_out = clt.TransitionVariableGroup(compartments.IS,
                                                                        transition_type,
                                                                        (transition_variables.IS_to_R,
                                                                         transition_variables.IS_to_H))

        transition_variable_groups.H_out = clt.TransitionVariableGroup(compartments.H,
                                                                       transition_type,
                                                                       (transition_variables.H_to_R,
                                                                        transition_variables.H_to_D))

        return transition_variable_groups

    def create_epi_metrics(self,
                           transition_variables: sc.objdict[str, clt.TransitionVariable]) \
            -> sc.objdict[str, clt.EpiMetric]:
        """
        Create all epi metric described in docstring (2 state
        variables total), save in sc.objdict, and return objdict
        """

        epi_metrics = sc.objdict()

        epi_metrics.pop_immunity_inf = \
            PopulationImmunityInf(getattr(self.state, "pop_immunity_inf"),
                                  transition_variables.R_to_S)

        if self.wastewater_enabled:
            epi_metrics.wastewater = \
                Wastewater(getattr(self.state, "wastewater"),  # initial value is set to null for now
                           transition_variables.S_to_E)

        epi_metrics.pop_immunity_hosp = \
            PopulationImmunityHosp(getattr(self.state, "pop_immunity_hosp"),
                                   transition_variables.R_to_S)

        return epi_metrics

    def run_model_checks(self,
                         include_printing=True):
        """
        Run flu model checks.

        Input checks:
            - SubpopState and SubpopParams instances should have
                fields with nonnegative values.
            - Initial values of compartments should be nonnegative
                integers.
            - Population sum of compartments should match
                total population computed at initialization
                (user should not change initial values
                after initialization).
        """

        if include_printing:
            print(">>> Running FluSubpopModel checks... \n")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        error_counter = 0

        state = self.state
        params = self.params

        for name, val in list(vars(state).items()) + list(vars(params).items()):
            if isinstance(val, np.ndarray):
                flattened_array = val.flatten()
                for val in flattened_array:
                    if val < 0:
                        if include_printing:
                            print(f"STOP! INPUT ERROR: {name} should not have negative values.")
                        error_counter += 1
            elif isinstance(val, float):
                if val < 0:
                    if include_printing:
                        print(f"STOP! INPUT ERROR: {name} should not be negative.")
                    error_counter += 1

        compartment_population_sum = np.zeros((self.params.num_age_groups,
                                               self.params.num_risk_groups))

        for name, compartment in self.compartments.items():
            compartment_population_sum += compartment.current_val
            flattened_current_val = compartment.current_val.flatten()
            for val in flattened_current_val:
                if val != int(val):
                    if include_printing:
                        print(f"STOP! INPUT ERROR: {name} should not have non-integer values.")
                    error_counter += 1
                if val < 0:
                    if include_printing:
                        print(f"STOP! INPUT ERROR: {name} should not have negative values.")
                    error_counter += 1

        if (compartment_population_sum != self.params.total_pop_age_risk).any():
            if include_printing:
                print(f"STOP! INPUT ERROR: sum of population in compartments must \n"
                      f"match specified total population value. Check \n"
                      f"\"total_pop_age_risk\" in model's \"params\" attribute \n"
                      f"and check compartments in state variables' init vals JSON.")
            error_counter += 1

        if error_counter == 0:
            if include_printing:
                print("OKAY! FluSubpopModel instance has passed input checks: \n"
                      "Compartment populations are nonnegative whole numbers \n"
                      "and add up to \"total_pop_age_risk\" in model's \n"
                      "\"params attribute.\" Fixed parameters are nonnegative.")
            return True
        else:
            if include_printing:
                print(f"Need to fix {error_counter} errors before simulating model.")
            return False


class FluMetapopModel(clt.MetapopModel):
    """
    MetapopModel-derived class specific to flu model.
    Assigns an instance of FluInterSubpopRepo to model --
    the repository holds all subpopulation models included
    in the metapopulation model, and also a DataFrame with
    travel proportions information.
    """

    def check_travel_proportions(self,
                                 include_printing=True):
        """
        Checks to make sure travel_proportions_mapping and
        travel_proportions_array (located on InterSubpopRepo instance)
        have correct format on MetapopModel.

        Validates that subpop_names_mapping is a dictionary
        whose keys are names of associated SubpopModel instances and
        whose length matches both the number of rows and columns of
        travel_proportions_array (is a square matrix). Makes sure that
        numerical values in travel_proportions_array are between [0,1].
        """

        if include_printing:
            print(">>> Running travel proportions checks... \n")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        error_counter = 0

        # Extract unique subpop_name values
        # Ensure uniqueness
        # Ensure they match the unique string IDs of each associated
        #   SubpopModel instance
        subpop_names = self.inter_subpop_repo.subpop_names_mapping
        num_subpop_names = len(subpop_names)
        travel_proportions_array = self.inter_subpop_repo.travel_proportions_array
        if np.shape(travel_proportions_array) != \
                (num_subpop_names, num_subpop_names):
            error_counter += 1
            if include_printing:
                print("Length of subpop_names_mapping dictionary must equal "
                      "number of rows and number of columns of travel_proportions_array.")
        if set(subpop_names) != set(self.subpop_models.keys()):
            error_counter += 1
            if include_printing:
                print("Each key in subpop_names_mapping must match a "
                      "name of an associated SubpopModel instance.")

        # Check if other values are between 0 and 1
        if not ((travel_proportions_array >= 0).all() and (travel_proportions_array <= 1).all()):
            error_counter += 1
            if include_printing:
                print("All numerical values must be between 0 and 1 "
                      "because they represent proportions.")

        if error_counter == 0:
            if include_printing:
                print("OKAY! Travel proportions input into FluMetapopModel is "
                      "correctly formatted.")
            return True
        else:
            if include_printing:
                print(f"Need to fix {error_counter} errors before simulating model.")
            return False

    def run_model_checks(self):

        self.check_travel_proportions()

        for subpop_model in self.subpop_models.values():
            subpop_model.run_model_checks()
