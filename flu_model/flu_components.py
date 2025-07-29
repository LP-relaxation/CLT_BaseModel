import datetime
import copy

import numpy as np
import pandas as pd
import sciris as sc

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from abc import ABC, abstractmethod

import torch
import clt_base as clt

from flu_data_structures import FluMetapopParamsTensors, FluMetapopStateTensors, FluPrecomputedTensors

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
        inf_induced_saturation (np.ndarray of positive floats):
            constant(s) modeling saturation of antibody
            production of infected individuals.
        inf_induced_immune_wane (positive float):
            rate at which infection-induced immunity
            against infection wanes.
        vax_induced_saturation (np.ndarray of positive floats):
            constant(s) modeling saturation of antibody
            production of vaccinated individuals.
        vax_induced_immune_wane (positive float):
            rate at which vaccine-induced immunity
            against infection wanes.
        inf_induced_inf_risk_reduce (positive float):
            reduction in risk of getting infected
            after getting infected
        inf_induced_hosp_risk_reduce (positive float):
            reduction in risk of hospitalization
            after getting infected
        inf_induced_death_risk_reduce (positive float):
            reduction in risk of death
            after getting infected
        vax_induced_inf_risk_reduce (positive float):
            reduction in risk of getting infected
            after getting vaccinated
        vax_induced_hosp_risk_reduce (positive float):
            reduction in risk of hospitalization
            after getting vaccinated
        vax_induced_death_risk_reduce (positive float):
            reduction in risk of death
            after getting vaccinated
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
        daily_vaccines (int):
            WARNING: THIS IS A PLACEHOLDER. See `DailyVaccines`
            class for more information. This will be deleted once
            we have historical vaccine data and set up the
            `DailyVaccines` `Schedule` properly.

    """

    num_age_groups: Optional[int] = None
    num_risk_groups: Optional[int] = None
    beta_baseline: Optional[float] = None
    total_pop_age_risk: Optional[np.ndarray] = None
    humidity_impact: Optional[float] = None

    inf_induced_saturation: Optional[float] = None
    inf_induced_immune_wane: Optional[float] = None
    vax_induced_saturation: Optional[float] = None
    vax_induced_immune_wane: Optional[float] = None
    inf_induced_inf_risk_reduce: Optional[float] = None
    inf_induced_hosp_risk_reduce: Optional[float] = None
    inf_induced_death_risk_reduce: Optional[float] = None
    vax_induced_inf_risk_reduce: Optional[float] = None
    vax_induced_hosp_risk_reduce: Optional[float] = None
    vax_induced_death_risk_reduce: Optional[float] = None

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
    relative_suscept_by_age: Optional[np.ndarray] = None

    prop_time_away_by_age: Optional[np.ndarray] = None
    contact_mult_travel: Optional[float] = None
    contact_mult_symp: Optional[float] = None

    total_contact_matrix: Optional[np.ndarray] = None
    school_contact_matrix: Optional[np.ndarray] = None
    work_contact_matrix: Optional[np.ndarray] = None

    daily_vaccines_constant: Optional[int] = None


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
        M (np.ndarray of positive floats):
            infection-induced population-level immunity
            for age-risk groups (holds current_val
            of EpiMetric "M").
        Mv (np.ndarray of positive floats):
            vaccine-induced population-level immunity
            for age-risk groups (holds current_val
            of EpiMetric "Mv").
        absolute_humidity (positive float):
            grams of water vapor per cubic meter g/m^3,
            used as seasonality parameter that influences
            transmission rate beta_baseline.
        flu_contact_matrix (np.ndarray of positive floats):
            |A| x |A| array, where |A| is the number of age
            groups -- element (a, a') corresponds to the number 
            of contacts that a person in age group a
            has with people in age-risk group a'.
        beta_reduce (float in [0,1]):
            starting value of DynamicVal "beta_reduce" on
            starting day of simulation -- this DynamicVal
            emulates a simple staged-alert policy
        force_of_infection (np.ndarray of positive floats):
            total force of infection from movement within
            home location, travel to other locations,
            and visitors from other locations
        daily_vaccines (np.ndarray of positive ints):
            holds current value of DailyVaccines instance,
            corresponding number of individuals who received influenza
            vaccine on that day, for given age-risk group
            (generally derived from historical data)
    """

    S: Optional[np.ndarray] = None
    E: Optional[np.ndarray] = None
    IP: Optional[np.ndarray] = None
    IS: Optional[np.ndarray] = None
    IA: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    D: Optional[np.ndarray] = None

    M: Optional[np.ndarray] = None
    Mv: Optional[np.ndarray] = None

    absolute_humidity: Optional[float] = None
    flu_contact_matrix: Optional[np.ndarray] = None
    beta_reduce: Optional[float] = 0.0
    force_of_infection: Optional[np.ndarray] = None

    daily_vaccines: Optional[np.ndarray] = None


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
    the rate depends on the `force_of_infection` attribute,
    which is a function of other subpopulations' states and
    parameters, and travel between subpopulations.

    If there is no metapopulation model, the rate
    is much simpler.

    See parent class docstring for other attributes.
    """

    def __init__(self,
                 origin: clt.Compartment,
                 destination: clt.Compartment,
                 transition_type: clt.TransitionTypes,
                 is_jointly_distributed: str = False):

        super().__init__(origin,
                         destination,
                         transition_type,
                         is_jointly_distributed)

        self.force_of_infection = None

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:

        # If `force_of_infection` has not been updated,
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
                             np.divide(np.reshape(np.sum(state.IS, axis=1), (2, 1)) + wtd_presymp_asymp,
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

        hosp_risk_reduce = params.inf_induced_hosp_risk_reduce

        if (np.asarray(hosp_risk_reduce) != 1).all():
            proportional_risk_reduction = hosp_risk_reduce / (1 - hosp_risk_reduce)
        else:
            proportional_risk_reduction = 1

        return np.asarray(params.IS_to_H_rate * params.IS_to_H_adjusted_prop /
                          (1 + proportional_risk_reduction * state.Mv))


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

        inf_induced_death_risk_reduce = params.inf_induced_death_risk_reduce
        vax_induced_death_risk_reduce = params.vax_induced_death_risk_reduce

        if (np.asarray(inf_induced_death_risk_reduce) != 1).all():
            inf_induced_proportional_risk_reduce = \
                inf_induced_death_risk_reduce / (1 - inf_induced_death_risk_reduce)
        else:
            inf_induced_proportional_risk_reduce = 1

        if (np.asarray(vax_induced_death_risk_reduce) != 1).all():
            vax_induced_proportional_risk_reduce = \
                vax_induced_death_risk_reduce / (1 - vax_induced_death_risk_reduce)
        else:
            vax_induced_proportional_risk_reduce = 1

        return np.asarray(params.H_to_D_adjusted_prop * params.H_to_D_rate /
                          (1 + inf_induced_proportional_risk_reduce * state.M +
                           vax_induced_proportional_risk_reduce * state.Mv))


class InfInducedImmunity(clt.EpiMetric):
    """
    EpiMetric-derived class for infection-induced
    population-level immunity.

    Population-level immunity increases as people move
    from "R" to "S" -- this is a design choice intended
    to avoid "double-counting." People in "R" cannot be
    infected at all. People who move from "R" to "S"
    are susceptible again, but these recently-recovered people
    should have partial immunity. To handle this phenomenon,
    this epi metric increases as people move from "R" to "S."

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
        # Note: the current values of transition variables already include
        #   discretization (division by the number of timesteps) -- therefore,
        #   we do not divide the first part of this equation by the number of
        #   timesteps -- see `TransitionVariable` class's methods for getting
        #   various realizations for more information

        return (self.R_to_S.current_val / params.total_pop_age_risk) * \
               (1 - params.inf_induced_saturation * state.M - params.vax_induced_saturation * state.Mv) - \
               params.inf_induced_immune_wane * state.M / num_timesteps


class VaxInducedImmunity(clt.EpiMetric):
    """
    EpiMetric-derived class for vaccine-induced
    population-level immunity.
    """

    def __init__(self, init_val):
        super().__init__(init_val)

    def get_change_in_current_val(self,
                                  state: FluSubpopState,
                                  params: FluSubpopParams,
                                  num_timesteps: int) -> np.ndarray:
        # Note: `state.daily_vaccines` (based on the value of the `DailyVaccines`
        #   `Schedule` is NOT divided by the number of timesteps -- so we need to
        #   do this division in the equation here.

        return state.daily_vaccines / (params.total_pop_age_risk * num_timesteps) - \
               params.vax_induced_immune_wane * state.Mv / num_timesteps


class BetaReduce(clt.DynamicVal):
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


class DailyVaccines(clt.Schedule):

    def __init__(self,
                 init_val: Optional[np.ndarray | float] = None,
                 filepath: Optional[str] = None):
        """
        WARNING: THIS IS A PLACEHOLDER RIGHT NOW --
        NEED TO REPLACE WITH REAL FUNCTION. Currently returns a
        constant `params.daily_vaccines_constant` each day --
        when we have historical data, we will need to use the
        dataframe to build this class and grab the historical
        according to the date. Then we should delete the
        `params.daily_vaccines_constant` value.


        Args:
            init_val (Optional[np.ndarray | float]):
                starting value(s) at the beginning of the simulation
            timeseries_df (Optional[pd.DataFrame] = None):
                has a "date" column with strings in format `"YYYY-MM-DD"`
                of consecutive calendar days, and other columns
                corresponding to values on those days
        """

        super().__init__(init_val)

        # df = pd.read_csv(filepath)
        # df["date"] = pd.to_datetime(df["date"], format='%m/%d/%y').dt.date
        # self.time_series_df = df

    def update_current_val(self, params, current_date: datetime.date) -> None:
        self.current_val = params.daily_vaccines_constant


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

    inf_risk_reduce = subpop_params.inf_induced_inf_risk_reduce
    if (np.asarray(inf_risk_reduce) != 1).all():
        proportional_risk_reduction = inf_risk_reduce / (1 - inf_risk_reduce)
    else:
        proportional_risk_reduction = 1

    return 1 + (proportional_risk_reduction *
                subpop_state.M)


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

    def update_current_val(self) -> None:
        pass

        # self.current_val = None


class FluSubpopModel(clt.SubpopModel):
    """
    Class for creating ImmunoSEIRS flu model with predetermined fixed
    structure -- initial values and epidemiological structure are
    populated by user-specified JSON files.

    Key method create_transmission_model returns a SubpopModel
    instance with S-E-I-H-R-D compartments and M
    and Mv epi metrics.
    
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
        - M is a InfInducedImmunity instance
        - Mv is a VaxInducedImmunity instance

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
                 name: str = ""):
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
        """

        # Assign config, params, and state to model-specific
        # types of dataclasses that have epidemiological parameter information
        # and sim state information

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

        dynamic_vals["beta_reduce"] = BetaReduce(init_val=0.0,
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
        schedules["daily_vaccines"] = DailyVaccines(filepath="")

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

        epi_metrics.M = \
            InfInducedImmunity(getattr(self.state, "M"),
                               transition_variables.R_to_S)

        epi_metrics.Mv = \
            VaxInducedImmunity(getattr(self.state, "Mv"))

        return epi_metrics


class FluMetapopModel(clt.MetapopModel, ABC):
    """
    MetapopModel-derived class specific to flu model.
    """

    def __init__(self,
                 subpop_models: list = [],
                 name: str = ""):

        super().__init__(subpop_models,
                         name)

        params_tensors = self.create_params_tensors(subpop_models)
        params_tensors.standardize_shapes()

        self.params_tensors = params_tensors

        self.state_tensors = FluMetapopStateTensors()


    def create_params_tensors(self,
                              subpop_models: list) -> FluMetapopParamsTensors:

        params_tensors = FluMetapopParamsTensors()

        for name in vars(subpop_models[0].params).keys():

            metapop_params = []

            for model in subpop_models:

                metapop_params.append(getattr(model.params, name))

            # If all values are scalars and equal to each other, then
            #   simply store a scalar (since its value is common
            #   across metapopulations)
            if all(isinstance(x, (int, float)) for x in metapop_params):
                first_val = metapop_params[0]
                if all(x == first_val for x in metapop_params):
                    metapop_params = first_val  # Collapse to scalar
                    setattr(params_tensors, name, torch.tensor(metapop_params))
            else:
                # torch `UserWarning`: creating a tensor from a list of numpy.ndarrays is extremely slow.
                #   Please consider converting the list to a single numpy.ndarray with numpy.array()
                #   before converting to a tensor.
                # Soooo we converted the list to an array!
                setattr(params_tensors, name, torch.tensor(np.asarray(metapop_params)))

        return params_tensors

    def update_state_tensors(self):

        for model in self.subpop_models:



    def apply_inter_subpop_updates(self):
        pass
