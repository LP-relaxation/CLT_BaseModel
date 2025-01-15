import datetime
import copy

import numpy as np
import pandas as pd
import sciris as sc

from dataclasses import dataclass
from typing import Optional, Union
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
    parameters in FluModel flu model. Along with FluSubpopState,
    is passed to get_current_rate and get_change_in_current_val.

    Assume that FluSubpopParams fields are constant or piecewise
    constant throughout the simulation. For variables that
    are more complicated and time-dependent, use a EpiMetric
    instead.

    Each field of datatype np.ndarray must be A x L,
    where A is the number of age groups and L is the number of
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
        immunity_hosp_increase_factor (positive float):
            factor by which population-level immunity
            against hospitalization grows after each
            case that recovers.
        immunity_inf_increase_factor (positive float):
            factor by which population-level immunity
            against infection grows after each case
                that recovers.
        immunity_saturation (np.ndarray of positive floats):
            constant(s) modeling saturation of antibody
            production of individuals.
        waning_factor_hosp (positive float):
            rate at which infection-induced immunity
            against hospitalization wanes.
        waning_factor_inf (positive float):
            rate at which infection-induced immunity
            against infection wanes.
        hosp_risk_reduction (positive float in [0,1]):
            reduction in hospitalization risk from
            infection-induced immunity.
        inf_risk_reduction (positive float in [0,1]):
            reduction in infection risk
            from infection-induced immunity.
        death_risk_reduction (positive float in [0,1]):
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
        viral_shedding_peak (positive float):
            the peak time of an individual's viral shedding.
        viral_shedding_magnitude (positive float):
            magnitude of the viral shedding.
        viral_shedding_duration (positive float):
            duration of the viral shedding,
            must be larger than viral_shedding_peak
        viral_shedding_feces_mass (positive float):
            average mass of feces (gram)
        relative_suscept_by_age (np.ndarray of positive floats in [0,1]):
            relative susceptibility to infection by age group
        time_away_prop_by_age (np.ndarray of positive floats in [0,1]):
            total proportion of time spent away from home by age group
        contact_reduction_travel (positive float in [0,1]):
            multiplier to reduce contact rate of traveling individuals
        contact_reduction_symp (positive float in [0,1]):
            multiplier to reduce contact rate of symptomatic individuals
    """

    num_age_groups: Optional[int] = None
    num_risk_groups: Optional[int] = None
    beta_baseline: Optional[float] = None
    total_pop_age_risk: Optional[np.ndarray] = None
    humidity_impact: Optional[float] = None
    immunity_hosp_increase_factor: Optional[float] = None
    immunity_inf_increase_factor: Optional[float] = None
    immunity_saturation: Optional[np.ndarray] = None
    waning_factor_hosp: Optional[float] = None
    waning_factor_inf: Optional[float] = None
    hosp_risk_reduction: Optional[float] = None
    inf_risk_reduction: Optional[float] = None
    death_risk_reduction: Optional[float] = None
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
    viral_shedding_peak: Optional[float] = None  # viral shedding parameters
    viral_shedding_magnitude: Optional[float] = None  # viral shedding parameters
    viral_shedding_duration: Optional[float] = None  # viral shedding parameters
    viral_shedding_feces_mass: Optional[float] = None  # viral shedding parameters
    relative_suscept_by_age: Optional[np.ndarray] = None,
    time_away_prop_by_age: Optional[np.ndarray] = None,
    contact_reduction_travel: Optional[float] = None
    contact_reduction_symp: Optional[float] = None


@dataclass
class FluSubpopState(clt.SubpopState):
    """
    Data container for pre-specified and fixed set of
    Compartment initial values and EpiMetric initial values
    in FluModel flu model.

    Each field below should be A x L np.ndarray, where
    A is the number of age groups and L is the number of risk groups.
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
            A x L x A x L array, where A is the number of age
            groups and L is the number of risk groups --
            element (a, l, a', l') corresponds to the number of
            contacts that a person in age-risk group a,l
            has with people in age-risk group a', l'.
        beta_reduct (float in [0,1]):
            starting value of DynamicVal "beta_reduct" on
            starting day of simulation -- this DynamicVal
            emulates a simple staged-alert policy
        wastewater (np.ndarray of positive floats):
            wastewater viral load
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


class SusceptibleToExposed(clt.TransitionVariable):
    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams):
        force_of_immunity = (1 + params.inf_risk_reduction * state.pop_immunity_inf)

        # We subtract absolute_humidity because higher humidity means less transmission
        beta_humidity_adjusted = (1 - state.absolute_humidity * params.humidity_impact) * \
                                 params.beta_baseline

        # Compute I / N -> original shape is (A, L)
        # Expand ratio for broadcasting -> new shape is (1, 1, A, L)
        I_N_ratio_expanded = ((
                                      state.IS + state.IP * params.IP_relative_inf + state.IA * params.IA_relative_inf)
                              / params.total_pop_age_risk)[None, None, :, :]

        # Expand force_of_immunity for broadcasting -> new shape is (A, L, 1, 1)
        force_of_immunity_expanded = force_of_immunity[:, :, None, None]

        # Element-wise multiplication and division by M_expanded
        # Sum over a' and l' (last two dimensions) -> result has shape (A, L)
        summand = np.sum(state.flu_contact_matrix * I_N_ratio_expanded / force_of_immunity_expanded, axis=(2, 3))

        return (1 - state.beta_reduct) * beta_humidity_adjusted * summand


class RecoveredToSusceptible(clt.TransitionVariable):
    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams):
        return np.full((params.num_age_groups, params.num_risk_groups), params.R_to_S_rate)


class ExposedToAsymp(clt.TransitionVariable):
    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams):
        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.E_to_I_rate * params.E_to_IA_prop)


class ExposedToPresymp(clt.TransitionVariable):
    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams):
        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.E_to_I_rate * (1 - params.E_to_IA_prop))


class PresympToSymp(clt.TransitionVariable):
    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams):
        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.IP_to_IS_rate)


class SympToRecovered(clt.TransitionVariable):
    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams):
        return np.full((params.num_age_groups, params.num_risk_groups),
                       (1 - params.IS_to_H_adjusted_prop) * params.IS_to_R_rate)


class AsympToRecovered(clt.TransitionVariable):
    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams):
        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.IA_to_R_rate)


class HospToRecovered(clt.TransitionVariable):
    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams):
        return np.full((params.num_age_groups, params.num_risk_groups),
                       (1 - params.H_to_D_adjusted_prop) * params.H_to_R_rate)


class SympToHosp(clt.TransitionVariable):
    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams):
        return np.asarray(params.IS_to_H_rate * params.IS_to_H_adjusted_prop /
                          (1 + params.hosp_risk_reduction * state.pop_immunity_hosp))


class HospToDead(clt.TransitionVariable):
    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams):
        return np.asarray(params.H_to_D_adjusted_prop * params.H_to_D_rate /
                          (1 + params.death_risk_reduction * state.pop_immunity_hosp))


class PopulationImmunityHosp(clt.EpiMetric):

    def __init__(self, init_val, R_to_S):
        super().__init__(init_val)
        self.R_to_S = R_to_S

    def get_change_in_current_val(self,
                                  state: FluSubpopState,
                                  params: FluSubpopParams,
                                  num_timesteps: int):
        pop_immunity_hosp = state.pop_immunity_hosp

        immunity_gain_numerator = params.immunity_hosp_increase_factor * self.R_to_S.current_val
        immunity_gain_denominator = params.total_pop_age_risk * \
                                    (1 + params.immunity_saturation * pop_immunity_hosp)

        immunity_gain = immunity_gain_numerator / immunity_gain_denominator
        immunity_loss = params.waning_factor_hosp * state.pop_immunity_hosp

        final_change = (immunity_gain - immunity_loss) / num_timesteps

        return np.asarray(final_change, dtype=np.float64)


class PopulationImmunityInf(clt.EpiMetric):
    def __init__(self, init_val, R_to_S):
        super().__init__(init_val)
        self.R_to_S = R_to_S

    def get_change_in_current_val(self,
                                  state: FluSubpopState,
                                  params: FluSubpopParams,
                                  num_timesteps: int):
        pop_immunity_inf = np.float64(state.pop_immunity_inf)

        immunity_gain_numerator = params.immunity_inf_increase_factor * self.R_to_S.current_val
        immunity_gain_denominator = params.total_pop_age_risk * \
                                    (1 + params.immunity_saturation * pop_immunity_inf)

        immunity_gain = immunity_gain_numerator / immunity_gain_denominator
        immunity_loss = params.waning_factor_inf * state.pop_immunity_inf

        final_change = (immunity_gain - immunity_loss) / num_timesteps

        return np.asarray(final_change, dtype=np.float64)


# test on the wastewater viral load simulation
class Wastewater(clt.EpiMetric):

    def __init__(self, init_val, S_to_E):
        super().__init__(init_val)
        self.S_to_E = S_to_E
        # preprocess
        self.flag_preprocessed = False
        self.viral_shedding = []
        self.viral_shedding_duration = None
        self.viral_shedding_magnitude = None
        self.viral_shedding_peak = None
        self.viral_shedding_feces_mass = None
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
        len_duration = self.viral_shedding_duration * self.num_timesteps

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
        self.viral_shedding_duration = copy.deepcopy(params.viral_shedding_duration)
        self.viral_shedding_magnitude = copy.deepcopy(params.viral_shedding_magnitude)
        self.viral_shedding_peak = copy.deepcopy(params.viral_shedding_peak)
        self.viral_shedding_feces_mass = copy.deepcopy(params.viral_shedding_feces_mass)
        self.num_timesteps = copy.deepcopy(num_timesteps)
        num_timesteps = np.float64(num_timesteps)
        self.viral_shedding = []
        # trapezoidal integral
        for time_idx in range(int(params.viral_shedding_duration * self.num_timesteps)):
            cur_time_point = time_idx / num_timesteps
            next_time_point = (time_idx + 1) / num_timesteps
            next_time_log_viral_shedding = params.viral_shedding_magnitude * next_time_point / \
                                           (params.viral_shedding_peak ** 2 + next_time_point ** 2)
            if time_idx == 0:
                interval_viral_shedding = params.viral_shedding_feces_mass * 0.5 * (
                        10 ** next_time_log_viral_shedding) / num_timesteps
            else:
                cur_time_log_viral_shedding = params.viral_shedding_magnitude * cur_time_point / \
                                              (params.viral_shedding_peak ** 2 + cur_time_point ** 2)
                interval_viral_shedding = params.viral_shedding_feces_mass * 0.5 \
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

    def clear_history(self) -> None:
        """
        Resets history_vals_list attribute to empty list.
        """
        self.flag_preprocessed = False
        self.viral_shedding = []
        self.viral_shedding_duration = None
        self.viral_shedding_magnitude = None
        self.viral_shedding_peak = None
        self.viral_shedding_feces_mass = None
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
        if np.sum(state.I) / np.sum(params.total_pop_age_risk) > 0.05:
            self.current_val = .5
            self.permanent_lockdown = True
        else:
            if not self.permanent_lockdown:
                self.current_val = 0.0


def absolute_humidity_func(current_date: datetime.date) -> float:
    """
    Note: this is a dummy function loosely based off of
    the absolute humidity data from Kaiming and Shraddha's
    new burden averted draft.

    TODO: replace this function with real humidity function

    The following calculation is used to achieve the correct
        upside-down parabola with the right min and max
        values and location
        max_value = 12.5
        0.00027 = (max_value - k) / ((0 - h) ** 2)

    Args:
        current_date (datetime.date):
            datetime.date object corresponding to
            real-world date

    Returns:
        float:
            nonnegative float between 3.4 and 12.5
            corresponding to absolute humidity
            that day of the year
    """

    # Convert datetime.date to integer between 1 and 365
    #   corresponding to day of the year
    day_of_year = current_date.timetuple().tm_yday

    # Minimum humidity occurs in January and December
    # Maximum humidity occurs in July
    return 12.5 - 0.00027 * (day_of_year % 365 - 180) ** 2


class AbsoluteHumidity(clt.Schedule):
    def update_current_val(self, current_date: datetime.date) -> None:
        self.current_val = absolute_humidity_func(current_date)


class FluContactMatrix(clt.Schedule):
    """
    Attributes:
        timeseries_df (pd.DataFrame):
            has a "date" column with strings in format "YYYY-MM-DD"
            of consecutive calendar days, and other columns
            named "is_school_day" (bool) and "is_work_day" (bool)
            corresponding to type of day.
        total_contact_matrix (np.ndarray):
            A x A np.ndarray, where A is the number of age groups

    See parent class docstring for other attributes.
    """

    def __init__(self,
                 init_val: Optional[Union[np.ndarray, float]] = None):
        super().__init__(init_val)

        df = pd.read_csv(base_path / "school_work_calendar.csv", index_col=0)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        self.time_series_df = df

        self.total_contact_matrix = np.array([[2.5, 0.5], [2, 1.5]]).reshape((2, 1, 2, 1))
        self.school_contact_matrix = np.array([[0.5, 0], [0.05, 0.1]]).reshape((2, 1, 2, 1))
        self.work_contact_matrix = np.array([[0, 0], [0, 0.0]]).reshape((2, 1, 2, 1))

    def update_current_val(self, current_date: datetime.date) -> None:
        """
        Subpopclasses must provide a concrete implementation of
        updating self.current_val in-place

        Args:
            current_date (datetime.date):
                real-world date corresponding to
                model's current simulation day
        """

        df = self.time_series_df

        try:
            current_row = df[df["date"] == current_date].iloc[0]
        except IndexError:
            print(f"Error: {current_date} is not in the Calendar's time_series_df.")

        self.current_val = self.total_contact_matrix - \
                           (1 - current_row["is_school_day"]) * self.school_contact_matrix - \
                           (1 - current_row["is_work_day"]) * self.work_contact_matrix


class InfectionForce(clt.InteractionTerm):

    def __init__(self,
                 subpop_name: str):
        self.subpop_name = subpop_name

    def update_current_val(self,
                           subpop_states_repo: clt.InterSubpopManager,
                           travel_proportions: pd.DataFrame,
                           subpop_params: clt.SubpopParams) -> None:

        pairwise_infection_forces = []

        self.current_val = np.sum(pairwise_infection_forces)


class FluSubpopModel(clt.SubpopModel):
    """
    Class for creating ImmunoSEIRS flu model with predetermined fixed
    structure -- initial values and epidemiological structure are
    populated by user-specified JSON files.

    Key method create_transmission_model returns a SubpopModel
    instance with S-E-I-H-R-D compartments and pop_immunity_inf
    and pop_immunity_hosp epi metrics. 
    
    The structure is as follows:
        - S = R_to_S - S_to_E
        - E = S_to_E - E_to_IP - E_to_IA
        - I = new_infected - IS_to_R - IS_to_H
        - H = IS_to_H - H_to_R - H_to_D
        - R = IS_to_R + H_to_R - R_to_S
        - D = H_to_D

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
                 state_dict: dict,
                 params_dict: dict,
                 config_dict: dict,
                 RNG: np.random.Generator,
                 wastewater_enabled: bool=False):
        """
        Args:
            state_dict (dict):
                holds current simulation state information,
                such as current values of epidemiological compartments
                and epi metrics -- keys and values respectively
                must match field names and format of FluSubpopState.
            params_dict (dict):
                holds epidemiological parameter values -- keys and
                values respectively must match field names and
                format of FluSubpopParams.
            config_dict (dict):
                holds configuration values -- keys and values
                respectively must match field names and format of
                Config.
            RNG (np.random.Generator):
                numpy random generator object used to obtain
                random numbers.
            wastewater_enabled (bool):
                if True, includes "wastewater" EpiMetric. Otherwise,
                excludes it.
        """

        # Assign config, params, and state to model-specific
        # types of dataclasses that have epidemiological parameter information
        # and sim state information

        self.wastewater_enabled = wastewater_enabled

        state = clt.make_dataclass_from_dict(FluSubpopState, state_dict)
        params = clt.make_dataclass_from_dict(FluSubpopParams, params_dict)
        config = clt.make_dataclass_from_dict(clt.Config, config_dict)

        # IMPORTANT NOTE: as always, we must be careful with mutable objects
        #   and generally use deep copies to avoid modification of the same
        #   object. But in this function call, using deep copies is unnecessary
        #   (redundant) because the parent class SubpopModel's __init__()
        #   creates deep copies.
        super().__init__(state, params, config, RNG)

        self.params.total_pop_age_risk = self.compute_total_pop_age_risk()

    def compute_total_pop_age_risk(self) -> np.ndarray:
        """

        Returns:
             total_pop_age_risk (np.ndarray):

        """

        total_pop_age_risk = np.zeros((self.params.num_age_groups,
                                       self.params.num_risk_groups))

        # At initialization (before simulation is run), each
        #   compartment's current val is equivalent to the initial val
        #   specified in the state variables' init val JSON.
        for compartment in self.compartments.values():
            total_pop_age_risk += compartment.current_val

        return total_pop_age_risk

    def create_interaction_terms(self) -> sc.objdict:

        if self.metapop_model:

            interaction_terms = sc.objdict()

            for name in self.metapop_model.subpop_model_dict.keys():
                interaction_terms[name] = InfectionForce(name)

            return interaction_terms

        else:

            return sc.objdict()

    def create_compartments(self) -> sc.objdict:
        """
        Create Compartment instances S-E-I-H-R-D (6 compartments total),
            save in sc.objdict, and return objdict
        """

        compartments = sc.objdict()

        for name in ("S", "E", "IP", "IS", "IA", "H", "R", "D"):
            compartments[name] = clt.Compartment(getattr(self.state, name))

        return compartments

    def create_dynamic_vals(self) -> sc.objdict:
        """
        Create all DynamicVal instances, save in sc.objdict, and return objdict
        """

        dynamic_vals = sc.objdict()

        dynamic_vals["beta_reduct"] = BetaReduct(init_val=0.0,
                                                 is_enabled=False)

        return dynamic_vals

    def create_schedules(self) -> sc.objdict():
        """
        Create all Schedule instances, save in sc.objdict, and return objdict
        """

        schedules = sc.objdict()

        schedules["absolute_humidity"] = AbsoluteHumidity()
        schedules["flu_contact_matrix"] = FluContactMatrix()

        return schedules

    def create_transition_variables(self) -> sc.objdict:
        """
        Create all TransitionVariable instances (7 transition variables total),
            save in sc.objdict, and return objdict
        """

        # NOTE: see the parent class SubpopModel's __init__() --
        #   create_transition_variables is called after
        #   self.config is assigned and after self.compartments
        #   has been created -- so these variables do exist
        # TODO: there is potentially a better way to design this
        #   (in SubpopModel) to be more EXPLICIT -- think about this...
        transition_type = self.config.transition_type
        compartments = self.compartments

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

    def create_transition_variable_groups(self) -> sc.objdict:
        """
        Create all transition variable groups described in docstring (2 transition
        variable groups total), save in sc.objdict, return
        """

        # Shortcuts for attribute access
        # NOTE: see the parent class SubpopModel's __init__() --
        #   create_transition_variable_groups is called after
        #   self.config is assigned and after
        #   self.compartments and self.transition_variables are created
        #   -- so these variables do exist
        # See similar NOTE in create_transition_variables function
        transition_type = self.config.transition_type
        compartments = self.compartments
        transition_variables = self.transition_variables

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

    def create_epi_metrics(self) -> sc.objdict:
        """
        Create all epi metric described in docstring (2 state
        variables total), save in sc.objdict, and return objdict
        """

        epi_metrics = sc.objdict()

        # Shortcuts for attribute access
        # NOTE: see the parent class SubpopModel's __init__() --
        #   create_epi_metrics is called after self.transition_variables
        #   are created -- so this variable exists
        # See similar NOTE in create_transition_variables
        #   and create_transition_variable_groups function
        transition_variables = self.transition_variables

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

    def run_model_checks(self):

        print(">>> Running flu model checks... \n")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        error_counter = 0

        state = self.state

        for name, val in vars(state).items():
            if isinstance(val, np.ndarray):
                flattened_array = val.flatten()
                for val in flattened_array:
                    if val < 0:
                        print(f"STOP! INPUT ERROR: {name} should not have negative values.")
                        error_counter += 1
            elif isinstance(val, float):
                if val < 0:
                    print(f"STOP! INPUT ERROR: {name} should not be negative.")
                    error_counter += 1

        compartment_population_sum = np.zeros((self.params.num_age_groups,
                                               self.params.num_risk_groups))

        for name, compartment in self.compartments.items():
            compartment_population_sum += compartment.current_val
            flattened_current_val = compartment.current_val.flatten()
            for val in flattened_current_val:
                if val != int(val):
                    print(f"STOP! INPUT ERROR: {name} should not have non-negative values.")
                    error_counter += 1

        if (compartment_population_sum != self.params.total_pop_age_risk).any():
            print(f"STOP! INPUT ERROR: sum of population in compartments must \n"
                  f"match specified total population value. Check \n"
                  f"\"total_pop_age_risk\" in model's \"params\" attribute \n"
                  f"and check compartments in state variables' init vals JSON.")
            error_counter += 1

        params = self.params

        # TODO: this has identical logic as the loop over state -- pull out as
        #   separate function
        for name, val in vars(params).items():
            if isinstance(val, np.ndarray):
                flattened_array = val.flatten()
                for val in flattened_array:
                    if val < 0:
                        print(f"STOP! INPUT ERROR: {name} should not have negative values.")
                        error_counter += 1
            elif isinstance(val, float):
                if val < 0:
                    print(f"STOP! INPUT ERROR: {name} should not be negative.")
                    error_counter += 1

        if error_counter == 0:

            print("OKAY! Flu model has passed input checks: \n"
                  "Compartment populations are nonnegative whole numbers \n"
                  "and add up to \"total_pop_age_risk\" in model's \n"
                  "\"params attribute.\" Fixed parameters are nonnegative.")

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
