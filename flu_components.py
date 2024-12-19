import datetime
import pandas as pd
import copy

import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
import base_components as base

base_path = Path(__file__).parent / "flu_demo_input_files"


# Note: for dataclasses, Optional is used to help with static type checking
# -- it means that an attribute can either hold a value with the specified
# datatype or it can be None


@dataclass
class FluFixedParams(base.FixedParams):
    r"""
    Data container for pre-specified and fixed epidemiological
    parameters in FluModel flu model. Along with FluSimState,
    is passed to get_current_rate and get_change_in_current_val.

    Assume that FluFixedParams fields are constant or piecewise
    constant throughout the simulation. For variables that
    are more complicated and time-dependent, use a EpiMetric
    instead.

    Each field of datatype np.ndarray must be A x L,
    where A is the number of age groups and L is the number of
    risk groups. Note: this means all arrays should be 2D.
    See FluSimState docstring for important formatting note
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
        total_population_val (np.ndarray of positive ints):
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
            the peak time of an indiviudal's viral shedding.
        viral_shedding_magnitude (positive float):
            magnitude of the viral shedding.
        viral_shedding_duration (positive float):
            duration of the viral shedding, must be larger than viral_shedding_peak
        viral_shedding_feces_mass (positive float)
            average mass of feces (gram)
    """

    num_age_groups: Optional[int] = None
    num_risk_groups: Optional[int] = None
    beta_baseline: Optional[float] = None
    total_population_val: Optional[np.ndarray] = None
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
    viral_shedding_peak: Optional[float] = None # viral shedding parameters
    viral_shedding_magnitude: Optional[float] = None # viral shedding parameters
    viral_shedding_duration: Optional[float] = None # viral shedding parameters
    viral_shedding_feces_mass: Optional[float] = None # viral shedding parameters

@dataclass
class FluSimState(base.SimState):
    """
    Data container for pre-specified and fixed set of
    EpiCompartment initial values and EpiMetric initial values
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
            (holds current_val of EpiCompartment "S").
        E (np.ndarray of positive floats):
            exposed compartment for age-risk groups --
            (holds current_val of EpiCompartment "E").
        IP (np.ndarray of positive floats):
            infected pre-symptomatic compartment for age-risk groups
            (holds current_val of EpiCompartment "IP").
        IS (np.ndarray of positive floats):
            infected symptomatic compartment for age-risk groups
            (holds current_val of EpiCompartment "IS").
        IA (np.ndarray of positive floats):
            infected asymptomatic compartment for age-risk groups
            (holds current_val of EpiCompartment "IA").
        H (np.ndarray of positive floats):
            hospital compartment for age-risk groups
            (holds current_val of EpiCompartment "H").
        R (np.ndarray of positive floats):
            recovered compartment for age-risk groups
            (holds current_val of EpiCompartment "R").
        D (np.ndarray of positive floats):
            dead compartment for age-risk groups
            (holds current_val of EpiCompartment "D").
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
    wastewater: Optional[np.ndarray] = None # wastewater viral load

class SusceptibleToExposed(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        force_of_immunity = (1 + fixed_params.inf_risk_reduction * sim_state.pop_immunity_inf)

        # We subtract absolute_humidity because higher humidity means less transmission
        beta_humidity_adjusted = (1 - sim_state.absolute_humidity * fixed_params.humidity_impact) * \
                                 fixed_params.beta_baseline

        # Compute I / N -> original shape is (A, L)
        # Expand ratio for broadcasting -> new shape is (1, 1, A, L)
        I_N_ratio_expanded = ((
                                      sim_state.IS + sim_state.IP * fixed_params.IP_relative_inf + sim_state.IA * fixed_params.IA_relative_inf)
                              / fixed_params.total_population_val)[None, None, :, :]

        # Expand force_of_immunity for broadcasting -> new shape is (A, L, 1, 1)
        force_of_immunity_expanded = force_of_immunity[:, :, None, None]

        # Element-wise multiplication and division by M_expanded
        # Sum over a' and l' (last two dimensions) -> result has shape (A, L)
        summand = np.sum(sim_state.flu_contact_matrix * I_N_ratio_expanded / force_of_immunity_expanded, axis=(2, 3))

        return (1 - sim_state.beta_reduct) * beta_humidity_adjusted * summand


class RecoveredToSusceptible(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups), fixed_params.R_to_S_rate)


class ExposedToAsymp(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       fixed_params.E_to_I_rate * fixed_params.E_to_IA_prop)


class ExposedToPresymp(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       fixed_params.E_to_I_rate * (1 - fixed_params.E_to_IA_prop))


class PresympToSymp(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       fixed_params.IP_to_IS_rate)


class SympToRecovered(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       (1 - fixed_params.IS_to_H_adjusted_prop) * fixed_params.IS_to_R_rate)


class AsympToRecovered(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       fixed_params.IA_to_R_rate)


class HospToRecovered(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       (1 - fixed_params.H_to_D_adjusted_prop) * fixed_params.H_to_R_rate)


class SympToHosp(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.asarray(fixed_params.IS_to_H_rate * fixed_params.IS_to_H_adjusted_prop /
                          (1 + fixed_params.hosp_risk_reduction * sim_state.pop_immunity_hosp))


class HospToDead(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.asarray(fixed_params.H_to_D_adjusted_prop * fixed_params.H_to_D_rate /
                          (1 + fixed_params.death_risk_reduction * sim_state.pop_immunity_hosp))


class PopulationImmunityHosp(base.EpiMetric):

    def __init__(self, name, init_val, R_to_S):
        super().__init__(name, init_val)
        self.R_to_S = R_to_S

    def get_change_in_current_val(self,
                                  sim_state: FluSimState,
                                  fixed_params: FluFixedParams,
                                  num_timesteps: int):
        # Ensure consistent float64 precision
        factor = np.float64(fixed_params.immunity_hosp_increase_factor)
        susceptible = np.float64(self.R_to_S.current_val)
        population = np.float64(fixed_params.total_population_val)
        saturation = np.float64(fixed_params.immunity_saturation)
        pop_immunity = np.float64(sim_state.pop_immunity_hosp)
        waning_factor = np.float64(fixed_params.waning_factor_hosp)
        num_timesteps = np.float64(num_timesteps)

        # Break down calculations
        gain_numerator = factor * susceptible
        gain_denominator = population * (1 + saturation * pop_immunity)
        immunity_gain = gain_numerator / gain_denominator

        immunity_loss = waning_factor * pop_immunity

        # Final result
        result = (immunity_gain - immunity_loss) / num_timesteps

        return np.asarray(result, dtype=np.float64)


class PopulationImmunityInf(base.EpiMetric):
    def __init__(self, name, init_val, R_to_S):
        super().__init__(name, init_val)
        self.R_to_S = R_to_S

    def get_change_in_current_val(self,
                                  sim_state: FluSimState,
                                  fixed_params: FluFixedParams,
                                  num_timesteps: int):

        # Convert all parameters to consistent float64 for high precision
        increase_factor = np.float64(fixed_params.immunity_inf_increase_factor)
        R_to_S = np.float64(self.R_to_S.current_val)
        total_population = np.float64(fixed_params.total_population_val)
        saturation = np.float64(fixed_params.immunity_saturation)
        population_immunity = np.float64(sim_state.pop_immunity_inf)
        waning_factor = np.float64(fixed_params.waning_factor_inf)
        num_timesteps = np.float64(num_timesteps)

        # Break down calculations for better readability and to avoid compounded rounding errors
        gain_numerator = increase_factor * R_to_S
        gain_denominator = total_population * (1 + saturation * population_immunity)
        immunity_gain = gain_numerator / gain_denominator

        immunity_loss = waning_factor * population_immunity

        # Compute result with full precision
        result = (immunity_gain - immunity_loss) / num_timesteps

        # Ensure the result is a NumPy array
        return np.asarray(result, dtype=np.float64)


# test on the wastewater viral load simulation
class Wastewater(base.EpiMetric):
    def __init__(self, name, init_val, S_to_E):
        super().__init__(name, init_val)
        self.S_to_E = S_to_E
        # preprocess
        self.flag_preprocessed = False
        self.viral_shedding = []
        self.viral_shedding_duration = None
        self.viral_shedding_magnitude = None
        self.viral_shedding_peak = None
        self.viral_shedding_feces_mass = None
        self.S_to_E_len = 5000 # preset to match the simulation time horizon
        self.S_to_E_history = np.zeros(self.S_to_E_len)
        self.cur_time_stamp = -1
        self.num_timesteps = None
        self.val_list_len = 10
        self.current_val_list = np.zeros(self.val_list_len)
        self.cur_idx_timestep = -1

    def get_change_in_current_val(self,
                                  sim_state: FluSimState,
                                  fixed_params: FluFixedParams,
                                  num_timesteps: int):
        if not self.flag_preprocessed: # preprocess the viral shedding function if not done yet
            self.preprocess(fixed_params, num_timesteps)
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

        # discrete convolution
        len_duration = self.viral_shedding_duration * self.num_timesteps
        if self.cur_time_stamp >= len_duration - 1:
            current_val = self.S_to_E_history[(self.cur_time_stamp - len_duration + 1):(self.cur_time_stamp + 1)] @ self.viral_shedding
        else:
            current_val = self.S_to_E_history[:(self.cur_time_stamp + 1)] @ self.viral_shedding[-(self.cur_time_stamp + 1):]

        self.current_val = current_val
        self.cur_idx_timestep += 1
        self.current_val_list[self.cur_idx_timestep] = current_val

    def preprocess(self,
                   fixed_params: FluFixedParams,
                   num_timesteps: int):
        # store the parameters locally
        self.viral_shedding_duration = copy.deepcopy(fixed_params.viral_shedding_duration)
        self.viral_shedding_magnitude = copy.deepcopy(fixed_params.viral_shedding_magnitude)
        self.viral_shedding_peak = copy.deepcopy(fixed_params.viral_shedding_peak)
        self.viral_shedding_feces_mass = copy.deepcopy(fixed_params.viral_shedding_feces_mass)
        self.num_timesteps = copy.deepcopy(num_timesteps)
        num_timesteps = np.float64(num_timesteps)
        self.viral_shedding = []
        # trapezoidal integral
        for time_idx in range(int(fixed_params.viral_shedding_duration * self.num_timesteps)):
            cur_time_point = time_idx / num_timesteps
            next_time_point = (time_idx + 1) / num_timesteps
            next_time_log_viral_shedding = fixed_params.viral_shedding_magnitude * next_time_point /\
                                          (fixed_params.viral_shedding_peak ** 2 + next_time_point ** 2 )
            if time_idx == 0:
                interval_viral_shedding = fixed_params.viral_shedding_feces_mass * 0.5 * (10 ** next_time_log_viral_shedding) / num_timesteps
            else:
                cur_time_log_viral_shedding = fixed_params.viral_shedding_magnitude * cur_time_point /\
                                          (fixed_params.viral_shedding_peak ** 2 + cur_time_point ** 2 )
                interval_viral_shedding = fixed_params.viral_shedding_feces_mass * 0.5 \
                                 * (10 ** cur_time_log_viral_shedding + 10 ** next_time_log_viral_shedding) / num_timesteps
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

        self.history_vals_list = []
        self.viral_shedding_daily = []
        self.S_to_E_history = np.zeros(self.S_to_E_len)
        self.cur_time_stamp = -1
        self.flag_preprocessed = False
        self.current_val_list = np.zeros(self.val_list_len)

class BetaReduct(base.DynamicVal):

    def __init__(self, name, init_val, is_enabled):
        super().__init__(name, init_val, is_enabled)
        self.permanent_lockdown = False

    def update_current_val(self, sim_state, fixed_params):
        if np.sum(sim_state.I) / np.sum(fixed_params.total_population_val) > 0.05:
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


class AbsoluteHumidity(base.Schedule):
    def update_current_val(self, current_date: datetime.date) -> None:
        self.current_val = absolute_humidity_func(current_date)


class FluContactMatrix(base.Schedule):
    """

    Attributes:
        timeseries_df (pd.DataFrame):
            has a "date" column with strings in format "YYYY-MM-DD"
            of consecutive calendar days, and other columns
            named "is_school_day" (bool) and "is_work_day" (bool)
            corresponding to type of day.
        total_contact_matrix (np.ndarray):
            (A x L) x (A x L) np.ndarray, where A is the number
            of age groups and L is the number of risk groups.

    See parent class docstring for other attributes.
    """

    def __init__(self,
                 name: str,
                 init_val: Optional[Union[np.ndarray, float]] = None):
        super().__init__(name, init_val)

        df = pd.read_csv(base_path / "school_work_calendar.csv", index_col=0)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        self.time_series_df = df

        self.total_contact_matrix = np.array([[2.5, 0.5], [2, 1.5]]).reshape((2, 1, 2, 1))
        self.school_contact_matrix = np.array([[0.5, 0], [0.05, 0.1]]).reshape((2, 1, 2, 1))
        self.work_contact_matrix = np.array([[0, 0], [0, 0.0]]).reshape((2, 1, 2, 1))

    def update_current_val(self, current_date: datetime.date) -> None:
        """
        Subclasses must provide a concrete implementation of
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


class FluModelConstructor(base.ModelConstructor):
    """
    Class for creating ImmunoSEIRS flu model with predetermined fixed
    structure -- initial values and epidemiological structure are
    populated by user-specified JSON files.

    Key method create_transmission_model returns a TransmissionModel
    instance with S-E-I-H-R-D compartments and pop_immunity_inf
    and pop_immunity_hosp epi metrics. 
    
    The structure
    is as follows:
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

    Attributes:
        config (Config):
            holds configuration values.
        fixed_params (FluFixedParams):
            holds epidemiological parameter values, read-in
            from user-specified JSON.
        sim_state (FluSimState):
            holds current simulation state information,
            such as current values of epidemiological compartments
            and epi metrics, read in from user-specified JSON.
        transition_variable_lookup (dict):
            maps string to corresponding TransitionVariable.
        transition_variable_group_lookup (dict):
            maps string to corresponding TransitionVariableGroup.
        compartment_lookup (dict):
            maps string to corresponding EpiCompartment,
            using the value of the EpiCompartment's "name" attribute.
        epi_metric_lookup (dict):
            maps string to corresponding EpiMetric,
            using the value of the EpiMetric's "name" attribute.
    """

    def __init__(self,
                 config_filepath: Optional[str] = None,
                 fixed_params_filepath: Optional[str] = None,
                 state_vars_init_vals_filepath: Optional[str] = None):
        """
        Create Config, FluFixedParams, and FluSimState instances
        using values from respective JSON files, and save these instances
        on the FluModelConstructor to construct a model.

        If any filepath is not specified, then user must manually assign
        the respective attribute (config, fixed_params, or sim_state)
        before using constructor to create a model.

        Attributes:
            config_filepath (Optional[str]):
                path to config JSON file (path includes actual filename
                with suffix ".json") -- all JSON fields must match
                name and datatype of Config instance attributes.
            fixed_params_filepath (Optional[str]):
                path to epidemiological parameters JSON file
                (path includes actual filename with suffix ".json")
                -- all JSON fields must match name and datatype of
                FixedParams instance attributes.
            state_vars_init_vals_filepath (Optional[str]):
                path to epidemiological compartments JSON file
                (path includes actual filename with suffix ".json")
                -- all JSON fields must match name and datatype of
                StateVariable instance attributes -- these initial
                values are used to populate sim_state attribute.
        """

        # Use same init method as abstract class --
        # creates "lookup" attributes (dictionaries for easy access)
        # and creates attributes config, fixed_params, and sim_state
        super().__init__()

        # Assign config, fixed_params, and sim_state to model-specific
        # types of dataclasses that have epidemiological parameter information
        # and sim state information
        if config_filepath:
            self.config = self.dataclass_instance_from_json(base.Config,
                                                            config_filepath)

        if fixed_params_filepath:
            self.fixed_params = self.dataclass_instance_from_json(FluFixedParams,
                                                                  fixed_params_filepath)

        if state_vars_init_vals_filepath:
            self.sim_state = \
                self.dataclass_instance_from_json(FluSimState,
                                                  state_vars_init_vals_filepath)

    def setup_epi_compartments(self) -> None:
        """
        Create EpiCompartment instances S-E-I-H-R-D (6 compartments total)
        and add them to compartment_lookup for dictionary access
        """

        for name in ("S", "E", "IP", "IS", "IA", "H", "R", "D"):
            self.compartment_lookup[name] = base.EpiCompartment(name, getattr(self.sim_state, name))

    def setup_dynamic_vals(self) -> None:
        """
        Create all DynamicVal instances and add them to dynamic_val_lookup attribute
            for dictionary access
        """

        self.dynamic_val_lookup["beta_reduct"] = BetaReduct(name="beta_reduct",
                                                            init_val=0.0,
                                                            is_enabled=False)

    def setup_schedules(self) -> None:
        """
        Create all Schedule instances and add them to schedule_lookup attribute
            for dictionary access
        """

        self.schedule_lookup["absolute_humidity"] = AbsoluteHumidity("absolute_humidity")
        self.schedule_lookup["flu_contact_matrix"] = FluContactMatrix("flu_contact_matrix")

    def setup_transition_variables(self) -> None:
        """
        Create all TransitionVariable instances (7 transition variables total)
            and add them to transition_variable_lookup attribute
            for dictionary access
        """

        compartments = self.compartment_lookup
        transition_type = self.config.transition_type

        # Reordering the tuples to put the transition function first
        transition_mapping = {

            "R_to_S": (RecoveredToSusceptible, "R_to_S", compartments["R"], compartments["S"]),

            "S_to_E": (SusceptibleToExposed, "S_to_E", compartments["S"], compartments["E"]),

            "IP_to_IS": (PresympToSymp,
                         "IP_to_IS", compartments["IP"], compartments["IS"]),

            "IA_to_R": (AsympToRecovered,
                        "IA_to_R", compartments["IA"], compartments["R"]),

            "E_to_IP": (ExposedToPresymp,
                        "E_to_IP", compartments["E"], compartments["IP"], True),

            "E_to_IA": (ExposedToAsymp,
                        "E_to_IA", compartments["E"], compartments["IA"], True),

            "IS_to_R": (SympToRecovered,
                        "IS_to_R", compartments["IS"], compartments["R"], True),

            "IS_to_H": (SympToHosp, "IS_to_H", compartments["IS"], compartments["H"], True),

            "H_to_R": (HospToRecovered,
                       "H_to_R", compartments["H"], compartments["R"], True),

            "H_to_D": (HospToDead, "H_to_D", compartments["H"], compartments["D"], True)
        }

        # Create transition variables dynamically
        # params[0] is the TransitionVariable subclass (e.g. RecoveredToSusceptible)
        # params[1:4] refers to the name, origin compartment, destination compartment list
        # params[4:] contains the Boolean indicating if the transition variable is jointly
        #   distributed (True if jointly distributed)
        self.transition_variable_lookup = {
            name: params[0](*params[1:4], transition_type, *params[4:])
            for name, params in transition_mapping.items()
        }

    def setup_transition_variable_groups(self) -> None:
        """
        Create all transition variable groups described in docstring (2 transition
        variable groups total) and add them to transition_variable_group_lookup attribute
        for dictionary access
        """

        # Shortcuts for attribute access
        compartment_lookup = self.compartment_lookup
        tvar_lookup = self.transition_variable_lookup
        transition_type = self.config.transition_type

        self.transition_variable_group_lookup = {
            "E_out": base.TransitionVariableGroup("E_out",
                                                  compartment_lookup["E"],
                                                  transition_type,
                                                  (tvar_lookup["E_to_IP"],
                                                   tvar_lookup["E_to_IA"])),

            "IS_out": base.TransitionVariableGroup("IS_out",
                                                   compartment_lookup["IS"],
                                                   transition_type,
                                                   (tvar_lookup["IS_to_R"],
                                                    tvar_lookup["IS_to_H"])),

            "H_out": base.TransitionVariableGroup("H_out",
                                                  compartment_lookup["H"],
                                                  transition_type,
                                                  (tvar_lookup["H_to_R"],
                                                   tvar_lookup["H_to_D"]))
        }

    def setup_epi_metrics(self) -> None:
        """
        Create all epi metric described in docstring (2 state
        variables total) and add them to epi_metric_lookup attribute
        for dictionary access
        """

        self.epi_metric_lookup["pop_immunity_inf"] = \
            PopulationImmunityInf("pop_immunity_inf",
                                  getattr(self.sim_state, "pop_immunity_inf"),
                                  self.transition_variable_lookup["R_to_S"])

        self.epi_metric_lookup["pop_immunity_hosp"] = \
            PopulationImmunityHosp("pop_immunity_hosp",
                                   getattr(self.sim_state, "pop_immunity_hosp"),
                                   self.transition_variable_lookup["R_to_S"])


        # test on the wastewater
        self.epi_metric_lookup["wastewater"] = \
            Wastewater("wastewater",
                       getattr(self.sim_state, "wastewater"), # initial value is set to null for now
                       self.transition_variable_lookup["S_to_E"])