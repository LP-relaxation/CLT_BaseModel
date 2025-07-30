import datetime
import copy

import numpy as np
import pandas as pd
import sciris as sc
from typing import Optional
from pathlib import Path
from abc import ABC, abstractmethod

import torch
import clt_base as clt

from .flu_data_structures import FluSubpopState, FluSubpopParams,\
    FluMetapopStateTensors, FluMetapopParamsTensors, FluPrecomputedTensors
from .flu_travel_functions import compute_travel_wtd_infectious


# Note: for dataclasses, Optional is used to help with static type checking
# -- it means that an attribute can either hold a value with the specified
# datatype or it can be None


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
    the rate depends on the `travel_wtd_infectious` attribute,
    which is a function of other subpopulations' states and
    parameters, and travel between subpopulations.

    If there is no metapopulation model, the rate
    is much simpler.

    Attributes:
        travel_wtd_infectious (np.ndarray of positive floats):
            weighted infectious count (exposure) from movement
            within home location, travel to other locations,
            and visitors from other locations

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

        self.travel_wtd_infectious = None

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:

        # If `travel_wtd_infectious` has not been updated,
        #   then there is no travel model -- so, simulate
        #   this subpopulation entirely independently and
        #   use the simplified transition rate that does not
        #   depend on travel dynamics

        beta_adjusted = compute_beta_adjusted(state, params)

        immune_force = compute_immunity_force(state, params)

        if self.travel_wtd_infectious is not None:
            # Need to convert tensor into array because combining np.ndarrays and
            #   tensors doesn't work, and everything else is an array
            return np.asarray(beta_adjusted * self.travel_wtd_infectious / immune_force)

        else:
            wtd_presymp_asymp = compute_wtd_presymp_asymp(state, params)

            return (beta_adjusted / immune_force) * \
                   np.matmul(state.flu_contact_matrix,
                             np.divide(state.IS + wtd_presymp_asymp, compute_pop_by_age(params)))


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
            A x 1 array -- where A is the number of age
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
        A x R array -- where A is the number of age groups
        and R is the number of risk groups -- representing
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


def compute_beta_adjusted(subpop_state: FluSubpopState,
                          subpop_params: FluSubpopParams) -> np.ndarray:
    """
    Computes humidity-adjusted beta
    """

    return subpop_params.beta_baseline * (1 + subpop_params.humidity_impact *
                                          np.exp(-180 * subpop_state.absolute_humidity))


def compute_pop_by_age(subpop_params: FluSubpopParams) -> np.ndarray:
    """
    Returns:
        np.ndarray:
            A x 1 array -- where A is the number of age groups --
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

        self.state_tensors = FluMetapopStateTensors()
        self.update_state_tensors()

        params_tensors = self.create_params_tensors()
        params_tensors.num_locations = torch.tensor(len(subpop_models))
        params_tensors.standardize_shapes()

        self.params_tensors = params_tensors

        self.precomputed = FluPrecomputedTensors(self.state_tensors,
                                                 self.params_tensors)

    def create_params_tensors(self) -> FluMetapopParamsTensors:

        # USE THE ORDERED DICTIONARY HERE FOR SAFETY!
        #   AGAIN, ORDER MATTERS BECAUSE ORDER DETERMINES
        #   THE SUBPOPULATION INDEX IN THE METAPOPULATION
        #   TENSOR!
        subpop_models = self._subpop_models_ordered

        params_tensors = FluMetapopParamsTensors()

        for name in vars(subpop_models[0].params).keys():

            metapop_vals = []

            for model in subpop_models.values():

                metapop_vals.append(getattr(model.params, name))

            # If all values are equal to each other, then
            #   simply store the first value (since its value is common
            #   across metapopulations)
            first_val = metapop_vals[0]
            if all(np.allclose(x, first_val) for x in metapop_vals):
                metapop_vals = first_val

            # Converting list of arrays to tensors is slow --
            #   better to convert to array first
            if isinstance(metapop_vals, list):
                metapop_vals = np.array(metapop_vals)

            setattr(params_tensors, name, torch.tensor(metapop_vals))

        return params_tensors

    def update_state_tensors(self) -> None:

        # ORDER MATTERS! USE ORDERED DICTIONARY HERE!
        #   See `create_params_tensors` for detailed note.
        subpop_models = self._subpop_models_ordered

        for name in vars(self.state_tensors).keys():

            # FluMetapopStateTensors has an attribute
            #   that is a dictionary called `init_vals` --
            #   disregard, as this only used to store
            #   initial values for resetting, but is not
            #   used in the travel model computation
            if name == "init_vals":
                continue

            metapop_vals = []

            for model in subpop_models.values():

                current_val = getattr(model.state, name)

                metapop_vals.append(current_val)

            setattr(self.state_tensors, name, torch.tensor(np.asarray(metapop_vals)))

    def apply_inter_subpop_updates(self) -> None:

        self.update_state_tensors()

        travel_wtd_infectious = compute_travel_wtd_infectious(self.state_tensors,
                                                              self.params_tensors,
                                                              self.precomputed)

        # Again, `self.subpop_models` is an ordered dictionary --
        #   so iterating over the dictionary like this is well-defined
        #   and responsible -- the order is important because it
        #   determines the order (index) in any metapopulation tensors
        subpop_models = self.subpop_models

        for i in range(len(subpop_models)):

            subpop_models.values()[i].transition_variables.S_to_E.travel_wtd_infectious = \
                np.squeeze(travel_wtd_infectious[i,:,:], axis=0)


