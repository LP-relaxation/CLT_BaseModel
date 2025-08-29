import datetime
import copy

import numpy as np
import pandas as pd
import sciris as sc
from typing import Optional
from abc import ABC

from functools import reduce

import torch
import clt_toolkit as clt

from dataclasses import fields, asdict
from .flu_travel_functions import compute_total_mixing_exposure
from .flu_data_structures import FluSubpopState, FluSubpopParams, \
    FluTravelStateTensors, FluTravelParamsTensors, \
    FluFullMetapopStateTensors, FluFullMetapopParamsTensors, \
    FluMixingParams, FluPrecomputedTensors, FluFullMetapopScheduleTensors, \
    FluSubpopSchedules


class FluSubpopModelError(clt.SubpopModelError):
    """Custom exceptions for flu subpopulation simulation model errors."""
    pass


class FluMetapopModelError(clt.MetapopModelError):
    """Custom exceptions for flu metapopulation simulation model errors."""
    pass


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
    the rate depends on the `total_mixing_exposure` attribute,
    which is a function of other subpopulations' states and
    parameters, and travel between subpopulations.

    If there is no metapopulation model, the rate
    is much simpler.

    Attributes:
        total_mixing_exposure (np.ndarray of positive floats):
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

        self.total_mixing_exposure = None

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """

        # If `total_mixing_exposure` has not been updated,
        #   then there is no travel model -- so, simulate
        #   this subpopulation entirely independently and
        #   use the simplified transition rate that does not
        #   depend on travel dynamics

        beta_adjusted = compute_beta_adjusted(state, params)

        inf_induced_inf_risk_reduce = params.inf_induced_inf_risk_reduce
        inf_induced_proportional_risk_reduce = inf_induced_inf_risk_reduce / (1 - inf_induced_inf_risk_reduce)

        vax_induced_inf_risk_reduce = params.vax_induced_inf_risk_reduce
        vax_induced_proportional_risk_reduce = vax_induced_inf_risk_reduce / (1 - vax_induced_inf_risk_reduce)

        immune_force = (1 + inf_induced_proportional_risk_reduce * state.M +
                        vax_induced_proportional_risk_reduce * state.MV)

        if self.total_mixing_exposure is not None:

            # Note here `self.total_mixing_exposure` includes
            #   `suscept_by_age` -- see `compute_total_mixing_exposure_prop`
            #   in `flu_travel_functions`

            # Need to convert tensor into array because combining np.ndarrays and
            #   tensors doesn't work, and everything else is an array
            return np.asarray(beta_adjusted * self.total_mixing_exposure / immune_force)

        else:
            wtd_presymp_asymp_by_age = compute_wtd_presymp_asymp_by_age(state, params)

            # Super confusing syntax... but this is the pain of having A x R,
            #   but having the contact matrix (contact patterns) be for
            #   ONLY age groups
            wtd_infectious_prop = np.divide(np.sum(state.IS, axis=1, keepdims=True) + wtd_presymp_asymp_by_age,
                                            compute_pop_by_age(params))

            raw_total_exposure = np.matmul(state.flu_contact_matrix, wtd_infectious_prop)

            # The total rate is only age-dependent -- it's the same rate across age groups
            return params.relative_suscept * (beta_adjusted * raw_total_exposure / immune_force)


class RecoveredToSusceptible(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "R" to "S" compartment. The functional form is the same across
    subpopulations.
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """
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
        """
        Returns:
            np.ndarray of shape (A, R)
        """
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
        """
        Returns:
            np.ndarray of shape (A, R)
        """

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
        """
        Returns:
            np.ndarray of shape (A, R)
        """
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
        """
        Returns:
            np.ndarray of shape (A, R)
        """
        inf_induced_hosp_risk_reduce = params.inf_induced_hosp_risk_reduce
        inf_induced_proportional_risk_reduce = inf_induced_hosp_risk_reduce / (1 - inf_induced_hosp_risk_reduce)

        vax_induced_hosp_risk_reduce = params.vax_induced_hosp_risk_reduce
        vax_induced_proportional_risk_reduce = vax_induced_hosp_risk_reduce / (1 - vax_induced_hosp_risk_reduce)

        immunity_force = (1 + inf_induced_proportional_risk_reduce * state.M +
                          vax_induced_proportional_risk_reduce * state.MV)

        return np.asarray((1 - params.IS_to_H_adjusted_prop / immunity_force) * params.IS_to_R_rate)


class AsympToRecovered(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "IA" to "R" compartment. The functional form is the same across
    subpopulations.
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """

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
        """
        Returns:
            np.ndarray of shape (A, R)
        """

        inf_induced_death_risk_reduce = params.inf_induced_death_risk_reduce
        vax_induced_death_risk_reduce = params.vax_induced_death_risk_reduce

        inf_induced_proportional_risk_reduce = \
            inf_induced_death_risk_reduce / (1 - inf_induced_death_risk_reduce)

        vax_induced_proportional_risk_reduce = \
            vax_induced_death_risk_reduce / (1 - vax_induced_death_risk_reduce)

        immunity_force = (1 + inf_induced_proportional_risk_reduce * state.M +
                          vax_induced_proportional_risk_reduce * state.MV)

        return np.full((params.num_age_groups, params.num_risk_groups),
                       (1 - params.H_to_D_adjusted_prop / immunity_force) * params.H_to_R_rate)


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
        """
        Returns:
            np.ndarray of shape (A, R)
        """

        inf_induced_hosp_risk_reduce = params.inf_induced_hosp_risk_reduce
        inf_induced_proportional_risk_reduce = inf_induced_hosp_risk_reduce / (1 - inf_induced_hosp_risk_reduce)

        vax_induced_hosp_risk_reduce = params.vax_induced_hosp_risk_reduce
        vax_induced_proportional_risk_reduce = vax_induced_hosp_risk_reduce / (1 - vax_induced_hosp_risk_reduce)

        immunity_force = (1 + inf_induced_proportional_risk_reduce * state.M +
                          vax_induced_proportional_risk_reduce * state.MV)

        return np.asarray(params.IS_to_H_rate * params.IS_to_H_adjusted_prop / immunity_force)


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
        """
        Returns:
            np.ndarray of shape (A, R)
        """

        inf_induced_death_risk_reduce = params.inf_induced_death_risk_reduce
        vax_induced_death_risk_reduce = params.vax_induced_death_risk_reduce

        inf_induced_proportional_risk_reduce = \
            inf_induced_death_risk_reduce / (1 - inf_induced_death_risk_reduce)

        vax_induced_proportional_risk_reduce = \
            vax_induced_death_risk_reduce / (1 - vax_induced_death_risk_reduce)

        immunity_force = (1 + inf_induced_proportional_risk_reduce * state.M +
                          vax_induced_proportional_risk_reduce * state.MV)

        return np.asarray(params.H_to_D_adjusted_prop * params.H_to_D_rate / immunity_force)


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
                                  num_timesteps: int) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """

        # Note: the current values of transition variables already include
        #   discretization (division by the number of timesteps) -- therefore,
        #   we do not divide the first part of this equation by the number of
        #   timesteps -- see `TransitionVariable` class's methods for getting
        #   various realizations for more information

        return (self.R_to_S.current_val / params.total_pop_age_risk) * \
               (1 - params.inf_induced_saturation * state.M - params.vax_induced_saturation * state.MV) - \
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
        """
        Returns:
            np.ndarray of shape (A, R)
        """

        # Note: `state.daily_vaccines` (based on the value of the `DailyVaccines`
        #   `Schedule` is NOT divided by the number of timesteps -- so we need to
        #   do this division in the equation here.

        return state.daily_vaccines / (params.total_pop_age_risk * num_timesteps) - \
               params.vax_induced_immune_wane * state.MV / num_timesteps


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
                 timeseries_df: pd.DataFrame = None):
        """
        Args:
            init_val (Optional[np.ndarray | float]):
                starting value(s) at the beginning of the simulation
            timeseries_df (Optional[pd.DataFrame] = None):
                must have "date" and "daily_vaccines" -- "date" entries must
                correspond to consecutive calendar days and must either
                be strings with `"YYYY-MM-DD"` format or `datetime.date`
                objects -- "value" entries correspond to historical
                number vaccinated on those days. Identical to
                `FluSubpopSchedules` field of same name.
        """

        super().__init__(init_val)

        self.timeseries_df = timeseries_df

    def update_current_val(self, params, current_date: datetime.date) -> None:
        self.current_val = self.timeseries_df.loc[
            self.timeseries_df["date"] == current_date, "daily_vaccines"].values[0]


class AbsoluteHumidity(clt.Schedule):

    def __init__(self,
                 init_val: Optional[np.ndarray | float] = None,
                 timeseries_df: pd.DataFrame = None):
        """
        Args:
            init_val (Optional[np.ndarray | float]):
                starting value(s) at the beginning of the simulation
            timeseries_df (Optional[pd.DataFrame] = None):
                must have columns "date" and "absolute_humidity" --
                "date" entries must correspond to consecutive calendar days
                and must either be strings with `"YYYY-MM-DD"` format or
                `datetime.date` objects -- "value" entries correspond to
                absolute humidity on those days. Identical to
                `FluSubpopSchedules` field of same name.
        """

        super().__init__(init_val)

        self.timeseries_df = timeseries_df

    def update_current_val(self, params, current_date: datetime.date) -> None:
        self.current_val = self.timeseries_df.loc[
            self.timeseries_df["date"] == current_date, "absolute_humidity"].values[0]


class FluContactMatrix(clt.Schedule):
    """
    Flu contact matrix.

    Attributes:
        timeseries_df (pd.DataFrame):
            must have columns "date", "is_school_day", and "is_work_day"
            -- "date" entries must correspond to consecutive calendar
            days and must either be strings with `"YYYY-MM-DD"` format
            or `datetime.date` object and "is_school_day" and
            "is_work_day" entries are Booleans indicating if that date is
            a school day or work day. Identical to `FluSubpopSchedules` field
            of same name.

    See parent class docstring for other attributes.
    """

    def __init__(self,
                 init_val: Optional[np.ndarray | float] = None,
                 timeseries_df: pd.DataFrame = None):

        super().__init__(init_val)

        self.timeseries_df = timeseries_df

    def update_current_val(self,
                           subpop_params: FluSubpopParams,
                           current_date: datetime.date) -> None:

        df = self.timeseries_df

        try:
            current_row = df[df["date"] == current_date].iloc[0]
            self.current_val = subpop_params.total_contact_matrix - \
                               (1 - current_row["is_school_day"]) * subpop_params.school_contact_matrix - \
                               (1 - current_row["is_work_day"]) * subpop_params.work_contact_matrix
        except IndexError:
            # print(f"Error: {current_date} is not in `timeseries_df`. Using total contact matrix.")
            self.current_val = subpop_params.total_contact_matrix


def compute_wtd_presymp_asymp_by_age(subpop_state: FluSubpopState,
                                     subpop_params: FluSubpopParams) -> np.ndarray:
    """
    Returns weighted sum of IP and IA compartment for
        subpopulation with given state and parameters.
        IP and IA are weighted by their relative infectiousness
        respectively, and then summed over risk groups.

    Returns:
        np.ndarray of shape (A, R)
    """

    # sum over risk groups
    wtd_IP = \
        subpop_params.IP_relative_inf * np.sum(subpop_state.IP, axis=1, keepdims=True)
    wtd_IA = \
        subpop_params.IA_relative_inf * np.sum(subpop_state.IA, axis=1, keepdims=True)

    return wtd_IP + wtd_IA


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


class FluSubpopModel(clt.SubpopModel):
    """
    Class for creating ImmunoSEIRS flu model with predetermined fixed
    structure -- initial values and epidemiological structure are
    populated by user-specified `JSON` files.

    Key method create_transmission_model returns a `SubpopModel`
    instance with S-E-I-H-R-D compartments and M
    and MV epi metrics.
    
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
        - MV is a VaxInducedImmunity instance

    Transition rates and update formulas are specified in
    corresponding classes.

    See parent class `SubpopModel`'s docstring for additional attributes.
    """

    def __init__(self,
                 state: FluSubpopState,
                 params: FluSubpopParams,
                 simulation_settings: FluSubpopSchedules,
                 RNG: np.random.Generator,
                 schedules_spec: FluSubpopSchedules,
                 name: str):
        """
        Args:
            state (FluSubpopState):
                holds current simulation state information,
                such as current values of epidemiological compartments
                and epi metrics.
            params (FluSubpopParams):
                holds epidemiological parameter values.
            simulation_settings (SimulationSettings):
                holds simulation settings.
            RNG (np.random.Generator):
                numpy random generator object used to obtain
                random numbers.
            schedules_spec (FluSubpopSchedules):
                holds dataframes that specify `Schedule` instances.
            name (str):
                unique name of MetapopModel instance.
        """

        self.schedules_spec = schedules_spec

        # IMPORTANT NOTE: as always, we must be careful with mutable objects
        # and generally use deep copies to avoid modification of the same
        # object. But in this function call, using deep copies is unnecessary
        # (redundant) because the parent class `SubpopModel`'s `__init__`
        # creates deep copies.
        super().__init__(state, params, simulation_settings, RNG, name)

    def create_compartments(self) -> sc.objdict[str, clt.Compartment]:

        # Create `Compartment` instances S-E-IA-IP-IS-H-R-D (7 compartments total)
        # Save instances in `sc.objdict` and return objdict

        compartments = sc.objdict()

        for name in ("S", "E", "IP", "IS", "IA", "H", "R", "D"):
            compartments[name] = clt.Compartment(getattr(self.state, name))

        return compartments

    def create_dynamic_vals(self) -> sc.objdict[str, clt.DynamicVal]:
        """
        Create all `DynamicVal` instances, save in `sc.objdict`, and return objdict
        """

        dynamic_vals = sc.objdict()

        dynamic_vals["beta_reduce"] = BetaReduce(init_val=0.0,
                                                 is_enabled=False)

        return dynamic_vals

    def create_schedules(self) -> sc.objdict[str, clt.Schedule]:
        """
        Create all `Schedule` instances, save in `sc.objdict`, and return objdict
        """

        schedules = sc.objdict()

        schedules["absolute_humidity"] = AbsoluteHumidity()
        schedules["flu_contact_matrix"] = FluContactMatrix()
        schedules["daily_vaccines"] = DailyVaccines()

        for field, df in asdict(self.schedules_spec).items():

            try:
                df["date"] = pd.to_datetime(df["date"], format='%Y-%m-%d').dt.date
            except ValueError as e:
                raise ValueError("Error: dates should be strings in YYYY-MM-DD format or "
                                 "`date.datetime` objects.") from e

            schedules[field].timeseries_df = df

        return schedules

    def create_transition_variables(self) -> sc.objdict[str, clt.TransitionVariable]:
        """
        Create all `TransitionVariable` instances,
        save in `sc.objdict`, and return objdict
        """

        # NOTE: see the parent class `SubpopModel`'s `__init__` --
        # `create_transition_variables` is called after
        # `simulation_settings` is assigned

        transition_type = self.simulation_settings.transition_type

        transition_variables = sc.objdict()

        S = self.compartments.S
        E = self.compartments.E
        IP = self.compartments.IP
        IS = self.compartments.IS
        IA = self.compartments.IA
        H = self.compartments.H
        R = self.compartments.R
        D = self.compartments.D

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

    def create_transition_variable_groups(self) -> sc.objdict[str, clt.TransitionVariableGroup]:
        """
        Create all transition variable groups described in docstring (2 transition
        variable groups total), save in `sc.objdict`, return objdict
        """

        # Shortcuts for attribute access
        # NOTE: see the parent class `SubpopModel`'s `__init__` --
        # `create_transition_variable_groups` is called after
        # `simulation_settings` is assigned

        transition_type = self.simulation_settings.transition_type

        transition_variable_groups = sc.objdict()

        transition_variable_groups.E_out = clt.TransitionVariableGroup(self.compartments.E,
                                                                       transition_type,
                                                                       (self.transition_variables.E_to_IP,
                                                                        self.transition_variables.E_to_IA))

        transition_variable_groups.IS_out = clt.TransitionVariableGroup(self.compartments.IS,
                                                                        transition_type,
                                                                        (self.transition_variables.IS_to_R,
                                                                         self.transition_variables.IS_to_H))

        transition_variable_groups.H_out = clt.TransitionVariableGroup(self.compartments.H,
                                                                       transition_type,
                                                                       (self.transition_variables.H_to_R,
                                                                        self.transition_variables.H_to_D))

        return transition_variable_groups

    def create_epi_metrics(self) -> sc.objdict[str, clt.EpiMetric]:
        """
        Create all epi metric described in docstring (2 state
        variables total), save in `sc.objdict`, and return objdict
        """

        epi_metrics = sc.objdict()

        epi_metrics.M = \
            InfInducedImmunity(getattr(self.state, "M"),
                               self.transition_variables.R_to_S)

        epi_metrics.MV = \
            VaxInducedImmunity(getattr(self.state, "MV"))

        return epi_metrics

    def modify_subpop_params(self,
                             updates_dict: dict):
        """
        This method lets users safely modify a single subpopulation
        parameters field; if this subpop model is associated with
        a metapop model, the metapopulation-wide tensors are updated
        automatically afterward. See also `modify_subpop_params` method on
        `FluMetapopModel`.

        Parameters:
            updates_dict (dict):
                Dictionary specifying values to update in a
                `FluSubpopParams` instance -- keys must match the
                field names of `FluSubpopParams`.
        """

        # If associated with metapop model, run this method
        #   on the metapop model itself to handle metapopulation-wide
        #   tensor updating
        if self.metapop_model:
            self.metapop_model.modify_subpop_params(self.name,
                                                    updates_dict)
        else:
            # Since `SubpopParams` is frozen, we return a new instance
            #   with the reflected updates
            self.params = clt.updated_dataclass(self.params, updates_dict)


class FluMetapopModel(clt.MetapopModel, ABC):
    """
    MetapopModel-derived class specific to flu model.
    """

    def __init__(self,
                 subpop_models: list[dict],
                 mixing_params: FluMixingParams,
                 name: str = ""):

        super().__init__(subpop_models,
                         mixing_params,
                         name)

        # Confirm validity and consistency of `FluMixingParams`
        try:
            num_locations = mixing_params.num_locations
        except KeyError:
            raise FluMetapopModelError("'mixing_params' must contain the key 'num_locations'. \n"
                                       "Please specify it before continuing.")
        if num_locations != len(subpop_models):
            raise FluMetapopModelError("'num_locations' should equal the number of items in \n"
                                       "'subpop_models'. Please amend before continuing.")

        self.travel_state_tensors = FluTravelStateTensors()
        self.update_travel_state_tensors()

        # `FluMixingParams` info is stored on `FluTravelParamsTensors` --
        # this order of operations below is important, because
        # `mixing_params` attribute must be defined before `update_travel_params_tensors()`
        # is called.
        self.mixing_params = mixing_params
        self.travel_params_tensors = FluTravelParamsTensors()
        self.update_travel_params_tensors()

        total_pop_LAR_tensor = self.compute_total_pop_LAR_tensor()

        self.precomputed = FluPrecomputedTensors(total_pop_LAR_tensor,
                                                 self.travel_params_tensors)

        # Generally not used unless using torch version
        self._full_metapop_params_tensors = None
        self._full_metapop_state_tensors = None
        self._full_metapop_schedule_tensors = None

    def modify_subpop_params(self,
                             subpop_name: str,
                             updates_dict: dict):
        """
        This method lets users safely modify a single subpopulation
        parameters field; the metapopulation-wide tensors are updated
        automatically afterward.

        In a `FluMetapopModel`, subpopulation parameters are combined into
        (L, A, R) tensors across L subpopulations.`FluSubpopParams` is a frozen
        dataclass to avoid users naively changing parameter values and getting
        undesirable results -- thus, `FluSubpopParams` on a subpopulation
        model cannot be updated directly.

        Parameters:
            subpop_name (str):
               Value must match the `name` attribute of one of the
               `FluSubpopModel` instances contained in this metapopulation
                model's `subpop_models` attribute.
            updates_dict (dict):
                Dictionary specifying values to update in a
                `FluSubpopParams` instance -- keys must match the
                field names of `FluSubpopParams`.
        """

        # Since `FluSubpopParams` is frozen, we return a new instance
        #   with the reflected updates
        self.subpop_models[subpop_name].params = clt.updated_dataclass(
            self.subpop_models[subpop_name].params, updates_dict
        )

        self.update_travel_params_tensors()

        # Adding this for extra safety in case the user does not
        # call `get_flu_torch_inputs` for accessing the
        # `FullMetapopParams` instance.

        # If this attribute is not `None`, it means we are using
        # the `torch` implementation, and we should update the
        # corresponding `FullMetapopParams` instance with the new
        # `FluMixingParams` values.
        if self._full_metapop_params_tensors:
            self.update_full_metapop_params_tensors()

    def modify_mixing_params(self,
                             updates_dict: dict):
        """
        This method lets users safely modify flu mixing parameters;
        the metapopulation-wide tensors are updated automatically afterward.
        `FluMixingParams` is a frozen dataclass to avoid users
        naively changing parameter values and getting undesirable results --
        thus, `FluMixingParams` cannot be updated directly.

        Parameters:
            updates_dict (dict):
                Dictionary specifying values to update in a
                `FluSubpopParams` instance -- keys must match the
                field names of `FluSubpopParams`. 
        """

        self.mixing_params = clt.updated_dataclass(self.mixing_params, updates_dict)
        self.update_travel_params_tensors()

        nonlocal_travel_prop = self.travel_params_tensors.travel_proportions.clone().fill_diagonal_(0.0)

        self.precomputed.sum_residents_nonlocal_travel_prop = nonlocal_travel_prop.sum(dim=1)

        # Adding this for extra safety in case the user does not
        # call `get_flu_torch_inputs` for accessing the
        # `FullMetapopParams` instance.

        # If this attribute is not `None`, it means we are using
        # the `torch` implementation, and we should update the
        # corresponding `FullMetapopParams` instance with the new
        # `FluMixingParams` values.
        if self._full_metapop_params_tensors:
            self.update_full_metapop_params_tensors()

    def compute_total_pop_LAR_tensor(self) -> torch.tensor:
        """
        For each subpopulation, sum initial values of population
        in each compartment for age-risk groups. Store all information
        as tensor and return tensor.

        Returns:
        --------
        torch.tensor of size (L, A, R):
            Total population (across all compartments) for
            location-age-risk (l, a, r).
        """

        # ORDER MATTERS! USE ORDERED DICTIONARY HERE
        #   to preserve correct index order in tensors!
        #   See `update_travel_params_tensors` for detailed note.
        subpop_models_ordered = self._subpop_models_ordered

        total_pop_LAR_tensor = torch.zeros(self.travel_params_tensors.num_locations,
                                           self.travel_params_tensors.num_age_groups,
                                           self.travel_params_tensors.num_risk_groups)

        # All subpop models should have the same compartments' keys
        for name in subpop_models_ordered[0].compartments.keys():

            metapop_vals = []

            for model in subpop_models_ordered.values():
                compartment = getattr(model.compartments, name)
                metapop_vals.append(compartment.current_val)

            total_pop_LAR_tensor = total_pop_LAR_tensor + torch.tensor(np.asarray(metapop_vals))

        return total_pop_LAR_tensor

    def update_state_tensors(self,
                             target: FluTravelStateTensors) -> None:
        """
        Update `target` instance in-place with current simulation
        values. Each field of `target` corresponds to a field in
        `FluSubpopState`, and contains either a tensor of size
        (L, A, R) or a tensor of size (L), where (l, a, r) refers to
        location-age-risk.
        """

        # ORDER MATTERS! USE ORDERED DICTIONARY HERE
        #   to preserve correct index order in tensors!
        #   See `update_travel_params_tensors` for detailed note.
        subpop_models_ordered = self._subpop_models_ordered

        for field in fields(target):

            name = field.name

            # FluTravelStateTensors has an attribute
            #   that is a dictionary called `init_vals` --
            #   disregard, as this only used to store
            #   initial values for resetting, but is not
            #   used in the travel model computation
            if name == "init_vals":
                continue

            metapop_vals = []

            for model in subpop_models_ordered.values():
                current_val = getattr(model.state, name)
                metapop_vals.append(current_val)

            # Probably want to update this to be cleaner...
            # `SubpopState` fields that correspond to `Schedule` instances
            # have initial values of `None` -- but we cannot build a tensor
            # with `None` values, so we convert values to 0s.
            if any(v is None for v in metapop_vals):
                setattr(target, name, torch.tensor(np.full(np.shape(metapop_vals), 0.0)))
            else:
                setattr(target, name, torch.tensor(np.asarray(metapop_vals)))

            # Only fields corresponding to `Schedule` instances can be
            # size (L) -- this is because the schedule value may be scalar for
            # each subpopulation. Other fields should all be size (L, A, R). 

    def update_travel_state_tensors(self) -> None:
        """
        Update `travel_state_tensors` attribute in-place.
        `FluTravelStateTensors` only has fields corresponding
        to state variables relevant for the travel model.
        Converts subpopulation-specific state to
        tensors of size (L, A, R) for location-age-risk
        (except for a few exceptions that have different dimensions).
        """

        self.update_state_tensors(self.travel_state_tensors)

    def update_full_metapop_state_tensors(self) -> None:
        """
        Update `_full_metapop_state_tensors` attribute in-place.
        `FluFullMetapopStateTensors` has fields corresponding
        to all state variables in the simulation.
        Converts subpopulation-specific state to
        tensors of size (L, A, R) for location-age-risk
        (except for a few exceptions that have different dimensions).
        """

        if self._full_metapop_state_tensors is None:
            self._full_metapop_state_tensors = FluFullMetapopStateTensors()
        self.update_state_tensors(self._full_metapop_state_tensors)

    def update_params_tensors(self,
                              target: FluTravelParamsTensors) -> FluTravelParamsTensors:
        """
        Update `target` in-place. Converts subpopulation-specific
        parameters to tensors of size (L, A, R) for location-age-risk,
        except for `num_locations` and `travel_proportions`, which
        have size 1 and (L, L) respectively.
        """

        # USE THE ORDERED DICTIONARY HERE FOR SAFETY!
        #   AGAIN, ORDER MATTERS BECAUSE ORDER DETERMINES
        #   THE SUBPOPULATION INDEX IN THE METAPOPULATION
        #   TENSOR!
        subpop_models_ordered = self._subpop_models_ordered

        # Subpop models should have the same A, R so grab
        #   from the first subpop model
        A = subpop_models_ordered[0].params.num_age_groups
        R = subpop_models_ordered[0].params.num_risk_groups

        for field in fields(target):

            name = field.name

            metapop_vals = []

            if name == "num_locations" or name == "travel_proportions":
                setattr(target, name, torch.tensor(getattr(self.mixing_params, name)))

            else:
                for model in subpop_models_ordered.values():
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
                    metapop_vals = np.asarray(metapop_vals)
                    # metapop_vals = np.stack([clt.to_AR_array(x, A, R) for x in metapop_vals])

                setattr(target, name, torch.tensor(metapop_vals))

        # Convert all tensors to correct size!
        target.standardize_shapes()

    def update_travel_params_tensors(self) -> None:
        """
        Update `travel_params_tensors` attribute in-place.
        `FluTravelParamsTensors` only has fields corresponding
        to parameters relevant for the travel model.
        Converts subpopulation-specific parameters to
        tensors of size (L, A, R) for location-age-risk
        (except for a few exceptions that have different dimensions).
        """

        self.update_params_tensors(target=self.travel_params_tensors)

    def update_full_metapop_params_tensors(self) -> None:
        """
        Update `_full_metapop_params_tensors` attribute in-place.
        `FluFullMetapopParamsTensors` has fields corresponding
        to all parameters in the simulation. Converts subpopulation-specific
        parameters to tensors of size (L, A, R) for location-age-risk
        (except for a few exceptions that have different dimensions).
        """

        if self._full_metapop_params_tensors is None:
            self._full_metapop_params_tensors = FluFullMetapopParamsTensors()
        self.update_params_tensors(target=self._full_metapop_params_tensors)

    def apply_inter_subpop_updates(self) -> None:
        """
        Update the `FluTravelStateTensors` according to the simulation state
        and compute the total mixing exposure, which includes across-subpopulation
        mixing/travel. Update the `total_mixing_exposure` attribute on each
        subpopulation's `SusceptibleToExposed` instance accordingly, so each
        of these transition variables can compute its transition rate.

        See `apply_inter_subpop_updates` on `MetapopModel` base class
        for logic of how/when this is called in the simulation.
        """

        self.update_travel_state_tensors()

        total_mixing_exposure = compute_total_mixing_exposure(self.travel_state_tensors,
                                                              self.travel_params_tensors,
                                                              self.precomputed)

        # Again, `self.subpop_models` is an ordered dictionary --
        #   so iterating over the dictionary like this is well-defined
        #   and responsible -- the order is important because it
        #   determines the order (index) in any metapopulation tensors
        subpop_models = self._subpop_models_ordered

        # Updates `total_mixing_exposure` attribute on each `SusceptibleToExposed`
        # instance -- this value captures across-population travel/mixing.
        for i in range(len(subpop_models)):
            subpop_models.values()[i].transition_variables.S_to_E.total_mixing_exposure = \
                total_mixing_exposure[i, :, :]

    def setup_full_metapop_schedule_tensors(self):
        """
        Creates `FluFullMetapopScheduleTensors` instance and assigns to
        `_full_metapop_schedule_tensors` attribute.

        For the metapopulation model's L locations/subpopulations, for each day,
        each value-related column in each schedule is either a float or
        array of size (A, R) for age-risk groups.

        We aggregate and reformat this schedule information and put it
        into a `FluFullMetapopScheduleTensors` instance, where fields
        correspond to a schedule value, and values are lists of tensors of
        size (L, A, R). The ith element of each list corresponds to the
        ith simulation day.
        """

        self._full_metapop_schedule_tensors = FluFullMetapopScheduleTensors()

        L = self.precomputed.L
        A = self.precomputed.A
        R = self.precomputed.R

        # Note: there is probably a more consistent way to do this,
        # because now `flu_contact_matrix` has two values: "is_school_day"
        # and "is_work_day" -- other schedules' dataframes only have one
        # relevant column value rather than two
        for item in [("absolute_humidity", "absolute_humidity"),
                     ("flu_contact_matrix", "is_school_day"),
                     ("flu_contact_matrix", "is_work_day"),
                     ("daily_vaccines", "daily_vaccines")]:

            schedule_name = item[0]
            values_column_name = item[1]

            metapop_vals = []

            for subpop_model in self._subpop_models_ordered.values():
                df = subpop_model.schedules[schedule_name].timeseries_df

                # Using the `start_real_date` specification given in subpop's `SimulationSettings`,
                # extract the relevant part of the dataframe with dates >= the simulation start date.
                # Note that `start_real_date` should be the same for each subpopulation
                start_date = datetime.datetime.strptime(subpop_model.simulation_settings.start_real_date, "%Y-%m-%d")
                df["simulation_day"] = (pd.to_datetime(df["date"], format="%Y-%m-%d") - start_date).dt.days
                df = df[df["simulation_day"] >= 0]

                # Make each day's value an A x R array
                # Pandas complains about `SettingWithCopyWarning` so we work on a copy explicitly to stop it
                #   from complaining...
                df = df.copy()
                df[values_column_name] = df[values_column_name].astype(object)
                df.loc[:, values_column_name] = df[values_column_name].apply(
                    lambda x, A=A, R=R: np.broadcast_to(np.asarray(x).reshape(1, 1), (A, R))
                )

                metapop_vals.append(np.asarray(df[values_column_name]))

            # IMPORTANT: tedious array/tensor shape/size manipulation here
            # metapop_vals: list of L arrays, each shape (num_days, A, R)
            # We need to transpose this... to be a list of num_days tensors, of size L x A x R
            num_items = metapop_vals[0].shape[0]

            # This is ugly and inefficient -- but at least we only do this once, when we get the initial
            #   state of a metapopulation model in tensor form
            transposed_metapop_vals = [torch.tensor(np.array([metapop_vals[l][i] for l in range(L)])) for i in
                                       range(num_items)]

            setattr(self._full_metapop_schedule_tensors, values_column_name, transposed_metapop_vals)

    def get_flu_torch_inputs(self) -> dict:
        """
        Prepares and returns metapopulation simulation data in tensor format
        that can be directly used for `torch` implementation.

        Returns:
             d (dict):
                Has keys "state_tensors", "params_tensors", "schedule_tensors",
                and "precomputed". Corresponds to `FluFullMetapopStateTensors`,
                `FluFullMetapopParamsTensors`, `FluFullMetapopScheduleTensors`,
                and `FluPrecomputedTensors` instances respectively.
        """

        # Note: does not support dynamic variables (yet). If want to
        #   run pytorch with dynamic variables, will need to create
        #   a method similar to `setup_full_metapop_schedule_tensors`
        #   but for dynamic variables. Also note that we cannot differentiate
        #   with respect to dynamic variables that are discontinuous
        #   (e.g. a 0-1 intervention) -- so we cannot optimize discontinuous
        #   dynamic variables.

        self.update_full_metapop_state_tensors()
        self.update_full_metapop_params_tensors()
        self._full_metapop_params_tensors.standardize_shapes()
        self.setup_full_metapop_schedule_tensors()

        d = {}

        d["state_tensors"] = copy.deepcopy(self._full_metapop_state_tensors)
        d["params_tensors"] = copy.deepcopy(self._full_metapop_params_tensors)
        d["schedule_tensors"] = copy.deepcopy(self._full_metapop_schedule_tensors)
        d["precomputed"] = copy.deepcopy(self.precomputed)

        return d
