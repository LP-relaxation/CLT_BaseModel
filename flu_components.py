import numpy as np
import json
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import base_components as base

import matplotlib.pyplot as plt
from plotting import create_basic_compartment_history_plot

from datetime import date


# Note: for dataclasses, Optional is used to help with static type checking
# -- it means that an attribute can either hold a value with the specified
# datatype or it can be None


def absolute_humidity_func(current_date: date):
    """
    Note: this is a dummy function loosely based off of
    the absolute humidity data from Kaiming and Shraddha's
    new burden averted draft.

    TODO: replace this function with real humidity function

    :param current_date: datetime.date,
        datetime.date object corresponding to
        real-world date
    :return: float,
        nonnegative float between 3.8 and 12.5
        corresponding to absolute humidity
        that day of the year
    """

    # Convert datetime.date to integer between 1 and 365
    #   corresponding to day of the year
    day_of_year = current_date.timetuple().tm_yday

    # Vertex of the parabola
    h = 180
    k = 3.8

    # This calculation is used to achieve the correct
    #   upside-down parabola with the right min and max
    #   values and location
    # max_value = 12.5
    # 0.00027 = (max_value - k) / ((0 - h) ** 2)

    # Calculate the value of the function
    # Shift by 180 (6 months roughly), because minimum humidity occurs in
    #   January, but Kaiming and Shraddha's graph starts in July
    return k + 0.00027 * (day_of_year - 180 - h) ** 2


@dataclass
class FluFixedParams(base.FixedParams):
    """
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

    TODO: when adding multiple strains, need to add subscripts
        to math of attributes and add strain-specific description

    Note: for attribute description, "pseudo-LaTeX" is used --
    backslashes are omitted due to their incompatibility
    with Python docstrings.

    Attributes
    ----------
    :ivar num_age_groups: number of age groups
            math variable: $|A|$, where $A$ is the set of age groups
    :ivar num_risk_groups: number of risk groups
            math variable: $|L|$, where $L$ is the set of risk groups
    :ivar beta_baseline: transmission rate
            math variable: $beta_0$
    :ivar total_population_val: total number in population,
        summed across all age-risk groups
            math variable: $N$
    :ivar humidity_impact: coefficient that determines
        how much absolute humidity affects beta_baseline
            math variable: $xi$
    :ivar immunity_hosp_increase_factor: factor by which
        population-level immunity against hospitalization
        grows after each case that recovers
            math variable: $g^H$
    :ivar immunity_inf_increase_factor: factor by which
        population-level immunity against infection
        grows after each case that recovers
            math variable: $g^I$
    :ivar immunity_saturation_constant: positive constant
        modeling saturation of antibody production of individuals
            math variable: $o$
    :ivar waning_factor_hosp: rate at which infection-induced
        immunity against hospitalization wanes
            math variable: $w^H$
    :ivar waning_factor_inf: rate at which infection-induced
        immunity against infection wanes
            math variable: $w^I$
    :ivar hosp_risk_reduction: reduction in hospitalization
        risk from infection-induced immunity
            math variable: $K^H$
    :ivar inf_risk_reduction: reduction in infection risk
        from infection-induced immunity
            math variable: $K^I$
    :ivar death_risk_reduction: reduction in death risk
        from infection-induced immunity
            math variable: $K^D$
    :ivar R_to_S_rate: rate at which people in R move to S
            math variable: $eta$
    :ivar E_to_I_rate: rate at which people in E move to I
            math variable: $omega$
    :ivar I_to_R_rate: rate at which people in I move to R
            math variable: $gamma$
    :ivar I_to_H_rate: rate at which people in I move to H
            math variable: $zeta$
    :ivar H_to_R_rate: rate at which people in H move to R
            math variable: $gamma_H$
    :ivar H_to_D_rate: rate at which people in H move to D
            math variable: $pi$
    :ivar I_to_H_adjusted_proportion: rate-adjusted proportion
        infected who are hospitalized based on age-risk groups
            math variable: $[tilde{mu}_{a, ell}]$
    :ivar H_to_D_adjusted_proportion: rate-adjusted proportion
        hospitalized who die based on age-risk groups
            math variable: $[tilde{nu}_{a, ell}]$
    """

    num_age_groups: Optional[int] = None
    num_risk_groups: Optional[int] = None
    beta_baseline: Optional[float] = None
    total_population_val: Optional[np.ndarray] = None
    humidity_impact: Optional[float] = None
    immunity_hosp_increase_factor: Optional[float] = None
    immunity_inf_increase_factor: Optional[float] = None
    immunity_saturation_constant: Optional[float] = None
    waning_factor_hosp: Optional[float] = None
    waning_factor_inf: Optional[float] = None
    hosp_risk_reduction: Optional[float] = None
    inf_risk_reduction: Optional[float] = None
    death_risk_reduction: Optional[float] = None
    R_to_S_rate: Optional[float] = None
    E_to_I_rate: Optional[float] = None
    I_to_R_rate: Optional[float] = None
    I_to_H_rate: Optional[float] = None
    H_to_R_rate: Optional[float] = None
    H_to_D_rate: Optional[float] = None
    I_to_H_adjusted_proportion: Optional[np.ndarray] = None
    H_to_D_adjusted_proportion: Optional[np.ndarray] = None


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

    Attributes
    ----------
    :param S: susceptible compartment for age-risk groups
        (EpiCompartment current_val)
            math variable: $S$
    :param E: exposed compartment for age-risk groups
        (EpiCompartment current_val)
            math variable: $E$
    :param I: infected compartment for age-risk groups
        (EpiCompartment current_val)
            math variable: $I$
    :param H: hospital compartment for age-risk groups
        (EpiCompartment current_val)
            math variable: $H$
    :param R: recovered compartment for age-risk groups
        (EpiCompartment current_val)
            math variable: $R$
    :param D: dead compartment for age-risk groups
        (EpiCompartment current_val)
            math variable: $D$
    :param population_immunity_hosp: infection-induced
        population-level immunity against hospitalization, for
        age-risk groups (EpiMetric current_val)
            math variable: $M^H$
    :param population_immunity_inf: infection-induced
        population-level immunity against infection, for
        age-risk groups (EpiMetric current_val)
            math variable: $M^I$
    :param absolute_humidity:
        grams of water vapor per cubic meter g/m^3,
        used as seasonality parameter that influences
        transmission rate beta_baseline
            math variable: $q$
    """

    S: Optional[np.ndarray] = None
    E: Optional[np.ndarray] = None
    I: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    D: Optional[np.ndarray] = None
    population_immunity_hosp: Optional[np.ndarray] = None
    population_immunity_inf: Optional[np.ndarray] = None
    absolute_humidity: Optional[float] = None


class NewExposed(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: base.SimState,
                         fixed_params):
        """
        :param sim_state:
        :param fixed_params:
        :return:
        """
        force_of_immunity = (1 + fixed_params.inf_risk_reduction * sim_state.population_immunity_inf)

        beta_humidity_adjusted = (1 - sim_state.absolute_humidity * fixed_params.humidity_impact) * \
                                 fixed_params.beta_baseline

        return np.asarray((1-sim_state.beta_reduct) * beta_humidity_adjusted * sim_state.I
                          / (fixed_params.total_population_val * force_of_immunity))


class NewSusceptible(base.TransitionVariable):
    def get_current_rate(self, sim_state, fixed_params):
        """
        :param sim_state:
        :param fixed_params:
        :return:
        """
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups), fixed_params.R_to_S_rate)


class NewInfected(base.TransitionVariable):
    def get_current_rate(self, sim_state, fixed_params):
        """
        :param sim_state:
        :param fixed_params:
        :return:
        """
        return np.full((fixed_params.num_age_groups, fixed_params.num_age_groups), fixed_params.E_to_I_rate)


class NewRecoveredHome(base.TransitionVariable):
    def get_current_rate(self, sim_state, fixed_params):
        """
        :param sim_state:
        :param fixed_params:
        :return:
        """
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       (1 - fixed_params.I_to_H_adjusted_proportion) * fixed_params.I_to_R_rate)


class NewRecoveredHosp(base.TransitionVariable):
    def get_current_rate(self, sim_state, fixed_params):
        """
        :param sim_state:
        :param fixed_params:
        :return:
        """
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       (1 - fixed_params.H_to_D_adjusted_proportion) * fixed_params.H_to_R_rate)


class NewHosp(base.TransitionVariable):
    def get_current_rate(self, sim_state, fixed_params):
        """
        :param sim_state:
        :param fixed_params:
        :return:
        """
        return np.asarray(fixed_params.I_to_H_rate * fixed_params.I_to_H_adjusted_proportion /
                          (1 + fixed_params.hosp_risk_reduction * sim_state.population_immunity_hosp))


class NewDead(base.TransitionVariable):
    def get_current_rate(self, sim_state, fixed_params):
        """
        :param sim_state:
        :param fixed_params:
        :return:
        """
        return np.asarray(fixed_params.H_to_D_adjusted_proportion * fixed_params.H_to_D_rate /
                          (1 + fixed_params.death_risk_reduction * sim_state.population_immunity_hosp))


class PopulationImmunityHosp(base.EpiMetric):
    def get_change_in_current_val(self,
                                  sim_state,
                                  fixed_params: FluFixedParams,
                                  num_timesteps):
        """
        :param sim_state:
        :param fixed_params:
        :param num_timesteps
        :return:
        """
        immunity_gain = (fixed_params.immunity_hosp_increase_factor * sim_state.R) / \
                        (fixed_params.total_population_val *
                         (1 + fixed_params.immunity_saturation_constant * sim_state.population_immunity_hosp))
        immunity_loss = fixed_params.waning_factor_hosp * sim_state.population_immunity_hosp

        return np.asarray(immunity_gain - immunity_loss) / num_timesteps


class PopulationImmunityInf(base.EpiMetric):
    def get_change_in_current_val(self,
                                  sim_state,
                                  fixed_params: FluFixedParams,
                                  num_timesteps):
        """
        :param sim_state:
        :param fixed_params:
        :param num_timesteps
        :return:
        """
        immunity_gain = (fixed_params.immunity_inf_increase_factor * sim_state.R) / \
                        (fixed_params.total_population_val * (1 + fixed_params.immunity_saturation_constant *
                                                              sim_state.population_immunity_inf))
        immunity_loss = fixed_params.waning_factor_inf * sim_state.population_immunity_inf

        return np.asarray(immunity_gain - immunity_loss) / num_timesteps


class BetaReduct(base.DynamicVal):

    def __init__(self, name):
        super().__init__(name)
        self.permanent_lockdown = False

    def update_current_val(self, sim_state, fixed_params):
        if np.sum(sim_state.I) / np.sum(fixed_params.total_population_val) > 0.05:
            self.current_val = 1
            self.permanent_lockdown = True
        else:
            if not self.permanent_lockdown:
                self.current_val = 0


class AbsoluteHumidity(base.Schedule):
    def update_current_val(self, current_date: date) -> None:
        self.current_val = absolute_humidity_func(current_date)


class FluModelConstructor(base.ModelConstructor):
    """
    Class for creating ImmunoSEIRS flu model with predetermined fixed
    structure -- initial values and epidemiological structure are
    populated by user-specified JSON files.

    Key method create_transmission_model returns a TransmissionModel
    instance with S-E-I-H-R-D compartments and population_immunity_inf
    and population_immunity_hosp epi metrics. The structure
    is as follows:
        S = new_susceptible - new_exposed
        E = new_exposed - new_infected
        I = new_infected - new_recovered_home - new_hospitalized
        H = new_hospitalized - new_recovered_hosp - new_dead
        R = new_recovered_home + new_recovered_hosp - new_susceptible
        D = new_dead

    The following are TransitionVariable instances:
        new_susceptible is a NewSusceptible instance
        new_exposed is a NewExposed instance
        new_infected is a NewInfected instance
        new_hospitalized is a NewHospitalized instance
        new_recovered_home is a NewRecoveredHome instance
        new_recovered_hosp is a NewRecoveredHosp instance
        new_dead is a NewDead instance

    There are two TransitionVariableGroups:
        I_out (since new_recovered_home and new_hospitalized are joint random variables)
        H_out (since new_recovered_hosp and new_dead are joint random variables)

    The following are EpiMetric instances:
        population_immunity_inf is a PopulationImmunityInf instance
        population_immunity_hosp is a PopulationImmunityHosp instance

    Transition rates and update formulas are specified in
        corresponding classes.

    Attributes
    ----------
    :ivar config: Config dataclass instance,
        holds configuration values
    :ivar fixed_params: FluFixedParams dataclass instance,
        holds epidemiological parameter values, read-in
        from user-specified JSON
    :ivar sim_state: FluSimState dataclass instance,
        holds current simulation state information,
        such as current values of epidemiological compartments
        and epi metrics, read in from user-specified JSON
    :ivar transition_variable_lookup: dict,
        maps string to corresponding TransitionVariable
    :ivar transition_variable_group_lookup: dict,
        maps string to corresponding TransitionVariableGroup
    :ivar compartment_lookup: dict,
        maps string to corresponding EpiCompartment,
        using the value of the EpiCompartment's "name" attribute
    :ivar epi_metric_lookup: dict,
        maps string to corresponding EpiMetric,
        using the value of the EpiMetric's "name" attribute
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

        :param config_filepath: Optional[str],
            path to config JSON file (path includes actual filename
            with suffix ".json") -- all JSON fields must match
            name and datatype of Config instance attributes
        :param fixed_params_filepath: Optional[str],
            path to epidemiological parameters JSON file
            (path includes actual filename with suffix ".json")
            -- all JSON fields must match name and datatype of
            FixedParams instance attributes
        :param state_vars_init_vals_filepath: Optional[str],
            path to epidemiological compartments JSON file
            (path includes actual filename with suffix ".json")
            -- all JSON fields must match name and datatype of
            StateVariable instance attributes -- these initial
            values are used to populate sim_state attribute
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
        Create compartments S-E-I-H-R-D (6 compartments total)
        and add them to compartment_lookup for dictionary access
        """

        for name in ("S", "E", "I", "H", "R", "D"):
            self.compartment_lookup[name] = base.EpiCompartment(name, getattr(self.sim_state, name))

    def setup_epi_metrics(self) -> None:
        """
        Create all epi metric described in docstring (2 state
        variables total) and add them to epi_metric_lookup attribute
        for dictionary access
        """

        self.epi_metric_lookup["population_immunity_inf"] = \
            PopulationImmunityInf("population_immunity_inf",
                                  getattr(self.sim_state, "population_immunity_inf"))

        self.epi_metric_lookup["population_immunity_hosp"] = \
            PopulationImmunityHosp("population_immunity_hosp",
                                   getattr(self.sim_state, "population_immunity_hosp"))

    def setup_dynamic_vals(self) -> None:
        self.dynamic_val_lookup["beta_reduct"] = BetaReduct("beta_reduct")

    def setup_schedules(self) -> None:
        self.schedule_lookup["absolute_humidity"] = AbsoluteHumidity("absolute_humidity")

    def setup_transition_variables(self) -> None:
        """
        Create all transition variables described in docstring (7 transition
        variables total) and add them to transition_variable_lookup attribute
        for dictionary access
        """

        compartments = self.compartment_lookup
        transition_type = self.config.transition_type

        # Reordering the tuples to put the transition function first
        transition_mapping = {
            "new_susceptible": (NewSusceptible, "new_susceptible", compartments["R"], compartments["S"]),
            "new_exposed": (NewExposed, "new_exposed", compartments["S"], compartments["E"]),
            "new_infected": (NewInfected, "new_infected", compartments["E"], compartments["I"]),
            "new_recovered_home": (
                NewRecoveredHome, "new_recovered_home", compartments["I"], compartments["R"], True),
            "new_hosp": (NewHosp, "new_hosp", compartments["I"], compartments["H"], True),
            "new_recovered_hosp": (
                NewRecoveredHosp, "new_recovered_hosp", compartments["H"], compartments["R"], True),
            "new_dead": (NewDead, "new_dead", compartments["H"], compartments["D"], True)
        }

        # Create transition variables dynamically
        # params[0] is the TransitionVariable subclass (e.g. NewSusceptible)
        # params[1:4] refers to the name, origin compartment, destination compartment
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
            "I_out": base.TransitionVariableGroup("I_out",
                                                  compartment_lookup["I"],
                                                  transition_type,
                                                  (tvar_lookup["new_recovered_home"],
                                                   tvar_lookup["new_hosp"])),
            "H_out": base.TransitionVariableGroup("H_out",
                                                  compartment_lookup["H"],
                                                  transition_type,
                                                  (tvar_lookup["new_recovered_hosp"],
                                                   tvar_lookup["new_dead"]))
        }
