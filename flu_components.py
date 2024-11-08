import numpy as np
import json
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Type
from pathlib import Path
base_path = Path(__file__).parent / "flu_demo_input_files"

import base_components as base

import matplotlib.pyplot as plt
from plotting import create_basic_compartment_history_plot

import datetime

import pandas as pd


# Note: for dataclasses, Optional is used to help with static type checking
# -- it means that an attribute can either hold a value with the specified
# datatype or it can be None


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

    Note:
        For attribute description, "pseudo-LaTeX" is used --
        backslashes are omitted due to their incompatibility
        with Python docstrings.

    Attributes:
        num_age_groups (positive int):
            number of age groups -- math variable:
            $|A|$, where $A$ is the set of age groups.
        num_risk_groups (positive int):
            number of risk groups -- math variable:
            $|L|$, where $L$ is the set of risk groups.
        beta_baseline (positive float): transmission rate
            -- math variable: $beta_0$.
        total_population_val (np.ndarray of positive ints):
            total number in population, summed across all
            age-risk groups -- math variable: $N$.
        humidity_impact (positive float):
            coefficient that determines how much absolute
            humidity affects beta_baseline -- math variable: $xi$.
        immunity_hosp_increase_factor (positive float):
            factor by which population-level immunity
            against hospitalization grows after each
            case that recovers -- math variable: $g^H$.
        immunity_inf_increase_factor (positive float):
            factor by which population-level immunity
            against infection grows after each case
                that recovers -- math variable: $g^I$.
        immunity_saturation_constant (positive float):
            constant modeling saturation of antibody
            production of individuals -- math variable: $o$.
        waning_factor_hosp (positive float):
            rate at which infection-induced immunity
            against hospitalization wanes -- math variable: $w^H$.
        waning_factor_inf (positive float):
            rate at which infection-induced immunity
            against infection wanes -- math variable: $w^I$.
        hosp_risk_reduction (positive float in [0,1]):
            reduction in hospitalization risk from
            infection-induced immunity -- math variable: $K^H$.
        inf_risk_reduction (positive float in [0,1]):
            reduction in infection risk
            from infection-induced immunity -- math variable: $K^I$.
        death_risk_reduction (positive float in [0,1]):
            reduction in death risk from infection-induced immunity
            -- math variable: $K^D$.
        R_to_S_rate (positive float):
            rate at which people in R move to S -- math variable: $eta$.
        E_to_I_rate (positive float):
            rate at which people in E move to I -- math variable: $omega$.
        I_to_R_rate (positive float):
            rate at which people in I move to R -- math variable: $gamma$.
        I_to_H_rate (positive float):
            rate at which people in I move to H -- math variable: $zeta$.
        H_to_R_rate (positive float):
            rate at which people in H move to R -- math variable: $gamma_H$.
        H_to_D_rate (positive float):
            rate at which people in H move to D -- math variable: $pi$.
        I_to_H_adjusted_proportion (np.ndarray of positive floats in [0,1]):
            rate-adjusted proportion -- infected who are hospitalized
            based on age-risk groups -- math variable: $[tilde{mu}_{a, ell}]$.
        H_to_D_adjusted_proportion (np.ndarray of positive floats in [0,1]):
            rate-adjusted proportion hospitalized who die based on
            age-risk groups -- math variable: $[tilde{nu}_{a, ell}]$.
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

    Attributes:
        S (np.ndarray of positive floats):
            susceptible compartment for age-risk groups --
            (holds current_val of EpiCompartment "S")
            -- math variable: $S$.
        E (np.ndarray of positive floats):
            exposed compartment for age-risk groups --
            (holds current_val of EpiCompartment "E")
            -- math variable: $E$.
        I (np.ndarray of positive floats):
            infected compartment for age-risk groups
            (holds current_val of EpiCompartment "I")
            -- math variable: $I$.
        H (np.ndarray of positive floats):
            hospital compartment for age-risk groups
            (holds current_val of EpiCompartment "H")
            -- math variable: $H$.
        R (np.ndarray of positive floats):
            recovered compartment for age-risk groups
            (holds current_val of EpiCompartment "R")
            -- math variable: $R$.
        D (np.ndarray of positive floats):
            dead compartment for age-risk groups
            (holds current_val of EpiCompartment "D")
            -- math variable: $D$.
        population_immunity_hosp (np.ndarray of positive floats):
            infection-induced population-level immunity against
            hospitalization, for age-risk groups (holds current_val
            of EpiMetric "population_immunity_hosp") -- math variable: $M^H$.
        population_immunity_inf (np.ndarray of positive floats):
            infection-induced population-level immunity against
            infection, for age-risk groups (holds current_val
            of EpiMetric "population_immunity_inf") -- math variable: $M^I$.
        absolute_humidity (positive float):
            grams of water vapor per cubic meter g/m^3,
            used as seasonality parameter that influences
            transmission rate beta_baseline -- math variable: $q$.
        flu_contact_matrix (np.ndarray of positive floats):
            A x L x A x L array, where A is the number of age
            groups and L is the number of risk groups --
            element (a, l, a', l') corresponds to the number of
            contacts that a person in age-risk group a,l
            has with people in age-risk group a', l' --
            math variable: "phi$.
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
    flu_contact_matrix: Optional[np.ndarray] = None


class NewExposed(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):

        force_of_immunity = (1 + fixed_params.inf_risk_reduction * sim_state.population_immunity_inf)

        beta_humidity_adjusted = (1 - sim_state.absolute_humidity * fixed_params.humidity_impact) * \
                                 fixed_params.beta_baseline

        # Compute I / N -> original shape is (A, L)
        # Expand ratio for broadcasting -> new shape is (1, 1, A, L)
        I_N_ratio_expanded = (sim_state.I / fixed_params.total_population_val)[None, None, :, :]

        # Expand force_of_immunity for broadcasting -> new shape is (A, L, 1, 1)
        force_of_immunity_expanded = force_of_immunity[:, :, None, None]

        # Element-wise multiplication and division by M_expanded
        # Sum over a' and l' (last two dimensions) -> result has shape (A, L)
        summand = np.sum(sim_state.flu_contact_matrix * I_N_ratio_expanded / force_of_immunity_expanded, axis=(2,3))

        return (1 - sim_state.beta_reduct) * beta_humidity_adjusted * summand


class NewSusceptible(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups), fixed_params.R_to_S_rate)


class NewInfected(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups), fixed_params.E_to_I_rate)


class NewRecoveredHome(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       (1 - fixed_params.I_to_H_adjusted_proportion) * fixed_params.I_to_R_rate)


class NewRecoveredHosp(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       (1 - fixed_params.H_to_D_adjusted_proportion) * fixed_params.H_to_R_rate)


class NewHosp(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.asarray(fixed_params.I_to_H_rate * fixed_params.I_to_H_adjusted_proportion /
                          (1 + fixed_params.hosp_risk_reduction * sim_state.population_immunity_hosp))


class NewDead(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.asarray(fixed_params.H_to_D_adjusted_proportion * fixed_params.H_to_D_rate /
                          (1 + fixed_params.death_risk_reduction * sim_state.population_immunity_hosp))


class PopulationImmunityHosp(base.EpiMetric):

    def __init__(self, name, init_val, new_susceptible):
        super().__init__(name, init_val)
        self.new_susceptible = new_susceptible

    def get_change_in_current_val(self,
                                  sim_state: FluSimState,
                                  fixed_params: FluFixedParams,
                                  num_timesteps: int):
        immunity_gain = (fixed_params.immunity_hosp_increase_factor * self.new_susceptible.current_val) / \
                        (fixed_params.total_population_val *
                         (1 + fixed_params.immunity_saturation_constant * sim_state.population_immunity_hosp))
        immunity_loss = fixed_params.waning_factor_hosp * sim_state.population_immunity_hosp

        return np.asarray(immunity_gain - immunity_loss) / num_timesteps


class PopulationImmunityInf(base.EpiMetric):
    def __init__(self, name, init_val, new_susceptible):
        super().__init__(name, init_val)
        self.new_susceptible = new_susceptible

    def get_change_in_current_val(self,
                                  sim_state: FluSimState,
                                  fixed_params: FluFixedParams,
                                  num_timesteps: int):
        immunity_gain = (fixed_params.immunity_inf_increase_factor * self.new_susceptible.current_val) / \
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


def absolute_humidity_func(current_date: datetime.date):
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

    # Shift by 180 (6 months roughly), because minimum humidity occurs in
    #   January, but Kaiming and Shraddha's graph starts in July
    return k + 0.00027 * (day_of_year - 180 - h) ** 2


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
                 init_val: Optional[Union[np.ndarray, float]]=None):
        super().__init__(name, init_val)

        df = pd.read_csv(base_path / "school_work_calendar.csv", index_col=0)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        self.time_series_df = df

        self.total_contact_matrix = np.array([[3.5, 1.5], [3, 3.5]]).reshape((2,1,2,1))
        self.school_contact_matrix = np.array([[2, 1], [1, 1]]).reshape((2,1,2,1))
        self.work_contact_matrix = np.array([[0, 0], [1, 2]]).reshape((2,1,2,1))

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
        Create compartments S-E-I-H-R-D (6 compartments total)
        and add them to compartment_lookup for dictionary access
        """

        for name in ("S", "E", "I", "H", "R", "D"):
            self.compartment_lookup[name] = base.EpiCompartment(name, getattr(self.sim_state, name))

    def setup_dynamic_vals(self) -> None:
        self.dynamic_val_lookup["beta_reduct"] = BetaReduct("beta_reduct")

    def setup_schedules(self) -> None:
        self.schedule_lookup["absolute_humidity"] = AbsoluteHumidity("absolute_humidity")
        self.schedule_lookup["flu_contact_matrix"] = FluContactMatrix("flu_contact_matrix")

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

    def setup_epi_metrics(self) -> None:
        """
        Create all epi metric described in docstring (2 state
        variables total) and add them to epi_metric_lookup attribute
        for dictionary access
        """

        self.epi_metric_lookup["population_immunity_inf"] = \
            PopulationImmunityInf("population_immunity_inf",
                                  getattr(self.sim_state, "population_immunity_inf"),
                                  self.transition_variable_lookup["new_susceptible"])

        self.epi_metric_lookup["population_immunity_hosp"] = \
            PopulationImmunityHosp("population_immunity_hosp",
                                   getattr(self.sim_state, "population_immunity_hosp"),
                                   self.transition_variable_lookup["new_susceptible"])