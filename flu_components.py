import numpy as np
import json
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from pathlib import Path

import matplotlib.pyplot as plt

from base_components import approx_binomial_probability_from_rate, \
    Config, TransitionVariableGroup, TransitionVariable, StateVariable, \
    EpiCompartment, TransmissionModel, dataclass_instance_from_json
from plotting import create_basic_compartment_history_plot


# Note: for dataclasses, Optional is used to help with static type checking
# -- it means that an attribute can either hold a value with the specified
# datatype or it can be None


@dataclass
class FluEpiParams:
    """
    Data container for pre-specified and fixed epidemiological
    parameters in ImmunoSEIRS flu model.

    Along with FluSimState, is passed to get_current_rate
    and get_change_in_current_val

    Each field of datatype np.ndarray must be A x L,
    where A is the number of age groups and L is the number of
    risk groups. Note: this means all arrays should be 2D.
    See FluSimState docstring for important formatting note
    on 2D arrays.

    TODO: when adding multiple strains, need to add subscripts
        to math of attributes and add strain-specific description

    Attributes
    ----------
    :ivar num_age_groups: number of age groups
            math variable: $|A|$, where $A$ is the set of age groups
    :ivar num_risk_groups: number of risk groups
            math variable: $|L|$, where $L$ is the set of risk groups
    :ivar beta: transmission rate
            math variable: $\beta$
    :ivar total_population_val: total number in population,
        summed across all age-risk groups
            math variable: $N$
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
            math variable: $\eta$
    :ivar E_to_I_rate: rate at which people in E move to I
            math variable: $\omega$
    :ivar I_to_R_rate: rate at which people in I move to R
            math variable: $\gamma$
    :ivar I_to_H_rate: rate at which people in I move to H
            math variable: $\zeta$
    :ivar H_to_R_rate: rate at which people in H move to R
            math variable: $\gamma_H$
    :ivar H_to_D_rate: rate at which people in H move to D
            math variable: $\pi$
    :ivar I_to_H_adjusted_proportion: rate-adjusted proportion
        infected who are hospitalized based on age-risk groups
            math variable: $[\tilde{\mu}_{a, \ell}]$
    :ivar H_to_D_adjusted_proportion: rate-adjusted proportion
        hospitalized who die based on age-risk groups
            math variable: $[\tilde{\nu}_{a, \ell}]$
    """

    num_age_groups: Optional[int] = None
    num_risk_groups: Optional[int] = None
    beta: Optional[float] = None
    total_population_val: Optional[float] = None
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
class FluSimState:
    """
    Data container for pre-specified and fixed set of
    EpiCompartment initial values and StateVariable initial values
    in ImmunoSEIRS flu model.

    Each field below should be A x L np.ndarray, where
    A is the number of age groups and L is the number of risk groups.
    Note: this means all arrays should be 2D. Even if there is
    1 age group and 1 risk group (no group stratification),
    each array should be 1x1, which is two-dimensional.
    For example, np.array([[100]]) is correct --
    np.array([100]) is wrong.

    Attributes
    ----------
    :ivar S: susceptible compartment for age-risk groups
            math variable: $S$
    :ivar E: exposed compartment for age-risk groups
            math variable: $E$
    :ivar I: infected compartment for age-risk groups
            math variable: $I$
    :ivar H: hospital compartment for age-risk groups
            math variable: $H$
    :ivar R: recovered compartment for age-risk groups
            math variable: $R$
    :ivar D: dead compartment for age-risk groups
            math variable: $D$
    :ivar population_immunity_hosp: infection-induced
        population-level immunity against hospitalization, for
        age-risk groups
            math variable: $M^H$
    :ivar population_immunity_inf: infection-induced
        population-level immunity against infection, for
        age-risk groups
            math variable: $M^I$
    """

    S: Optional[np.ndarray] = None
    E: Optional[np.ndarray] = None
    I: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    D: Optional[np.ndarray] = None
    population_immunity_hosp: Optional[np.ndarray] = None
    population_immunity_inf: Optional[np.ndarray] = None


class NewExposed(TransitionVariable):
    def get_current_rate(self, sim_state, epi_params):
        force_of_immunity = (1 + epi_params.inf_risk_reduct * sim_state.population_immunity_inf)
        return np.asarray(epi_params.beta * sim_state.I
                          / (epi_params.total_population_val * force_of_immunity))


class NewSusceptible(TransitionVariable):
    def get_current_rate(self, sim_state, epi_params):
        return np.full((epi_params.num_age_groups, epi_params.num_risk_groups), epi_params.R_to_S_rate)


class NewInfected(TransitionVariable):
    def get_current_rate(self, sim_state, epi_params):
        return np.full((epi_params.num_age_groups, epi_params.num_age_groups), epi_params.E_to_I_rate)


class NewRecoveredHome(TransitionVariable):
    def get_current_rate(self, sim_state, epi_params):
        return np.full((epi_params.num_age_groups, epi_params.num_risk_groups),
                       (1 - epi_params.I_to_H_adjusted_proportion) * epi_params.I_to_R_rate)


class NewRecoveredHosp(TransitionVariable):
    def get_current_rate(self, sim_state, epi_params):
        return np.full((epi_params.num_age_groups, epi_params.num_risk_groups),
                       (1 - epi_params.H_to_D_adjusted_proportion) * epi_params.H_to_R_rate)


class NewHosp(TransitionVariable):
    def get_current_rate(self, sim_state, epi_params):
        return np.asarray(epi_params.I_to_H_rate * epi_params.I_to_H_adjusted_proportion /
                          (1 + epi_params.hosp_risk_reduct * sim_state.population_immunity_hosp))


class NewDead(TransitionVariable):
    def get_current_rate(self, sim_state, epi_params):
        return np.asarray(epi_params.H_to_D_adjusted_proportion * epi_params.H_to_D_rate /
                          (1 + epi_params.death_risk_reduction * sim_state.population_immunity_hosp))


class PopulationImmunityHosp(StateVariable):
    def get_change_in_current_val(self, sim_state, epi_params, num_timesteps):
        immunity_gain = (epi_params.immunity_hosp_increase_factor * sim_state.R) / \
                        (epi_params.total_population_val *
                         (1 + epi_params.immunity_saturation_constant * sim_state.population_immunity_hosp))
        immunity_loss = epi_params.waning_factor_hosp * sim_state.population_immunity_hosp

        return np.asarray(immunity_gain - immunity_loss) / num_timesteps


class PopulationImmunityInf(StateVariable):
    def get_change_in_current_val(self, sim_state, epi_params, num_timesteps):
        immunity_gain = (epi_params.immunity_inf_increase_factor * sim_state.R) / \
                        (epi_params.total_population_val * (1 + epi_params.immunity_saturation_constant *
                                                            sim_state.population_immunity_inf))
        immunity_loss = epi_params.waning_factor_inf * sim_state.population_immunity_inf

        return np.asarray(immunity_gain - immunity_loss) / num_timesteps


class ImmunoSEIRSConstructor:
    """
    Class for creating ImmunoSEIRS model with predetermined fixed
    structure -- initial values and epidemiological structure are
    populated by user-specified JSON files.

    Key method create_transmission_model returns a TransmissionModel
    instance with S-E-I-H-R-D compartments and population_immunity_inf
    and population_immunity_hosp state variables. The structure
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

    The following are StateVariable instances:
        population_immunity_inf is a PopulationImmunityInf instance
        population_immunity_hosp is a PopulationImmunityHosp instance

    Transition rates and update formulas are specified in
        corresponding classes.

    Attributes
    ----------
    :ivar config: Config dataclass instance,
        holds configuration values
    :ivar epi_params: FluEpiParams dataclass instance,
        holds epidemiological parameter values, read-in
        from user-specified JSON
    :ivar sim_state: FluSimState dataclass instance,
        holds current simulation state information,
        such as current values of epidemiological compartments
        and state variables, read in from user-specified JSON
    :ivar transition_variable_lookup: dict,
        maps string to corresponding TransitionVariable
    :ivar transition_variable_group_lookup: dict,
        maps string to corresponding TransitionVariableGroup
    :ivar compartment_lookup: dict,
        maps string to corresponding EpiCompartment,
        using the value of the EpiCompartment's "name" attribute
    :ivar state_variable_lookup: dict,
        maps string to corresponding StateVariable,
        using the value of the StateVariable's "name" attribute
    """

    def __init__(self,
                 config_filepath,
                 epi_params_filepath,
                 epi_compartments_state_vars_init_vals_filepath):
        """
        Create Config, FluEpiParams, and FluSimState instances
        using values from respective JSON files, and save these instances
        on the ImmunoSEIRSConstructor to construct a model.

        The main method

        :param config_filepath: str,
            path to config JSON file (path includes actual filename
            with suffix ".json") -- all JSON fields must match
            name and datatype of Config instance attributes
        :param epi_params_filepath: str,
            path to epidemiological parameters JSON file
            (path includes actual filename with suffix ".json")
            -- all JSON fields must match name and datatype of
            EpiParams instance attributes
        :param epi_compartments_state_vars_init_vals_filepath: str,
            path to epidemiological compartments JSON file
            (path includes actual filename with suffix ".json")
            -- all JSON fields must match name and datatype of
            EpiCompartment instance attributes
        """
        self.config = dataclass_instance_from_json(Config, config_filepath)
        self.epi_params = dataclass_instance_from_json(FluEpiParams, epi_params_filepath)
        self.sim_state = dataclass_instance_from_json(FluSimState,
                                                      epi_compartments_state_vars_init_vals_filepath)

        self.transition_variable_lookup = {}
        self.transition_variable_group_lookup = {}
        self.compartment_lookup = {}
        self.state_variable_lookup = {}

    def setup_epi_compartments(self):
        """
        Create compartments S-E-I-H-R-D (6 compartments total)
        and add them to compartment_lookup for dictionary access
        """

        for name in ("S", "E", "I", "H", "R", "D"):
            self.compartment_lookup[name] = EpiCompartment(name, getattr(self.sim_state, name))

    def setup_transition_variables(self):
        """
        Create all transition variables described in docstring (7 transition
        variables total) and add them to transition_variable_lookup attribute
        for dictionary access
        """

        compartments = self.compartment_lookup
        transition_type = self.config.transition_type

        # Reordering the tuples to put the transition function first
        transition_mapping = {
            "new_susceptible": (NewSusceptible, compartments["R"], compartments["S"]),
            "new_exposed": (NewExposed, compartments["S"], compartments["E"]),
            "new_infected": (NewInfected, compartments["E"], compartments["I"]),
            "new_recovered_home": (NewRecoveredHome, compartments["I"], compartments["R"], True),
            "new_hosp": (NewHosp, compartments["I"], compartments["H"], True),
            "new_recovered_hosp": (NewRecoveredHosp, compartments["H"], compartments["R"], True),
            "new_dead": (NewDead, compartments["H"], compartments["D"], True)
        }

        # Create transition variables dynamically
        # params[0] is the TransitionVariable subclass (e.g. NewSusceptible)
        # params[1:3] refers to the origin compartment and destination compartment
        # params[3:] contains the Boolean indicating if the transition variable is jointly
        #   distributed (True if jointly distributed)
        self.transition_variable_lookup = {
            name: params[0](*params[1:3], transition_type, *params[3:])
            for name, params in transition_mapping.items()
        }

    def setup_transition_variable_groups(self):
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
            "I_out": TransitionVariableGroup(compartment_lookup["I"],
                                             transition_type,
                                             (tvar_lookup["new_recovered_home"],
                                              tvar_lookup["new_hosp"])),
            "H_out": TransitionVariableGroup(compartment_lookup["H"],
                                             transition_type,
                                             (tvar_lookup["new_recovered_hosp"],
                                              tvar_lookup["new_dead"]))
        }

    def setup_state_variables(self):
        """
        Create all state variable groups described in docstring (2 state
        variables total) and add them to state_variable_lookup attribute
        for dictionary access
        """

        self.state_variable_lookup["population_immunity_inf"] = \
            PopulationImmunityInf("population_immunity_inf",
                                  getattr(self.sim_state, "population_immunity_inf"))

        self.state_variable_lookup["population_immunity_hosp"] = \
            PopulationImmunityHosp("population_immunity_hosp",
                                   getattr(self.sim_state, "population_immunity_hosp"))

    def create_transmission_model(self, RNG_seed):
        """
        :param RNG_seed: int,
            used to initialize the model's RNG for generating
            random variables and random transitions
        :return: TransmissionModel instance,
            S-E-I-H-R-D model with 7 transition variables,
            2 transition variable groups, and 2 state variables
            for population-level immunity -- see class docstring
            for details -- initial values and epidemiological parameters
            are loaded from user-specified JSON files during
            ImmunoSEIRConstructor initialization.
        """

        # Setup objects for model
        self.setup_epi_compartments()
        self.setup_transition_variables()
        self.setup_transition_variable_groups()
        self.setup_state_variables()

        # Get dictionary values as lists to pass as TransmissionModel __init__ arguments
        flu_compartments = list(self.compartment_lookup.values())
        flu_transition_variables = list(self.transition_variable_lookup.values())
        flu_transition_variable_groups = list(self.transition_variable_group_lookup.values())
        flu_state_variables = list(self.state_variable_lookup.values())

        return TransmissionModel(flu_compartments,
                                 flu_transition_variables,
                                 flu_transition_variable_groups,
                                 flu_state_variables,
                                 self.sim_state,
                                 self.epi_params,
                                 self.config,
                                 RNG_seed)
