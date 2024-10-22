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

    Attributes
    ----------
    :ivar num_age_groups:
    :ivar num_risk_groups:
    :ivar beta:
    :ivar total_population_val:
    :ivar immunity_hosp_increase_factor:
    :ivar immunity_inf_increase_factor:
    :ivar immunity_hosp_saturation_constant:
    :ivar immunity_inf_saturation_constant:
    :ivar waning_factor_hosp:
    :ivar waning_factor_inf:
    :ivar efficacy_immunity_hosp:
    :ivar efficacy_immunity_inf:
    :ivar efficacy_immunity_death:
    :ivar R_to_S_rate:
    :ivar E_to_I_rate:
    :ivar I_to_R_rate:
    :ivar I_to_H_rate:
    :ivar H_to_R_rate:
    :ivar H_to_D_rate:
    :ivar I_to_H_adjusted_proportion:
    :ivar H_to_D_adjusted_proportion:
    """

    num_age_groups: Optional[int] = None
    num_risk_groups: Optional[int] = None
    beta: Optional[float] = None
    total_population_val: Optional[float] = None
    immunity_hosp_increase_factor: Optional[float] = None
    immunity_inf_increase_factor: Optional[float] = None
    immunity_hosp_saturation_constant: Optional[float] = None
    immunity_inf_saturation_constant: Optional[float] = None
    waning_factor_hosp: Optional[float] = None
    waning_factor_inf: Optional[float] = None
    efficacy_immunity_hosp: Optional[float] = None
    efficacy_immunity_inf: Optional[float] = None
    efficacy_immunity_death: Optional[float] = None
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

    Attributes
    ----------
    :ivar S:
    :ivar E:
    :ivar I:
    :ivar H:
    :ivar R:
    :ivar D:
    :ivar population_immunity_hosp:
    :ivar population_immunity_inf:
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
        force_of_immunity = (1 + epi_params.efficacy_immunity_inf * sim_state.population_immunity_inf)
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
                          (1 + epi_params.efficacy_immunity_hosp * sim_state.population_immunity_hosp))


class NewDead(TransitionVariable):
    def get_current_rate(self, sim_state, epi_params):
        return np.asarray(epi_params.H_to_D_adjusted_proportion * epi_params.H_to_D_rate /
                          (1 + epi_params.efficacy_immunity_death * sim_state.population_immunity_hosp))


class PopulationImmunityHosp(StateVariable):
    def get_change_in_current_val(self, sim_state, epi_params, num_timesteps):
        immunity_gain = (epi_params.immunity_hosp_increase_factor * sim_state.R) / \
                        (epi_params.total_population_val *
                         (1 + epi_params.immunity_hosp_saturation_constant * sim_state.population_immunity_hosp))
        immunity_loss = epi_params.waning_factor_hosp * sim_state.population_immunity_hosp

        return np.asarray(immunity_gain - immunity_loss) / num_timesteps


class PopulationImmunityInf(StateVariable):
    def get_change_in_current_val(self, sim_state, epi_params, num_timesteps):
        immunity_gain = (epi_params.immunity_inf_increase_factor * sim_state.R) / \
                        (epi_params.total_population_val * (1 + epi_params.immunity_inf_saturation_constant *
                                                            sim_state.population_immunity_inf))
        immunity_loss = epi_params.waning_factor_inf * sim_state.population_immunity_inf

        return np.asarray(immunity_gain - immunity_loss) / num_timesteps


class ImmunoSEIRSConstructor:
    """
    Class for creating ImmunoSEIRS model with predetermined fixed
    structure -- initial values and epidemiological structure are
    populated by user-specified json files.

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
        from user-specified .json
    :ivar sim_state: FluSimState dataclass instance,
        holds current simulation state information,
        such as current values of epidemiological compartments
        and state variables, read in from user-specified .json
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
        :param config_filepath: str,
            path to config json file (path includes actual filename
            with suffix ".json") -- all json fields must match
            name and datatype of Config instance attributes
        :param epi_params_filepath: str,
            path to epidemiological parameters json file
            (path includes actual filename with suffix ".json")
            -- all json fields must match name and datatype of
            EpiParams instance attributes
        :param epi_compartments_state_vars_init_vals_filepath: str,
            path to epidemiological compartments json file
            (path includes actual filename with suffix ".json")
            -- all json fields must match name and datatype of
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
        for name in ("S", "E", "I", "H", "R", "D"):
            self.compartment_lookup[name] = EpiCompartment(name, getattr(self.sim_state, name))

    def setup_transition_variables(self):
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
        self.transition_variable_lookup = {
            name: params[0](*params[1:3], transition_type, *params[3:])
            for name, params in transition_mapping.items()
        }

    def setup_transition_variable_groups(self):
        compartment_lookup = self.compartment_lookup
        tvar_lookup = self.transition_variable_lookup

        transition_type = self.config.transition_type

        self.transition_variable_group_lookup = {
            "I_out": TransitionVariableGroup(compartment_lookup["I"],
                                             transition_type,
                                             (tvar_lookup["new_recovered_home"], tvar_lookup["new_hosp"])),
            "H_out": TransitionVariableGroup(compartment_lookup["H"],
                                             transition_type,
                                             (tvar_lookup["new_recovered_hosp"], tvar_lookup["new_dead"]))
        }

    def setup_state_variables(self):
        self.state_variable_lookup["population_immunity_inf"] = \
            PopulationImmunityInf("population_immunity_inf", getattr(self.sim_state, "population_immunity_inf"))
        self.state_variable_lookup["population_immunity_hosp"] = \
            PopulationImmunityHosp("population_immunity_hosp", getattr(self.sim_state, "population_immunity_hosp"))

    def create_transmission_model(self, RNG_seed):
        self.setup_epi_compartments()
        self.setup_transition_variables()
        self.setup_transition_variable_groups()
        self.setup_state_variables()

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
