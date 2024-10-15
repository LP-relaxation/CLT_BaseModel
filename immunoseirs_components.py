import numpy as np
import json
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, make_dataclass
from typing import Optional

from pathlib import Path

import matplotlib.pyplot as plt

from base_components import approx_binomial_probability_from_rate, \
    Config, TransitionVariableGroup, TransitionVariable, StateVariable, \
    EpiCompartment, TransmissionModel
from plotting import create_basic_compartment_history_plot


@dataclass
class EpiParams:
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
class SimState:
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
    def get_change_in_current_val(self, sim_state, epi_params):
        immunity_gain = (epi_params.immunity_hosp_increase_factor * sim_state.R) / \
                        (epi_params.total_population_val *
                         (1 + epi_params.immunity_hosp_saturation_constant * sim_state.population_immunity_hosp))
        immunity_loss = epi_params.waning_factor_hosp * sim_state.population_immunity_hosp

        return np.asarray(immunity_gain - immunity_loss)


class PopulationImmunityInf(StateVariable):
    def get_change_in_current_val(self, sim_state, epi_params):
        immunity_gain = (epi_params.immunity_inf_increase_factor * sim_state.R) / \
                        (epi_params.total_population_val * (1 + epi_params.immunity_inf_saturation_constant *
                                                            sim_state.population_immunity_inf))
        immunity_loss = epi_params.waning_factor_inf * sim_state.population_immunity_inf

        return np.asarray(immunity_gain - immunity_loss)


def dataclass_instance_from_json(dataclass_name, json_filepath):
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    # to numpy arrays to support numpy operations
    for key, val in data.items():
        if type(val) is list:
            data[key] = np.asarray(val)

    # breakpoint()

    return dataclass_name(**data)


class ImmunoSEIRSConstructor:

    def __init__(self,
                 config_filepath,
                 epi_params_filepath,
                 epi_compartments_state_vars_init_vals_filepath):
        self.config = dataclass_instance_from_json(Config, config_filepath)
        self.epi_params = dataclass_instance_from_json(EpiParams, epi_params_filepath)
        self.sim_state = dataclass_instance_from_json(SimState, epi_compartments_state_vars_init_vals_filepath)

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

        immunoseirs_compartments = list(self.compartment_lookup.values())
        immunoseirs_transition_variables = list(self.transition_variable_lookup.values())
        immunoseirs_transition_variable_groups = list(self.transition_variable_group_lookup.values())
        immunoseirs_state_variables = list(self.state_variable_lookup.values())

        return TransmissionModel(immunoseirs_compartments,
                                 immunoseirs_transition_variables,
                                 immunoseirs_transition_variable_groups,
                                 immunoseirs_state_variables,
                                 self.sim_state,
                                 self.epi_params,
                                 self.config,
                                 RNG_seed)

base_path = Path(__file__).parent / "instance1_1age_1risk_test"

immunoseirs_config_filepath = base_path / "config.json"
immunoseirs_epi_params_filepath = base_path / "epi_params.json"
immunoseirs_epi_compartments_state_vars_init_vals_filepath = base_path / "epi_compartments_state_vars_init_vals.json"

immunoseirs_constructor = ImmunoSEIRSConstructor(immunoseirs_config_filepath,
                                                 immunoseirs_epi_params_filepath,
                                                 immunoseirs_epi_compartments_state_vars_init_vals_filepath)

immunoseirs_model = immunoseirs_constructor.create_transmission_model(np.random.SeedSequence())

import time

start = time.time()

immunoseirs_model.simulate_until_time_period(365)

print(time.time() - start)

create_basic_compartment_history_plot(immunoseirs_model)


