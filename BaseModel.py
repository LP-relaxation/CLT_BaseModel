import numpy as np
import matplotlib.pyplot as plt
import time

from collections import namedtuple


def compute_deterministic_realization(base_count, rate):
    return base_count * rate


def compute_binomial_realization(base_count, rate):
    return


class TransitionVariable:

    def __init__(self, name, transition_type):
        self.name = name
        self.transition_type = transition_type
        self.compute_realization = None

        if transition_type == "deterministic":
            self.compute_realization = compute_deterministic_realization
        elif transition_type == "binomial":
            self.compute_realization = compute_binomial_realization


class EpiCompartment:
    __slots__ = ("name", "initial_val", "current_day_val",
                 "current_timestep_val", "previous_day_val",
                 "history_vals_list")

    def __init__(self, name, initial_val):
        self.name = name
        self.initial_val = initial_val

        self.current_day_val = initial_val  # current day's values
        self.current_timestep_val = initial_val
        self.previous_day_val = []  # previous day's values

        self.history_vals_list = []  # historical values

    varnames_dict = {}


class SimulationParams:

    def __init__(self, timesteps_per_day, starting_simulation_day=0):
        self.timesteps_per_day = timesteps_per_day
        self.starting_simulation_day = starting_simulation_day


class ImmunityTracker:

    def __init__(self, current_variant_prevalence):
        self.current_variant_prevalence = current_variant_prevalence


class BaseModel:

    def __init__(self,
                 list_of_epi_compartments,
                 list_of_transition_variables,
                 epi_params,
                 simulation_params,
                 RNG_seed=50505050):

        self.list_of_epi_compartments = list_of_epi_compartments
        self.list_of_transition_variables = list_of_transition_variables

        self.epi_params = epi_params
        self.simulation_params = simulation_params

        self.current_simulation_day = simulation_params.starting_simulation_day

        self.name_to_compartment_dict = {}

        for compartment in self.list_of_epi_compartments:
            self.name_to_compartment_dict[compartment.name] = compartment
            setattr(self, compartment.name, compartment)

        self.bit_generator = np.random.MT19937(seed=RNG_seed)
        self.RNG = np.random.Generator(self.bit_generator)

    def set_previous_day_vals_to_current_day_vals(self):
        for compartment in self.name_to_compartment_dict.values():
            compartment.previous_day_val = compartment.current_day_val

    def update_current_day_vals(self):
        for compartment in self.name_to_compartment_dict.values():
            compartment.current_day_val = compartment.current_timestep_val

    def update_history_vals_list(self):
        for compartment in self.name_to_compartment_dict.values():
            compartment.history_vals_list.append(compartment.current_day_val)

    def create_compartment_attribute_dict(self, key_string, value_string):
        d = {}
        for object in self.list_of_epi_compartments:
            d[getattr(object, key_string)] = getattr(object, value_string)
        return d

    def update_compartments_from_dict(self, name_to_val_dict, attribute_to_update):
        for name, val in name_to_val_dict.items():
            setattr(self.name_to_compartment_dict[name], attribute_to_update, name)

    def simulate_until_time_period(self, last_simulation_day):
        # last_simulation_day is inclusive endpoint
        for day in range(last_simulation_day + 1):
            self.simulate_next_day()

    def simulate_next_day(self):

        # start_time = time.time()

        self.set_previous_day_vals_to_current_day_vals()

        self.simulate_discretized_timesteps()

        self.update_current_day_vals()
        self.update_history_vals_list()

        # print(time.time() - start_time)

    def simulate_discretized_timesteps(self):

        pass


