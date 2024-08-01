import numpy as np
import matplotlib.pyplot as plt
import time

from collections import namedtuple

class TransitionProbabilityDistributions:

    def compute_deterministic_realization(base_count, rate, RNG):
        return base_count * rate

    def compute_binomial_realization(base_count, rate, RNG):
        return RNG.binomial(n=int(base_count), p=rate)


class TransitionVariable:

    def __init__(self, name, transition_type, base_count_epi_compartment_name):
        self.name = name
        self.transition_type = transition_type
        self.base_count_epi_compartment_name = base_count_epi_compartment_name

        self.current_realization = 0
        self.current_rate = 0

        if transition_type == "deterministic":
            self.compute_realization = TransitionProbabilityDistributions.compute_deterministic_realization
        elif transition_type == "binomial":
            self.compute_realization = TransitionProbabilityDistributions.compute_binomial_realization


class EpiParams:

    def __init__(self):
        pass


class EpiCompartment:
    __slots__ = ("name", "initial_val", "inflow_varnames_list", "outflow_varnames_list", "current_val", "history_vals_list")

    def __init__(self, name, initial_val, inflow_varnames_list, outflow_varnames_list):
        self.name = name
        self.initial_val = initial_val

        self.inflow_varnames_list = inflow_varnames_list
        self.outflow_varnames_list = outflow_varnames_list

        self.current_val = initial_val  # current day's values

        self.history_vals_list = []  # historical values



class SimulationParams:

    def __init__(self, timesteps_per_day, starting_simulation_day=0):
        self.timesteps_per_day = timesteps_per_day
        self.starting_simulation_day = starting_simulation_day


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

        self.name_to_epi_compartment_dict = {}
        self.name_to_transition_variable_dict = {}

        for compartment in self.list_of_epi_compartments:
            self.name_to_epi_compartment_dict[compartment.name] = compartment
            setattr(self, compartment.name, compartment)

        for transition_variable in self.list_of_transition_variables:
            self.name_to_transition_variable_dict[transition_variable.name] = transition_variable
            setattr(self, transition_variable.name, transition_variable)

        self.bit_generator = np.random.MT19937(seed=RNG_seed)
        self.RNG = np.random.Generator(self.bit_generator)

        self.name_to_transition_rate_dict = {}

        self.current_day_counter = 0

    def reset(self):
        for compartment in self.list_of_epi_compartments:
            compartment.current_val = compartment.initial_val
            compartment.history_vals_list = []
        for transition_variable in self.list_of_transition_variables:
            transition_variable.current_rate = 0
            transition_variable.current_realization = 0
        self.current_day_counter = 0

    def update_history_vals_list(self):
        for compartment in self.list_of_epi_compartments:
            compartment.history_vals_list.append(compartment.current_val.copy())

    def simulate_until_time_period(self, last_simulation_day):

        # last_simulation_day is inclusive endpoint
        while self.current_day_counter < last_simulation_day + 1:
            self.simulate_discretized_timesteps()
            self.update_history_vals_list()

    def simulate_discretized_timesteps(self):

        for timestep in range(self.simulation_params.timesteps_per_day):

            self.update_discretized_rates()

            for transition_variable in self.list_of_transition_variables:
                transition_variable.current_realization = transition_variable.compute_realization(
                    getattr(self, transition_variable.base_count_epi_compartment_name).current_val,
                    transition_variable.current_rate,
                    self.RNG)

            for compartment in self.list_of_epi_compartments:
                total_inflow = 0
                total_outflow = 0
                for varname in compartment.inflow_varnames_list:
                    total_inflow += self.name_to_transition_variable_dict[varname].current_realization
                for varname in compartment.outflow_varnames_list:
                    total_outflow += self.name_to_transition_variable_dict[varname].current_realization
                compartment.current_val += (total_inflow - total_outflow)

        self.current_day_counter += 1

    def update_discretized_rates(self):

        pass


