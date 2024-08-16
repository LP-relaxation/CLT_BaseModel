import numpy as np
import matplotlib.pyplot as plt
import time

from collections import namedtuple


def approx_binomial_probability_from_rate(rate, time_interval_length):

    '''
    Converts a rate (events per time) to the probability of any event
        occurring in the next time period of time_interval_length,
        assuming the number of events occurring in time_interval_length
        follows a Poisson distribution with given rate parameter

    The probability of 0 events in time_interval_length is
        e^(-rate * time_interval_length), so the probability of any event
        in time_interval_length is 1 - e^(-rate * time_interval_length)

    :param rate: positive scalar,
        rate parameter in a Poisson distribution
    :param time_interval_length: positive scalar,
        time interval over which event may occur
    :return: positive scalar in (0,1)
    '''

    return 1 - np.exp(-rate * time_interval_length)


class TransitionProbabilityDistributions:
    '''
    Container to hold methods for computing deterministic and random
        transition realizations between compartments. All such methods are class
        methods because we do not create instances of this class.
    '''

    @classmethod
    def compute_deterministic_realization(cls, base_count, rate, num_timesteps, RNG=None):
        return base_count * rate / num_timesteps

    @classmethod
    def compute_binomial_realization(cls, base_count, rate, num_timesteps, RNG):
        return RNG.binomial(n=int(base_count),
                            p=approx_binomial_probability_from_rate(rate, 1/num_timesteps))

    @classmethod
    def compute_multinomial_realization(cls, base_count, probabilities_list, RNG):
        return RNG.multinomial(int(base_count), probabilities_list)


class OutgoingTransitionVariableGroup:

    def __init__(self,
                 name,
                 base_count_epi_compartment,
                 transition_variables_list):

        self.name = name
        self.base_count_epi_compartment = base_count_epi_compartment
        self.transition_variables_list = transition_variables_list

    def compute_total_rate(self):

        total_rate = 0

        for transition_variable in self.transition_variables_list:
            total_rate += transition_variable.current_rate

        return total_rate

    def compute_realization(self, timesteps_per_day, RNG):

        total_rate = self.compute_total_rate()

        total_outgoing_probability = approx_binomial_probability_from_rate(total_rate, 1/timesteps_per_day)

        probabilities_list = []

        for transition_variable in self.transition_variables_list:
            probabilities_list.append((transition_variable.current_rate / total_rate) * total_outgoing_probability)

        probabilities_list.append(1 - total_outgoing_probability)
        probabilities_list = np.array(probabilities_list).flatten()

        base_count = self.base_count_epi_compartment.current_val

        multinomial_realizations_list = TransitionProbabilityDistributions.compute_multinomial_realization(base_count,
                                                                                                           probabilities_list,
                                                                                                           RNG)

        for ix in range(len(self.transition_variables_list)):
            self.transition_variables_list[ix].current_realization = multinomial_realizations_list[ix]


class TransitionVariable:

    def __init__(self,
                 name,
                 transition_type,
                 base_count_epi_compartment,
                 is_jointly_distributed=False):

        __slots__ = ("name",
                     "transition_type",
                     "base_count_epi_compartment",
                     "is_jointly_distributed")

        self.name = name
        self.base_count_epi_compartment = base_count_epi_compartment

        self.current_realization = None
        self.current_rate = None

        self.is_jointly_distributed = is_jointly_distributed

        if self.is_jointly_distributed:
            pass
        else:
            if transition_type == "deterministic":
                self.compute_realization = TransitionProbabilityDistributions.compute_deterministic_realization
            elif transition_type == "binomial":
                self.compute_realization = TransitionProbabilityDistributions.compute_binomial_realization

    @property
    def base_count(self):
        return self.base_count_epi_compartment.current_val

    def reset(self):
        self.current_realization = None
        self.current_rate = None


class EpiParams:

    def __init__(self):
        pass


class EpiCompartment:
    __slots__ = ("name",
                 "initial_val",
                 "inflow_t_var_names_list",
                 "outflow_t_var_names_list",
                 "current_val",
                 "history_vals_list",
                 "is_population_compartment")

    def __init__(self,
                 name,
                 initial_val,
                 inflow_t_var_names_list,
                 outflow_t_var_names_list,
                 is_population_compartment=True):
        self.name = name
        self.initial_val = initial_val

        self.inflow_t_var_names_list = inflow_t_var_names_list
        self.outflow_t_var_names_list = outflow_t_var_names_list

        self.current_val = initial_val  # current day's values

        self.history_vals_list = []  # historical values

        self.is_population_compartment = is_population_compartment


class SimulationParams:

    def __init__(self, timesteps_per_day, starting_simulation_day=0):
        self.timesteps_per_day = timesteps_per_day
        self.starting_simulation_day = starting_simulation_day


class BaseModel:

    def __init__(self,
                 epi_params,
                 simulation_params,
                 transition_type="deterministic",
                 RNG_seed=np.random.SeedSequence()):

        self.epi_params = epi_params
        self.simulation_params = simulation_params
        self.transition_type = transition_type

        self.current_simulation_day = simulation_params.starting_simulation_day

        self.name_to_epi_compartment_dict = {}
        self.name_to_t_var_dict = {}
        self.name_to_outgoing_t_var_group_dict = {}

        self.bit_generator = np.random.MT19937(seed=RNG_seed)
        self.RNG = np.random.Generator(self.bit_generator)

        self.current_day_counter = 0

    def add_epi_compartment(self,
                            name,
                            init_val,
                            incoming_tvar_names_list,
                            outgoing_tvar_names_list,
                            is_population_compartment=True):

        compartment = EpiCompartment(name,
                                     init_val,
                                     incoming_tvar_names_list,
                                     outgoing_tvar_names_list,
                                     is_population_compartment)

        self.name_to_epi_compartment_dict[name] = compartment
        setattr(self, name, compartment)

    def add_transition_variable(self,
                                name,
                                transition_type,
                                base_count_epi_compartment,
                                is_jointly_distributed=False):

        transition_variable = TransitionVariable(name,
                                                 transition_type,
                                                 base_count_epi_compartment,
                                                 is_jointly_distributed)

        self.name_to_t_var_dict[name] = transition_variable
        setattr(self, name, transition_variable)

    def add_outgoing_transition_variable_group(self,
                                               name,
                                               base_count_epi_compartment,
                                               transition_variable_list):

        transition_variable_group = OutgoingTransitionVariableGroup(name,
                                                                    base_count_epi_compartment,
                                                                    transition_variable_list)

        self.name_to_outgoing_t_var_group_dict[name] = transition_variable_group
        setattr(self, name, transition_variable_group)

    def reset(self):

        for compartment in self.epi_compartments_list:
            compartment.current_val = compartment.initial_val
            compartment.history_vals_list = []
        for transition_variable in self.transition_variables_list:
            transition_variable.current_rate = None
            transition_variable.current_realization = None
        self.current_day_counter = 0

    def update_history_vals_list(self):

        for compartment in self.name_to_epi_compartment_dict.values():
            compartment.history_vals_list.append(compartment.current_val.copy())

    def simulate_until_time_period(self, last_simulation_day):

        # last_simulation_day is inclusive endpoint
        while self.current_day_counter < last_simulation_day + 1:
            self.simulate_discretized_timesteps()
            self.update_history_vals_list()

    def simulate_discretized_timesteps(self):

        timesteps_per_day = self.simulation_params.timesteps_per_day

        for timestep in range(timesteps_per_day):

            self.update_discretized_rates()

            for t_var_group in self.name_to_outgoing_t_var_group_dict.values():
                t_var_group.compute_realization(timesteps_per_day,
                                                self.RNG)

            for transition_variable in self.name_to_t_var_dict.values():
                if transition_variable.is_jointly_distributed:
                    continue
                else:
                    transition_variable.current_realization = transition_variable.compute_realization(
                       transition_variable.base_count,
                       transition_variable.current_rate,
                       timesteps_per_day,
                       self.RNG)

            for compartment in self.name_to_epi_compartment_dict.values():
                total_inflow = 0
                total_outflow = 0
                for varname in compartment.inflow_t_var_names_list:
                    total_inflow += self.name_to_t_var_dict[varname].current_realization
                for varname in compartment.outflow_t_var_names_list:
                    total_outflow += self.name_to_t_var_dict[varname].current_realization
                compartment.current_val += (total_inflow - total_outflow)

            for transition_variable in self.name_to_t_var_dict.values():
                transition_variable.reset()

        self.current_day_counter += 1

    def update_discretized_rates(self):

        pass
