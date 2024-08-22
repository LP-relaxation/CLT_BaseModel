import numpy as np
import matplotlib.pyplot as plt
import time
import json

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
    def compute_deterministic_binomial_realization(cls, base_count, rate, num_timesteps, RNG=None):
        '''
        Deterministically computes number transitioning between
            EpiCompartments in next timestep. Rather than sampling binomial
            random variable, returns mean value base_count * p(rate / num_timesteps)
            with element-wise multiplication, where p(rate / num_timesteps) converts
            rate to a probability using a Poisson approximation.

        :param base_count: array_like of positive integers,
            corresponding to population counts in an EpiCompartment instance

        :param rate: array_like of positive scalars, same shape as base_count
            corresponding to rate (events per time) at which people transition
            between compartments

        :param num_timesteps: positive integer,
            number of timesteps per day in the simulation

        :param RNG: numpy Generator instance,
            argument not used -- only included to create same function
            signature as compute_binomial_realization

        :return: array_like of positive scalars, same shape as base_count and rate,
            returns mean value base_count * p(rate / num_timesteps) with
            element-wise multiplication, where p(rate / num_timesteps)
            converts rate to a probability using a Poisson approximation
        '''

        return base_count * rate / num_timesteps

    @classmethod
    def compute_binomial_realization(cls, base_count, rate, num_timesteps, RNG):
        '''
        Using RNG for random number generation, stochastically computes
            number transitioning between EpiCompartments in next timestep
            according to binomial distribution with parameters given by base_count
            and p(rate / num_timesteps), which converts rate to a probability using
            a Poisson approximation.

        :param base_count: array_like of positive integers,
            corresponding to population counts in an EpiCompartment instance

        :param rate: array_like of positive scalars, same shape as base_count
            corresponding to rate (events per time) at which people transition
            between compartments

        :param num_timesteps: positive integer,
            number of timesteps per day in the simulation

        :param RNG: numpy Generator instance,
            manages random bits to generate random variables

        :return: array_like of positive scalars, same shape as base_count and rate,
            returns realization of Binomial random variable with parameters
            base_count and p(rate / num_timesteps), where p(rate / num_timesteps)
            converts rate to a probability using a Poisson approximation
        '''

        return RNG.binomial(n=int(base_count),
                            p=approx_binomial_probability_from_rate(rate, 1 / num_timesteps))

    @classmethod
    def compute_deterministic_multinomial_realization(cls, base_count, probabilities_list, RNG=None):
        '''
        Deterministically computes number transitioning between
            EpiCompartments in next timestep. Rather than sampling multinomial
            random variable, returns mean value base_count * probabilities_list
            with element-wise multiplication.

        :param base_count: array_like of positive integers,
            corresponding to population counts in an EpiCompartment instance

        :param probabilities_list: array-like of numbers in (0,1), must sum to 1,
            each element i corresponds to probability of outcome i for each random
            trial

        :param RNG: numpy Generator instance,
            argument not used -- only included to create same function
            signature as compute_multinomial_realization

        :return: array_like of positive scalars, same shape as base_count and rate,
            returns realization of multinomial random variable with parameters
            base_count and probabilities_list
        '''

        return np.array(base_count * np.array(probabilities_list), dtype=int)

    @classmethod
    def compute_multinomial_realization(cls, base_count, probabilities_list, RNG):
        '''
        Using RNG for random number generation, stochastically computes
            number transitioning between EpiCompartments in next timestep
            according to multinomial distribution with parameters given by base_count
            and probabilities_list.

        :param base_count: array_like of positive integers,
            corresponding to population counts in an EpiCompartment instance

        :param probabilities_list: array-like of numbers in (0,1), must sum to 1,
            each element i corresponds to probability of outcome i for each random
            trial

        :param RNG: numpy Generator instance,
            manages random bits to generate random variables

        :return: array_like of positive scalars, same shape as base_count and rate,
            returns realization of multinomial random variable with parameters
            base_count and probabilities_list
        '''

        return RNG.multinomial(int(base_count), probabilities_list)


class TransitionVariableGroup:
    '''
    Container for TransitionVariable objects to handle joint sampling,
        when there are multiple outflows from a single compartment

    For example, if all outflows of compartment H are: R and D,
        i.e. from the Hospital compartment, people either go to Recovered
        or Dead, a TransitionVariableGroup that holds both R and D handles
        the correct correlation structure between R and D
    '''

    def __init__(self,
                 name,
                 base_count_epi_compartment,
                 transition_variables_list,
                 is_stochastic):
        '''
        :param name: string,
            name of transition variable group
        :param base_count_epi_compartment: EpiCompartment object,
            corresponds to common EpiCompartment that transitions
            variables in transition_variable_list outflow from
        :param transition_variable_list: array-like of TransitionVariable objects,
            holds all TransitionVariable objects that outflow from
            base_count_epi_compartment
        :param is_stochastic: Boolean,
            True if binomial / multinomial random transitions are used,
            False if the deterministic means of the binomial / multinomial
            distributions are used instead of random sampling
        '''
        self.name = name
        self.is_stochastic = is_stochastic
        self.base_count_epi_compartment = base_count_epi_compartment
        self.transition_variables_list = transition_variables_list

    def compute_total_rate(self):
        '''
        :return: array-like of positive numbers,
            sum of current rates of transition variables in group
        '''

        total_rate = 0

        for transition_variable in self.transition_variables_list:
            total_rate += transition_variable.current_rate

        return total_rate

    def compute_realization(self, timesteps_per_day, RNG):

        '''
        Compute jointly distributed transition variable outflows, scaled by
            timesteps_per_day, using multinomial distribution and
            random numbers from RNG
        After sampling, updates current_realization attribute on all
            TransitionVariable instances contained in this
            transition variable group

        :param timesteps_per_day: positive integer,
            number of discretized steps per simulation day
        :param RNG: numpy Generator object,
            used to generate multinomial random variables
        '''

        # Total rate at which people leave the base compartment
        total_rate = self.compute_total_rate()

        # Convert the total rate into a total outgoing probability, scaled by the timesteps per day
        total_outgoing_probability = approx_binomial_probability_from_rate(total_rate, 1 / timesteps_per_day)

        probabilities_list = []

        # Create probabilities_list, where element i corresponds to the
        #   transition variable i's current rate divided by the total rate,
        #   multiplied by the total outgoing probability
        # This generates the probabilities array that parameterizes the
        #   multinomial distribution
        for transition_variable in self.transition_variables_list:
            probabilities_list.append((transition_variable.current_rate / total_rate) * total_outgoing_probability)

        # Append the probability that a person stays in the compartment
        probabilities_list.append(1 - total_outgoing_probability)
        probabilities_list = np.array(probabilities_list).flatten()

        base_count = self.base_count_epi_compartment.current_val

        # Sample from the multinomial distribution
        if self.is_stochastic:
            multinomial_realizations_list = \
                TransitionProbabilityDistributions.compute_multinomial_realization(base_count,
                                                                                   probabilities_list,
                                                                                   RNG)
        else:
            multinomial_realizations_list = \
                TransitionProbabilityDistributions.compute_deterministic_multinomial_realization(base_count,
                                                                                                 probabilities_list,
                                                                                                 RNG)

        # Since the ith element in probabilities_list corresponds to the ith transition variable
        #   in transition_variables_list, the ith element in multinomial_realizations_list
        #   also corresponds to the ith transition variable in transition_variables_list
        # Update the current realization of the transition variables contained in this group
        for ix in range(len(self.transition_variables_list)):
            self.transition_variables_list[ix].current_realization = multinomial_realizations_list[ix]


class TransitionVariable:

    def __init__(self,
                 name,
                 base_count_epi_compartment,
                 is_stochastic=True,
                 is_jointly_distributed=False):
        '''
        :param name: string,
            name of TransitionVariable
        :param base_count_epi_compartment: EpiCompartment object,
            compartment from which TransitionVariable flows
        :param is_stochastic: Boolean,
            True if binomial / multinomial random transitions are used,
            False if the deterministic means of the binomial / multinomial
            distributions are used instead of random sampling
        :param is_jointly_distributed: Boolean,
            True if there are multiple TransitionVariable outflows
            from the same base_count_epi_compartment and
            random sampling must be handled jointly (then this
            TransitionVariable instance is also contained in a
            TransitionVariableGroup). False otherwise, in which case
            random sampling is marginal.
        '''

        __slots__ = ("name",
                     "base_count_epi_compartment",
                     "is_stochastic",
                     "is_jointly_distributed")

        self.name = name
        self.base_count_epi_compartment = base_count_epi_compartment

        self.current_realization = None
        self.current_rate = None

        self.is_jointly_distributed = is_jointly_distributed

        if self.is_jointly_distributed:
            pass
        else:
            if is_stochastic:
                self.compute_realization = TransitionProbabilityDistributions.compute_binomial_realization
            else:
                self.compute_realization = TransitionProbabilityDistributions.compute_deterministic_binomial_realization

    @property
    def base_count(self):
        return self.base_count_epi_compartment.current_val

    def reset(self):
        self.current_realization = None
        self.current_rate = None


class EpiParams:
    '''
    Container for epidemiological parameters.
    '''

    def __init__(self):
        pass


class EpiCompartment:
    '''
    Population compartment in a compartmental epidemiological model.

    For example, in an S-E-I-R model, we would have an EpiCompartment
        instance for each compartment.
    '''

    __slots__ = ("name",
                 "init_val",
                 "current_val",
                 "inflow_transition_variable_names_list",
                 "outflow_transition_variable_names_list",
                 "history_vals_list")

    def __init__(self,
                 name,
                 init_val,
                 inflow_transition_variable_names_list,
                 outflow_transition_variable_names_list):
        '''
        :param name: string,
            name of EpiCompartment
        :param init_val: array-like of positive integers,
            starting population for EpiCompartment
        :param incoming_transition_variable_names_list: array-like of TransitionVariable objects,
            list of TransitionVariable objects that inflow to epi compartment
        :param outgoing_transition_variable_names_list: array-like of TransitionVariable objects,
            list of TransitionVariable objects that outflow from epi compartment
        '''

        self.name = name
        self.init_val = init_val
        self.current_val = init_val.copy()

        self.inflow_transition_variable_names_list = inflow_transition_variable_names_list
        self.outflow_transition_variable_names_list = outflow_transition_variable_names_list

        self.history_vals_list = []  # historical values


class StateVariable:
    '''
    Class for variables that are deterministic functions of
        the simulation state (meaning epi compartment values, other parameters,
        and time.)

    For example, population-level immunity against hospitalization and
        infection can both be modeled as a StateVariable instance.
    '''

    def __init__(self,
                 name,
                 init_val):
        self.name = name
        self.init_val = init_val
        self.current_val = init_val.copy()
        self.change_in_current_val = None


class SimulationParams:

    def __init__(self, timesteps_per_day, starting_simulation_day=0):
        self.timesteps_per_day = timesteps_per_day
        self.starting_simulation_day = starting_simulation_day


class BaseModel:

    def __init__(self,
                 is_stochastic=True,
                 RNG_seed=np.random.SeedSequence()):

        self.is_stochastic = is_stochastic

        self.name_to_epi_compartment_dict = {}
        self.name_to_transition_variable_dict = {}
        self.name_to_transition_variable_group_dict = {}
        self.name_to_state_variable_dict = {}

        self.bit_generator = np.random.MT19937(seed=RNG_seed)
        self.RNG = np.random.Generator(self.bit_generator)

        self.current_day_counter = 0

        self.epi_params = None
        self.simulation_params = None

    def add_epi_compartment(self,
                            name,
                            init_val,
                            incoming_transition_variable_names_list,
                            outgoing_transition_variable_names_list):
        '''
        Constructor method for adding EpiCompartment instance to model
        Automatically constructs TransitionVariable instances and
            TransitionVariableGroup instances based on
            incoming_transition_variable_names_list and
            outgoing_transition_variable_names_list

        Creates model attribute named "name" that is assigned to
            new EpiCompartment instance
        Updates name_to_epi_compartment_dict attribute by adding
            (name, EpiCompartment) as (key, value) pair
        For each variable name in incoming_transition_variable_names_list
            and outgoing_transition_variable_names_list, creates a
            corresponding TransitionVariable if one does not already exist
            and assigns it as a model attribute
        If outgoing_Transition_variable_names_list is longer than 1,
            then multiple outflows exist and a TransitionVariableGroup is
            created and also assigned as a model attribute with
            the automatic name "name" [name of epi compartment] + "_out"

        For example, if all outgoing transition variables from H are:
            "H_to_D" and "H_to_R", then these transition variables are
            automatically created, and a transition variable group
            named "H_out" is automatically created

        See EpiCompartment class (__init__ function) for parameters.
        '''

        epi_compartment = EpiCompartment(name,
                                         init_val,
                                         incoming_transition_variable_names_list,
                                         outgoing_transition_variable_names_list)

        self.name_to_epi_compartment_dict[name] = epi_compartment
        setattr(self, name, epi_compartment)

        # If there are multiple outflows from this epi compartment,
        #   create corresponding TransitionVariables where is_jointly_distributed = True
        #   and create a TransitionVariableGroup instance to hold these transition variables
        if len(outgoing_transition_variable_names_list) > 1:
            for transition_variable_name in outgoing_transition_variable_names_list:
                self.add_transition_variable(transition_variable_name,
                                             epi_compartment,
                                             self.is_stochastic,
                                             True)

            # Create a list of transition variable objects to create a TransitionVariableGroup
            transition_variables_list = [
                self.name_to_transition_variable_dict[vname] for vname in outgoing_transition_variable_names_list]

            # Create a new TransitionVariableGroup with the name "name" [name of epi compartment] + "_out"
            self.add_transition_variable_group(name + "_out",
                                               epi_compartment,
                                               transition_variables_list,
                                               self.is_stochastic)
        else:
            for transition_variable_name in outgoing_transition_variable_names_list:
                self.add_transition_variable(transition_variable_name,
                                             epi_compartment,
                                             self.is_stochastic,
                                             False)

    def add_transition_variable(self,
                                name,
                                transition_type,
                                base_count_epi_compartment,
                                is_jointly_distributed=False):
        '''
        Constructor method for adding TransitionVariable instance to model

        Creates model attribute named "name" that is assigned to
            new TransitionVariable instance
        Updates name_to_transition_variable_dict attribute by adding
            (name, TransitionVariable) as (key, value) pair

        For example, if all outflows of compartment H are: R and D,
            i.e. from the Hospital compartment, people either go to Recovered
            or Dead, then TransitionVariable instances R and D have
            attribute is_jointly_distributed = True

        See TransitionVariable class (__init__ function) for parameters.
        '''

        transition_variable = TransitionVariable(name,
                                                 transition_type,
                                                 base_count_epi_compartment,
                                                 is_jointly_distributed)

        self.name_to_transition_variable_dict[name] = transition_variable
        setattr(self, name, transition_variable)

    def add_transition_variable_group(self,
                                      name,
                                      base_count_epi_compartment,
                                      transition_variable_list,
                                      is_stochastic):
        '''
        Constructor method for adding TransitionVariableGroup instance to model

        Creates model attribute named "name" that is assigned to
            new TransitionVariableGroup instance
        Updates name_to_transition_variable_group_dict attribute by adding
            (name, TransitionVariableGroup instance) as new (key, value) pair

        For example, if all outflows of compartment H are: R and D,
             i.e. from the Hospital compartment, people either go to Recovered
             or Dead, then base_count_epi_compartment is H and
             transition_variable_list is [R, D], where H, R, D are all
             EpiCompartment objects

        See TransitionVariableGroup class (__init__ function) for parameters.
        '''

        transition_variable_group = TransitionVariableGroup(name,
                                                            base_count_epi_compartment,
                                                            transition_variable_list,
                                                            is_stochastic)

        self.name_to_transition_variable_group_dict[name] = transition_variable_group
        setattr(self, name, transition_variable_group)

    def add_state_variable(self,
                           name,
                           init_val):
        '''
        Constructor method for adding StateVariable instance to model

        Creates model attribute named "name" that is assigned to
            new StateVariable instance
        Updates name_to_state_variable_dict attribute by adding
            (name, StateVariable instance) as new (key, value) pair

        :param name: string,
            name of state variable
        :param init_val: array-like,
            initial value of state variable at simulation start
        '''

        state_variable = StateVariable(name, init_val)

        self.name_to_state_variable_dict[name] = state_variable
        setattr(self, name, state_variable)

    def add_epi_params_from_json(self, json_filename):
        '''
        Assign epi_params attribute to EpiParams instance
        Load json_filename and assign values to epi_params attribute
        '''
        self.epi_params = EpiParams()

        with open(json_filename) as f:
            d = json.load(f)

        for key, value in d.items():
            setattr(self.epi_params, key, value)

    def add_epi_params(self, epi_params):
        self.epi_params = epi_params

    def add_simulation_params(self, simulation_params):
        self.simulation_params = simulation_params

    def reset(self):
        '''
        Reset each compartment, state variable, and transition variable
            to default/starting values
        Reset current_day_counter to 0

        For each EpiCompartment instance, set current_val to init_val
            and reset history_vals_list to []
        For each StateVariable instance, set current_val to init_val
            and change_in_current_val to None
        For each TransitionVariable instance, set current_rate to None
            and current_realization to None
        "Rewind" simulation time to 0 by setting current_day_counter to 0

        Note that TransitionVariableGroup do not store/possess their own
            current values or rates -- they are simply containers for
            TransitionVariables, and so TransitionVariableGroup instances
            do not need to be reset
        '''

        for compartment in self.name_to_epi_compartment_dict.values():
            compartment.current_val = compartment.init_val
            compartment.history_vals_list = []

        for state_variable in self.name_to_state_variable_dict.values():
            state_variable.current_val = state_variable.init_val
            state_variable.change_in_current_val = None

        for transition_variable in self.name_to_transition_variable_dict.values():
            transition_variable.current_rate = None
            transition_variable.current_realization = None

        self.current_day_counter = 0

    def update_history_vals_list(self):
        '''
        For each EpiCompartment instance attached to current model,
            append copy of compartment's current_val to
            compartment's history_vals_list
        '''

        for compartment in self.name_to_epi_compartment_dict.values():
            compartment.history_vals_list.append(compartment.current_val.copy())

    def simulate_until_time_period(self, last_simulation_day):
        '''
        Advance simulation model time until last_simulation_day
        Advance time by iterating through simulation days,
            which are simulated by iterating through discretized
            timesteps
        Save daily simulation data as history

        :param last_simulation_day: positive integer,
            stop simulation at last_simulation_day (i.e. exclusive,
            simulate up to but not including last_simulation_day)
        '''

        # last_simulation_day is exclusive endpoint
        while self.current_day_counter < last_simulation_day:
            self.simulate_discretized_timesteps()
            self.update_history_vals_list()

    def simulate_discretized_timesteps(self):
        '''
        Subroutine for simulate_until_time_period
        Iterates through discretized timesteps to simulate next
            simulation day. Granularity of discretization is given by
            attribute simulation_params.timesteps_per_day
        Properly scales transition variable realizations and changes in state
            variables by specified timesteps per day
        '''

        # Attribute lookup shortcut
        timesteps_per_day = self.simulation_params.timesteps_per_day

        for timestep in range(timesteps_per_day):

            self.compute_change_in_state_variables()

            # Users will inherit from base class BaseModel and override
            #   these functions to customize their model
            self.update_state_variables()
            self.compute_transition_rates()

            # Obtain transition variable realizations for jointly distributed transition variables
            #   (i.e. when there are multiple transition variable otuflows from an epi compartment)
            for transition_variable_group in self.name_to_transition_variable_group_dict.values():
                transition_variable_group.compute_realization(timesteps_per_day, self.RNG)

            # Obtain transition variable realizations for marginally distributed transition variables
            #   (i.e. when there is only one transition variable outflow from an epi compartment)
            # If transition variable is jointly distributed, then its realization has already
            #   been computed by its transition variable group container previously, so skip it
            for transition_variable in self.name_to_transition_variable_dict.values():
                if transition_variable.is_jointly_distributed:
                    continue
                else:
                    transition_variable.current_realization = transition_variable.compute_realization(
                        transition_variable.base_count,
                        transition_variable.current_rate,
                        timesteps_per_day,
                        self.RNG)

            # Update all epi compartments
            for compartment in self.name_to_epi_compartment_dict.values():
                total_inflow = 0
                total_outflow = 0
                for varname in compartment.inflow_transition_variable_names_list:
                    total_inflow += self.name_to_transition_variable_dict[varname].current_realization
                for varname in compartment.outflow_transition_variable_names_list:
                    total_outflow += self.name_to_transition_variable_dict[varname].current_realization
                compartment.current_val += (total_inflow - total_outflow)
                # print(compartment.name, compartment.current_val)

            # Reset transition variable current_realization and current_rate to None
            for tvar in self.name_to_transition_variable_dict.values():
                tvar.reset()

        # Move to next day in simulation
        self.current_day_counter += 1

    def update_state_variables(self):
        '''
        Helper function to update state variables with proper scale
            according to timesteps per day
        '''

        timesteps_per_day = self.simulation_params.timesteps_per_day

        for svar in self.name_to_state_variable_dict.values():
            svar.current_val += svar.change_in_current_val / timesteps_per_day

    def compute_change_in_state_variables(self):

        pass

    def compute_transition_rates(self):

        pass
