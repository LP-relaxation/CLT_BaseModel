import numpy as np
import json
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


def approx_binomial_probability_from_rate(rate, interval_length):
    """
    Converts a rate (events per time) to the probability of any event
    occurring in the next time interval of length interval_length,
    assuming the number of events occurring in time interval
    follows a Poisson distribution with given rate parameter

    The probability of 0 events in time_interval_length is
    e^(-rate * time_interval_length), so the probability of any event
    in time_interval_length is 1 - e^(-rate * time_interval_length)

    :parameter rate: array-like or positive scalars,
        dimension A x L (number of age groups
        x number of risk groups), rate parameters
        in a Poisson distribution
    :parameter interval_length: positive scalar,
        length of time interval in simulation days
    :return: array-like of positive scalars,
        dimension A x L
    """

    return 1 - np.exp(-rate * interval_length)


@dataclass
class Config:
    timesteps_per_day: int = 7
    starting_simulation_day: int = 0
    transition_type: str = "binomial"


class TransitionVariableGroup(ABC):
    """
    Container for TransitionVariable objects to handle joint sampling,
    when there are multiple outflows from a single compartment

    For example, if all outflows of compartment H are: R and D,
    i.e. from the Hospital compartment, people either go to Recovered
    or Dead, a TransitionVariableGroup that holds both R and D handles
    the correct correlation structure between R and D
    """

    def __init__(self,
                 origin,
                 transition_type,
                 transition_variables):
        self.origin = origin
        self.transition_variables = transition_variables

        self.assign_transition_type(transition_type)

        self.current_realizations_list = []

    @property
    def transition_type(self):
        return self._transition_type

    def assign_transition_type(self, transition_type):
        """
        Updates transition_type and updates get_joint_realization
        method according to transition_type.

        :parameter transition_type: str,
            Supported values are in {"multinomial", "multinomial_taylor_approx", "poisson",
            "multinomial_deterministic", "multinomial_taylor_approx_deterministic",
            "poisson_deterministic"}. Determines which method to use to compute
            realization, corresponding to get_binomial_realization,
            get_binomial_taylor_approx_realization, etc... See methods for
            rigorous explanation.
        """

        # If marginal transition type is binomial, then
        #   joint transition type is multinomial
        transition_type = transition_type.replace("binomial", "multinomial")

        self._transition_type = transition_type
        self.assign_get_joint_realization_func(transition_type)

    def assign_get_joint_realization_func(self, transition_type):
        """
        Updates get_joint_realization method according to transition_type.

        Overrides get_joint_realization so that joint transitions
        are computed according to the desired transition_type.

        :parameter transition_type: str,
            Supported values are in {"multinomial", "multinomial_taylor_approx", "poisson",
            "multinomial_deterministic", "multinomial_taylor_approx_deterministic",
            "poisson_deterministic"}. Determines which method to use to compute
            realization, corresponding to get_binomial_realization,
            get_binomial_taylor_approx_realization, etc... See methods for
            rigorous explanation.
        """

        self.get_joint_realization = getattr(self, "get_" + transition_type + "_realization")

    def get_total_rate(self):
        """
        Return the age-risk-specific total transition rate,
        which is the sum of the current rate of each transition variable
        in this transition variable group

        Used to properly scale multinomial probabilities vector so
        that elements sum to 1

        :return: array-like of positive numbers,
            size equal to number of age groups x number of risk groups,
            sum of current rates of transition variables in
            transition variable group
        """

        # axis 0: corresponds to outgoing transition variable
        # axis 1: corresponds to age groups
        # axis 2: corresponds to risk groups
        # --> summing over axis 0 gives us the total rate for each age-risk group
        return np.sum(self.get_current_rates_array(), axis=0)

    def get_probabilities_array(self, timesteps_per_day):
        """
        Returns an array of probabilities used for joint binomial
        (multinomial) transitions (get_multinomial_realization method)

        :return: array-like of positive numbers,
            size equal to (length of outgoing transition variables list + 1)
            x number of age groups x number of risk groups --
            note the "+1" corresponds to the multinomial outcome of staying
            in the same epi compartment (not transitioning to any outgoing
            epi compartment)
        """

        total_rate = self.get_total_rate()

        total_outgoing_probability = approx_binomial_probability_from_rate(total_rate, 1 / timesteps_per_day)

        # Create probabilities_list, where element i corresponds to the
        #   transition variable i's current rate divided by the total rate,
        #   multiplied by the total outgoing probability
        # This generates the probabilities array that parameterizes the
        #   multinomial distribution
        probabilities_list = []

        for transition_variable in self.transition_variables:
            probabilities_list.append((transition_variable.current_rate / total_rate) * total_outgoing_probability)

        # Append the probability that a person stays in the compartment
        probabilities_list.append(1 - total_outgoing_probability)

        return np.asarray(probabilities_list)

    def get_current_rates_array(self):
        """
        Returns an array of current rates of transition variables in
        self.transition_variables -- ith element in array
        corresponds to current rate of ith transition variable

        :return: array-like of positive numbers,
            size equal to length of outgoing transition variables list
            x number of age groups x number of risk groups
        """

        current_rates_list = [tvar.current_rate for tvar in self.transition_variables]

        return np.asarray(current_rates_list)

    def get_joint_realization(self):
        pass

    def get_multinomial_realization(self, RNG, timesteps_per_day):
        """
        Returns an array of transition realizations (number transitioning
        to outgoing compartments) sampled from multinomial distribution

        :return: array-like of positive numbers,
            size equal to (length of outgoing transition variables list + 1)
            x number of age groups x number of risk groups --
            note the "+1" corresponds to the multinomial outcome of staying
            in the same epi compartment (not transitioning to any outgoing
            epi compartment)
        """

        probabilities_array = self.get_probabilities_array(timesteps_per_day)

        num_outflows = len(self.transition_variables)

        num_age_groups, num_risk_groups = np.shape(self.origin.current_val)

        # We use num_outflows + 1 because for the multinomial distribution we explicitly model
        #   the number who stay/remain in the compartment
        realizations_array = np.zeros((num_outflows + 1, num_age_groups, num_risk_groups))

        for age_group in range(num_age_groups):
            for risk_group in range(num_risk_groups):
                realizations_array[:, age_group, risk_group] = RNG.multinomial(
                    np.asarray(self.origin.current_val[age_group, risk_group], dtype=int),
                    probabilities_array[:, age_group, risk_group])

        return realizations_array

    def get_multinomial_taylor_approx_realization(self, RNG, timesteps_per_day):
        """
        Returns an array of transition realizations (number transitioning
        to outgoing compartments) sampled from multinomial distribution
        using Taylor Series approximation for probability parameter

        :return: array-like of positive numbers,
            size equal to (length of outgoing transition variables list + 1)
            x number of age groups x number of risk groups --
            note the "+1" corresponds to the multinomial outcome of staying
            in the same epi compartment (not transitioning to any outgoing
            epi compartment)
        """

        num_outflows = len(self.transition_variables)

        current_rates_array = self.get_current_rates_array()

        total_rate = self.get_total_rate()

        # Multiply current rates array by length of time interval (1 / timesteps_per_day)
        # Also append additional value corresponding to probability of
        #   remaining in current epi compartment (not transitioning at all)
        # Note: vstack function here works better than append function because append
        #   automatically flattens the resulting array, resulting in dimension issues
        current_scaled_rates_array = np.vstack((current_rates_array / timesteps_per_day,
                                                np.expand_dims(1 - total_rate / timesteps_per_day, axis=0)))

        num_age_groups, num_risk_groups = np.shape(self.origin.current_val)

        # We use num_outflows + 1 because for the multinomial distribution we explicitly model
        #   the number who stay/remain in the compartment
        realizations_array = np.zeros((num_outflows + 1, num_age_groups, num_risk_groups))

        for age_group in range(num_age_groups):
            for risk_group in range(num_risk_groups):
                realizations_array[:, age_group, risk_group] = RNG.multinomial(
                    np.asarray(self.origin.current_val[age_group, risk_group], dtype=int),
                    current_scaled_rates_array[:, age_group, risk_group])

        return realizations_array

    def get_poisson_realization(self, RNG, timesteps_per_day):
        """
        Returns an array of transition realizations (number transitioning
        to outgoing compartments) sampled from Poisson distribution

        :return: array-like of positive numbers,
            size equal to length of outgoing transition variables list
            x number of age groups x number of risk groups
        """

        num_outflows = len(self.transition_variables)

        num_age_groups, num_risk_groups = np.shape(self.origin.current_val)

        realizations_array = np.zeros((num_outflows, num_age_groups, num_risk_groups))

        transition_variables = self.transition_variables

        for age_group in range(num_age_groups):
            for risk_group in range(num_risk_groups):
                for outflow_ix in range(num_outflows):
                    realizations_array[outflow_ix, age_group, risk_group] = RNG.poisson(
                        self.origin.current_val[age_group, risk_group] *
                        transition_variables[outflow_ix].current_rate[
                            age_group, risk_group] * 1 / timesteps_per_day)

        return realizations_array

    def get_multinomial_deterministic_realization(self, timesteps_per_day):
        """
        Deterministic counterpart to get_multinomial_realization --
        uses mean (n x p, i.e. total counts x probability array) as realization
        rather than randomly sampling

        :return: array-like of positive numbers,
            size equal to (length of outgoing transition variables list + 1)
            x number of age groups x number of risk groups --
            note the "+1" corresponds to the multinomial outcome of staying
            in the same epi compartment (not transitioning to any outgoing
            epi compartment)
        """

        probabilities_array = self.get_probabilities_array(timesteps_per_day)
        return self.origin.current_val * probabilities_array

    def get_multinomial_taylor_approx_deterministic_realization(self, timesteps_per_day):
        """
        Deterministic counterpart to get_multinomial_taylor_approx_realization --
        uses mean (n x p, i.e. total counts x probability array) as realization
        rather than randomly sampling

        :return: array-like of positive numbers,
            size equal to (length of outgoing transition variables list + 1)
            x number of age groups x number of risk groups --
            note the "+1" corresponds to the multinomial outcome of staying
            in the same epi compartment (not transitioning to any outgoing
            epi compartment)
        """

        current_rates_array = self.get_current_rates_array()
        return self.origin.current_val * current_rates_array / timesteps_per_day

    def get_poisson_deterministic_realization(self, timesteps_per_day):
        """
        Deterministic counterpart to get_poisson_realization --
        uses mean (rate array) as realization rather than randomly sampling

        :return: array-like of positive numbers,
            size equal to length of outgoing transition variables list
            x number of age groups x number of risk groups
        """

        return self.get_current_rates_array() / timesteps_per_day

    def reset(self):
        self.current_realizations_list = []

    def update_transition_variable_realizations(self):
        """
        Updates current_realization attribute on all
        TransitionVariable instances contained in this
        transition variable group
        """

        # Since the ith element in probabilities_array corresponds to the ith transition variable
        #   in transition_variables, the ith element in multinomial_realizations_list
        #   also corresponds to the ith transition variable in transition_variables
        # Update the current realization of the transition variables contained in this group
        for ix in range(len(self.transition_variables)):
            self.transition_variables[ix].current_realization = self.current_realizations_list[ix, :, :]


class TransitionVariable(ABC):

    def __init__(self,
                 origin,
                 destination,
                 transition_type,
                 is_jointly_distributed=False):

        self.origin = origin
        self.destination = destination

        self.is_jointly_distributed = is_jointly_distributed

        self.assign_transition_type(transition_type)

        self.current_rate = 0
        self.current_realization = 0

    @abstractmethod
    def get_current_rate(self, sim_state, epi_params):
        pass

    def update_origin_outflow(self):
        self.origin.current_outflow += self.current_realization

    def update_destination_inflow(self):
        self.destination.current_inflow += self.current_realization

    @property
    def transition_type(self):
        return self._transition_type

    def assign_get_realization_func(self, transition_type):
        if self.is_jointly_distributed:
            pass
        else:
            self.get_realization = getattr(self, "get_" + transition_type + "_realization")

    def assign_transition_type(self, transition_type):
        self._transition_type = transition_type
        self.assign_get_realization_func(transition_type)

    def get_binomial_realization(self, RNG, timesteps_per_day):
        return RNG.binomial(n=np.asarray(self.base_count, dtype=int),
                            p=approx_binomial_probability_from_rate(self.current_rate, 1 / timesteps_per_day))

    def get_binomial_taylor_approx_realization(self, RNG, timesteps_per_day):
        return RNG.binomial(n=np.asarray(self.base_count, dtype=int),
                            p=self.current_rate * (1 / timesteps_per_day))

    def get_poisson_realization(self, RNG, timesteps_per_day):
        return RNG.poisson(self.base_count * self.current_rate * (1 / timesteps_per_day))

    def get_binomial_deterministic_realization(self, timesteps_per_day):
        return np.asarray(self.base_count *
                          approx_binomial_probability_from_rate(self.current_rate, 1 / timesteps_per_day),
                          dtype=int)

    def get_binomial_taylor_approx_deterministic_realization(self, timesteps_per_day):
        return np.asarray(self.base_count * self.current_rate * (1 / timesteps_per_day), dtype=int)

    def get_poisson_deterministic_realization(self, timesteps_per_day):
        return np.asarray(self.base_count * self.current_rate * (1 / timesteps_per_day), dtype=int)

    @property
    def base_count(self):
        return self.origin.current_val


class EpiCompartment:

    def __init__(self,
                 name,
                 init_val):
        self.name = name
        self.init_val = init_val

        self.current_val = copy.deepcopy(init_val)
        self.current_inflow = np.zeros(np.shape(init_val))
        self.current_outflow = np.zeros(np.shape(init_val))

        self.history_vals_list = []

    def update_current_val(self):
        self.current_val += self.current_inflow - self.current_outflow

    def update_sim_state(self, sim_state):
        setattr(sim_state, self.name, self.current_val)

    def reset_inflow(self):
        self.current_inflow = np.zeros(np.shape(self.current_inflow))

    def reset_outflow(self):
        self.current_outflow = np.zeros(np.shape(self.current_outflow))

    def update_history(self):
        self.history_vals_list.append(copy.deepcopy(self.current_val))

    def clear_history(self):
        self.history_vals_list = []


class StateVariable(ABC):
    """
    Class for variables that are deterministic functions of
    the simulation state (meaning epi compartment values, other parameters,
    and time.)

    For example, population-level immunity against hospitalization and
    infection can both be modeled as a StateVariable instance.
    """

    def __init__(self,
                 name,
                 init_val):
        self.name = name
        self.init_val = init_val
        self.current_val = copy.deepcopy(init_val)
        self.change_in_current_val = None

    @abstractmethod
    def get_change_in_current_val(self, sim_state, epi_params):
        pass

    def update_current_val(self, timesteps_per_day):
        self.current_val += self.change_in_current_val / timesteps_per_day

    def update_sim_state(self, sim_state):
        setattr(sim_state, self.name, self.current_val)


class TransmissionModel:

    def __init__(self,
                 compartments,
                 transition_variables,
                 transition_variable_groups,
                 state_variables,
                 sim_state,
                 epi_params,
                 config,
                 RNG_seed):

        self.compartments = compartments
        self.transition_variables = transition_variables
        self.transition_variable_groups = transition_variable_groups
        self.state_variables = state_variables

        self.sim_state = sim_state

        self.epi_params = epi_params
        self.config = config

        self._bit_generator = np.random.MT19937(seed=RNG_seed)
        self.RNG = np.random.Generator(self._bit_generator)

        self.current_day_counter = 0

    def simulate_until_time_period(self, last_simulation_day):
        """
        Advance simulation model time until last_simulation_day

        Advance time by iterating through simulation days,
        which are simulated by iterating through discretized
        timesteps

        Save daily simulation data as history

        :parameter last_simulation_day: positive int,
            stop simulation at last_simulation_day (i.e. exclusive,
            simulate up to but not including last_simulation_day)
        """

        # last_simulation_day is exclusive endpoint
        while self.current_day_counter < last_simulation_day:
            self.simulate_discretized_timesteps()

    def simulate_discretized_timesteps(self):
        """
        Subroutine for simulate_until_time_period

        Iterates through discretized timesteps to simulate next
        simulation day. Granularity of discretization is given by
        attribute config.timesteps_per_day

        Properly scales transition variable realizations and changes
        in state variables by specified timesteps per day
        """

        # Attribute lookup shortcut
        timesteps_per_day = self.config.timesteps_per_day
        sim_state = self.sim_state
        epi_params = self.epi_params
        config = self.config

        for timestep in range(timesteps_per_day):

            for tvar in self.transition_variables:
                tvar.current_rate = tvar.get_current_rate(self.sim_state, epi_params)

            for svar in self.state_variables:
                svar.change_in_current_val = svar.get_change_in_current_val(sim_state, epi_params)

            # Obtain transition variable realizations for jointly distributed transition variables
            #   (i.e. when there are multiple transition variable outflows from an epi compartment)
            for tvargroup in self.transition_variable_groups:
                tvargroup.current_realizations_list = tvargroup.get_joint_realization(self.RNG, timesteps_per_day)
                tvargroup.update_transition_variable_realizations()

            # Obtain transition variable realizations for marginally distributed transition variables
            #   (i.e. when there is only one transition variable outflow from an epi compartment)
            # If transition variable is jointly distributed, then its realization has already
            #   been computed by its transition variable group container previously, so skip it
            for tvar in self.transition_variables:
                if tvar.is_jointly_distributed:
                    continue
                else:
                    tvar.current_realization = tvar.get_realization(self.RNG, timesteps_per_day)

            # In-place updates -- advance the simulation
            for tvar in self.transition_variables:
                tvar.update_origin_outflow()
                tvar.update_destination_inflow()

            for svar in self.state_variables:
                svar.update_current_val(config.timesteps_per_day)
                svar.update_sim_state(sim_state)

            for compartment in self.compartments:
                compartment.update_current_val()
                compartment.update_history()
                compartment.update_sim_state(sim_state)
                compartment.reset_inflow()
                compartment.reset_outflow()

        # Move to next day in simulation
        self.current_day_counter += 1
