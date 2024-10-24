import numpy as np
import json
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class TransitionTypes(str, Enum):
    BINOMIAL = "binomial"
    BINOMIAL_DETERMINISTIC = "binomial_deterministic"
    BINOMIAL_TAYLOR_APPROX = "binomial_taylor_approx"
    BINOMIAL_TAYLOR_APPROX_DETERMINISTIC = "binomial_taylor_approx_deterministic"
    POISSON = "poisson"
    POISSON_DETERMINISTIC = "poisson_deterministic"


class JointTransitionTypes(str, Enum):
    MULTINOMIAL = "multinomial"
    MULTINOMIAL_DETERMINISTIC = "multinomial_deterministic"
    MULTINOMIAL_TAYLOR_APPROX = "multinomial_taylor_approx"
    MULTINOMIAL_TAYLOR_APPROX_DETERMINISTIC = "multinomial_taylor_approx_deterministic"
    POISSON = "poisson"
    POISSON_DETERMINISTIC = "poisson_deterministic"


def approx_binomial_probability_from_rate(rate, interval_length):
    """
    Converts a rate (events per time) to the probability of any event
    occurring in the next time interval of length interval_length,
    assuming the number of events occurring in time interval
    follows a Poisson distribution with given rate parameter

    The probability of 0 events in time_interval_length is
    e^(-rate * time_interval_length), so the probability of any event
    in time_interval_length is 1 - e^(-rate * time_interval_length)

    Rate must be A x L np.ndarray, where A is the number of age groups
    and L is the number of risk groups. Rate is transformed to
    A x L np.ndarray corresponding to probabilities.

    Parameters
    ----------
    :param rate: np.ndarray of positive scalars,
        dimension A x L (number of age groups
        x number of risk groups), rate parameters
        in a Poisson distribution
    :param interval_length: positive scalar,
        length of time interval in simulation days
    :return: np.ndarray of positive scalars,
        dimension A x L
    """

    return 1 - np.exp(-rate * interval_length)


@dataclass
class Config:
    """
    Stores simulation configuration values.

    Attributes
    ----------
    :ivar timesteps_per_day: int,
        number of discretized timesteps within a simulation
        day -- more timesteps_per_day mean smaller discretization
        time intervals, which may cause the model to run slower
    :ivar transition_type: str,
        valid value must be from TransitionTypes, specifying the
        probability distribution of transitions between compartments
    """

    timesteps_per_day: int = 7
    transition_type: str = TransitionTypes.BINOMIAL


class TransitionVariableGroup(ABC):
    """
    Container for TransitionVariable objects to handle joint sampling,
    when there are multiple outflows from a single compartment

    For example, if all outflows of compartment H are: R and D,
    i.e. from the Hospital compartment, people either go to Recovered
    or Dead, a TransitionVariableGroup that holds both R and D handles
    the correct correlation structure between R and D

    When an instance is initialized, its get_joint_realization attribute
    is dynamically assigned to a method according to its transition_type
    attribute. This enables all instances to use the same method during
    simulation.

    Attributes
    ----------
    :ivar origin: EpiCompartment,
        specifies origin of TransitionVariableGroup --
        corresponding populations leave this compartment
    :ivar _transition_type: str,
        only values defined in JointTransitionTypes Enum are valid,
        specifies joint probability distribution of all outflows
        from origin
    :ivar transition_variables: list-like of TransitionVariable instances,
        specifying TransitionVariable instances that outflow from origin --
        order does not matter
    :ivar get_joint_realization: function,
        assigned at initialization, generates realizations according
        to probability distribution given by _transition_type attribute,
        returns either (M x A x L) or ((M+1) x A x L) np.ndarray,
        where M is the length of transition_variables (i.e., number of
        outflows from origin), A is number of age groups, L is number of
        risk groups
    :ivar current_realizations_list: list,
        used to store results from get_joint_realization --
        has either M or M+1 np.ndarrays of size A x L

    See __init__ docstring for other attributes.
    """

    def __init__(self,
                 origin,
                 transition_type,
                 transition_variables):
        """
        :param transition_type: str,
            only values defined in TransitionTypes Enum are valid, specifying
            probability distribution of transitions between compartments

        See class docstring for other parameters.
        """

        self.origin = origin
        self.transition_variables = transition_variables

        # If marginal transition type is any kind of binomial transition,
        #   then its joint transition type is a multinomial counterpart
        # For example, if the marginal transition type is TransitionTypes.BINOMIAL_DETERMINISTIC,
        #   then the joint transition type is JointTransitionTypes.MULTINOMIAL_DETERMINISTIC
        transition_type = transition_type.replace("binomial", "multinomial")
        self._transition_type = transition_type

        # Dynamically assign a method to get_joint_realization attribute
        #   based on the value of transition_type
        # getattr fetches a method by name
        self.get_joint_realization = getattr(self, "get_" + transition_type + "_realization")

        self.current_realizations_list = []

    @property
    def transition_type(self):
        return self._transition_type

    def get_total_rate(self):
        """
        Return the age-risk-specific total transition rate,
        which is the sum of the current rate of each transition variable
        in this transition variable group

        Used to properly scale multinomial probabilities vector so
        that elements sum to 1

        :return: np.ndarray of positive numbers,
            size equal to number of age groups x number of risk groups,
            sum of current rates of transition variables in
            transition variable group
        """

        # axis 0: corresponds to outgoing transition variable
        # axis 1: corresponds to age groups
        # axis 2: corresponds to risk groups
        # --> summing over axis 0 gives the total rate for each age-risk group
        return np.sum(self.get_current_rates_array(), axis=0)

    def get_probabilities_array(self, num_timesteps):
        """
        Returns an array of probabilities used for joint binomial
        (multinomial) transitions (get_multinomial_realization method)

        :return: np.ndarray of positive numbers,
            size equal to (length of outgoing transition variables list + 1)
            x number of age groups x number of risk groups --
            note the "+1" corresponds to the multinomial outcome of staying
            in the same epi compartment (not transitioning to any outgoing
            epi compartment)
        """

        total_rate = self.get_total_rate()

        total_outgoing_probability = approx_binomial_probability_from_rate(total_rate,
                                                                           1 / num_timesteps)

        # Create probabilities_list, where element i corresponds to the
        #   transition variable i's current rate divided by the total rate,
        #   multiplied by the total outgoing probability
        # This generates the probabilities array that parameterizes the
        #   multinomial distribution
        probabilities_list = []

        for transition_variable in self.transition_variables:
            probabilities_list.append((transition_variable.current_rate / total_rate) *
                                      total_outgoing_probability)

        # Append the probability that a person stays in the compartment
        probabilities_list.append(1 - total_outgoing_probability)

        return np.asarray(probabilities_list)

    def get_current_rates_array(self):
        """
        Returns an array of current rates of transition variables in
        self.transition_variables -- ith element in array
        corresponds to current rate of ith transition variable

        :return: np.ndarray of positive numbers,
            size equal to length of outgoing transition variables list
            x number of age groups x number of risk groups
        """

        current_rates_list = []
        for tvar in self.transition_variables:
            current_rates_list.append(tvar.current_rate)

        return np.asarray(current_rates_list)

    def get_joint_realization(self):
        pass

    def get_multinomial_realization(self, RNG, num_timesteps):
        """
        Returns an array of transition realizations (number transitioning
        to outgoing compartments) sampled from multinomial distribution

        :return: np.ndarray of positive numbers,
            size equal to (length of outgoing transition variables list + 1)
            x number of age groups x number of risk groups --
            note the "+1" corresponds to the multinomial outcome of staying
            in the same epi compartment (not transitioning to any outgoing
            epi compartment)
        """

        probabilities_array = self.get_probabilities_array(num_timesteps)

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

    def get_multinomial_taylor_approx_realization(self, RNG, num_timesteps):
        """
        Returns an array of transition realizations (number transitioning
        to outgoing compartments) sampled from multinomial distribution
        using Taylor Series approximation for probability parameter

        :return: np.ndarray of positive numbers,
            size equal to (length of outgoing transition variables list + 1)
            x number of age groups x number of risk groups --
            note the "+1" corresponds to the multinomial outcome of staying
            in the same epi compartment (not transitioning to any outgoing
            epi compartment)
        """

        num_outflows = len(self.transition_variables)

        current_rates_array = self.get_current_rates_array()

        total_rate = self.get_total_rate()

        # Multiply current rates array by length of time interval (1 / num_timesteps)
        # Also append additional value corresponding to probability of
        #   remaining in current epi compartment (not transitioning at all)
        # Note: vstack function here works better than append function because append
        #   automatically flattens the resulting array, resulting in dimension issues
        current_scaled_rates_array = np.vstack((current_rates_array / num_timesteps,
                                                np.expand_dims(1 - total_rate / num_timesteps, axis=0)))

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

    def get_poisson_realization(self, RNG, num_timesteps):
        """
        Returns an array of transition realizations (number transitioning
        to outgoing compartments) sampled from Poisson distribution

        :return: np.ndarray of positive numbers,
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
                            age_group, risk_group] * 1 / num_timesteps)

        return realizations_array

    def get_multinomial_deterministic_realization(self, num_timesteps):
        """
        Deterministic counterpart to get_multinomial_realization --
        uses mean (n x p, i.e. total counts x probability array) as realization
        rather than randomly sampling

        :return: np.ndarray of positive numbers,
            size equal to (length of outgoing transition variables list + 1)
            x number of age groups x number of risk groups --
            note the "+1" corresponds to the multinomial outcome of staying
            in the same epi compartment (not transitioning to any outgoing
            epi compartment)
        """

        probabilities_array = self.get_probabilities_array(num_timesteps)
        return self.origin.current_val * probabilities_array

    def get_multinomial_taylor_approx_deterministic_realization(self, num_timesteps):
        """
        Deterministic counterpart to get_multinomial_taylor_approx_realization --
        uses mean (n x p, i.e. total counts x probability array) as realization
        rather than randomly sampling

        :return: np.ndarray of positive numbers,
            size equal to (length of outgoing transition variables list + 1)
            x number of age groups x number of risk groups --
            note the "+1" corresponds to the multinomial outcome of staying
            in the same epi compartment (not transitioning to any outgoing
            epi compartment)
        """

        current_rates_array = self.get_current_rates_array()
        return self.origin.current_val * current_rates_array / num_timesteps

    def get_poisson_deterministic_realization(self, num_timesteps):
        """
        Deterministic counterpart to get_poisson_realization --
        uses mean (rate array) as realization rather than randomly sampling

        :return: np.ndarray of positive numbers,
            size equal to length of outgoing transition variables list
            x number of age groups x number of risk groups
        """

        return self.get_current_rates_array() / num_timesteps

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
            self.transition_variables[ix].current_realization = \
                self.current_realizations_list[ix, :, :]


class TransitionVariable(ABC):
    """
    Abstract base class for transition variables in
    epidemiological model.

    For example, in an S-I-R model, the new number infected
    every iteration (the number going from S to I) in an iteration
    is modeled as a TransitionVariable subclass, with a concrete
    implementation of the abstract method get_current_rate.

    When an instance is initialized, its get_realization attribute
    is dynamically assigned, just like in the case of
    TransitionVariableGroup instantiation.

    Attributes
    ----------
    :ivar _transition_type: str,
        only values defined in TransitionTypes Enum are valid, specifying
        probability distribution of transitions between compartments
    :ivar get_current_rate: function,
        provides specific implementation for computing current rate
        as a function of current simulation state and epidemiological parameters
    :ivar current_rate: np.ndarray,
        holds output from get_current_rate method -- used to generate
        random variable realizations for transitions between compartments
    :ivar current_realization: np.ndarray,
        holds realization of random variable parameterized by current_rate
        attribute

    See __init__ docstring for other attributes.
    """

    def __init__(self,
                 origin,
                 destination,
                 transition_type,
                 is_jointly_distributed=False):
        """
        :param origin: EpiCompartment,
            the compartment from which Transition Variable exits
        :param destination: EpiCompartment,
            the compartment which Transition Variable enters
        :param transition_type: str,
            only values defined in TransitionTypes Enum are valid, specifying
            probability distribution of transitions between compartments
        :param is_jointly_distributed: Boolean,
            indicates if transition quantity must be jointly computed
            (i.e. if there are multiple outflows from the origin compartment)

        """

        self.origin = origin
        self.destination = destination

        # Also see __init__ method in TransitionVariableGroup class.
        #   The structure is similar.
        self._transition_type = transition_type
        self._is_jointly_distributed = is_jointly_distributed

        if is_jointly_distributed:
            self.get_realization = None
        else:
            self.get_realization = getattr(self, "get_" + transition_type + "_realization")

        self.current_rate = 0
        self.current_realization = 0

    @abstractmethod
    def get_current_rate(self, sim_state, epi_params) -> np.ndarray:
        """
        Computes and returns current rate of transition variable,
        based on current state of the simulation and epidemiological parameters.
        Output should be a numpy array of size A x L, where A is
        sim_state.num_age_groups and L is sim_state.num_risk_groups

        :param sim_state: DataClass,
            holds simulation state (current values of
            EpiCompartment instances and StateVariable
            instances)
        :param epi_params: DataClass,
            holds values of epidemiological parameters
        :return: np.ndarray,
            holds age-risk transition rate,
            must be same shape as origin.init_val,
            i.e. be size A x L, where A is sim_state.num_age_groups
            and L is sim_state.num_risk_groups
        """
        pass

    def update_origin_outflow(self):
        self.origin.current_outflow += self.current_realization

    def update_destination_inflow(self):
        self.destination.current_inflow += self.current_realization

    @property
    def transition_type(self):
        return self._transition_type

    @property
    def is_jointly_distributed(self):
        return self._is_jointly_distributed

    def get_binomial_realization(self, RNG, num_timesteps):
        return RNG.binomial(n=np.asarray(self.base_count, dtype=int),
                            p=approx_binomial_probability_from_rate(self.current_rate, 1 / num_timesteps))

    def get_binomial_taylor_approx_realization(self, RNG, num_timesteps):
        return RNG.binomial(n=np.asarray(self.base_count, dtype=int),
                            p=self.current_rate * (1 / num_timesteps))

    def get_poisson_realization(self, RNG, num_timesteps):
        return RNG.poisson(self.base_count * self.current_rate * (1 / num_timesteps))

    def get_binomial_deterministic_realization(self, num_timesteps):
        return np.asarray(self.base_count *
                          approx_binomial_probability_from_rate(self.current_rate, 1 / num_timesteps),
                          dtype=int)

    def get_binomial_taylor_approx_deterministic_realization(self, num_timesteps):
        return np.asarray(self.base_count * self.current_rate * (1 / num_timesteps), dtype=int)

    def get_poisson_deterministic_realization(self, num_timesteps):
        return np.asarray(self.base_count * self.current_rate * (1 / num_timesteps), dtype=int)

    @property
    def base_count(self):
        return self.origin.current_val


class EpiCompartment:
    """
    Class for epidemiological compartments (e.g. Susceptible,
        Exposed, Infected, etc...)

    Attributes
    ----------
    :ivar current_val: np.ndarray,
        same size as init_val, holds current value of EpiCompartment
        for age-risk groups
    :ivar current_inflow: np.ndarray,
        same size as current_val, used to sum up all
        transition variable realizations incoming to this compartment
        for age-risk groups
    :ivar current_outflow: np.ndarray,
        same size of current_val, used to sum up all
        transition variable realizations outgoing from this compartment
        for age-risk groups
    :ivar history_vals_list: list of np.ndarrays,
        each element is the same size of current_val, holds
        history of compartment states for age-risk groups --
        element t corresponds to previous current_val value at
        end of simulation day t
    """

    def __init__(self,
                 name,
                 init_val):
        """
        :param name: str,
            user-specified name for compartment
        :param init_val: 2D np.ndarray of integers,
            corresponding to initial population in compartment,
            where i,jth entry corresponds to age group i
            and risk group j

            If only one age group and risk group, init_val still
            must be 2D -- e.g. np.array([[100]]), not np.array([100])
        """

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
        """
        Saves current value to history by appending current_val attribute
            to history_vals_list in place

        WARNING: deep copying is CRUCIAL because current_val is a mutable
            np.ndarray -- without deep copying, history_vals_list would
            have the same value for all elements
        """
        self.history_vals_list.append(copy.deepcopy(self.current_val))

    def clear_history(self):
        self.history_vals_list = []


class StateVariable(ABC):
    """
    Abstract base class for state variables in epidemiological model.

    This is intended for variables that are deterministic functions of
    the simulation state (including epi compartment values, other parameters,
    and time.)

    For example, population-level immunity variables should be
    modeled as a StateVariable subclass, with a concrete
    implementation of the abstract method get_change_in_current_val.

    Attributes
    ----------
    :ivar current_val: np.ndarray,
        same size as init_val, holds current value of State Variable
        for age-risk groups
    :ivar change_in_current_val: np.ndarray,
        initialized to None, but during simulation holds change in
        current value of StateVariable for age-risk groups
        (size A x L, where A is number of risk groups and L is number
        of age groups)

    See __init__ docstring for other attributes.
    """

    def __init__(self,
                 name,
                 init_val):
        """
        :param name: str,
            name of StateVariable
        :param init_val: 2D np.ndarray of nonnegative floats,
            corresponding to initial value of state variable,
            where i,jth entry corresponds to age group i and
            risk group j
        """

        self.name = name
        self.init_val = init_val

        self.current_val = copy.deepcopy(init_val)
        self.change_in_current_val = None

        self.history_vals_list = []

    @abstractmethod
    def get_change_in_current_val(self, sim_state, epi_params, num_timesteps) -> np.ndarray:
        """
        Computes and returns change in current value of state variable,
        based on current state of the simulation and epidemiological parameters.
        Output should be a numpy array of size A x L, where A is
        sim_state.num_age_groups and L is sim_state.num_risk_groups

        :param sim_state: DataClass,
            holds simulation state (current values of
            EpiCompartment instances and StateVariable
            instances)
        :param epi_params: DataClass,
            holds values of epidemiological parameters
        :param num_timesteps: int,
            number of timesteps -- used to determine time interval
            length for discretization
        :return: np.ndarray,
            size A x L, where A is sim_state.num_age_groups and L is
            sim_state.num_risk_groups
        """
        pass

    def update_current_val(self):
        self.current_val += self.change_in_current_val

    def update_sim_state(self, sim_state):
        setattr(sim_state, self.name, self.current_val)

    def update_history(self):
        """
        Saves current value to history by appending current_val attribute
            to history_vals_list in place

        WARNING: deep copying is CRUCIAL because current_val is a mutable
            np.ndarray -- without deep copying, history_vals_list would
            have the same value for all elements
        """
        self.history_vals_list.append(copy.deepcopy(self.current_val))

    def clear_history(self):
        self.history_vals_list = []


class TransmissionModel:
    """
    Contains and manages all necessary components for
    simulating a compartmental model, including compartments,
    transition variables and transition variable groups,
    state variables, a data container for the current simulation state,
    epidemiological parameters, simulation experiment configuration
    parameters, and a random number generator.

    All city-level models, regardless of disease type and
    compartment/transition structure, are instances of this class.

    When creating an instance, the order of elements does not matter
    within compartments, transition_variables, transition_variable_groups,
    and state_variables. The "flow" and "physics" information are stored
    on the objects.

    Attributes
    ----------
    :ivar compartments: list-like,
        list of all the model's EpiCompartment instances
    :ivar transition_variables: list-like,
        list of all the model's TransitionVariable instances
    :ivar transition_variable_groups: list-like,
        list of all the model's TransitionVariableGroup instances
    :ivar state_variables: list-like,
        list of all the model's StateVariable instances
    :ivar sim_state: DataClass,
        data container for the model's current values of
        EpiCompartment instances and StateVariable instances --
        there must be one field for each EpiCompartment instance
        and for each StateVariable instance -- each field's name must
        match the "name" attribute of the corresponding EpiCompartment
        or StateVariable instance
    :ivar epi_params: DataClass,
        data container for the model's epidemiological parameters,
        such as the "Greek letters" characterizing sojourn times
        in compartments
    :ivar config: DataClass,
        data container for the model's simulation configuration values
    :ivar RNG: np.random.Generator object,
        used to generate random variables and control reproducibility
    :ivar current_day_counter: int,
        tracks current simulation day -- incremented by +1
        when config.timesteps_per_day discretized timesteps
        have completed

    See __init__ docstring for other attributes.
    """

    def __init__(self,
                 compartments,
                 transition_variables,
                 transition_variable_groups,
                 state_variables,
                 sim_state,
                 epi_params,
                 config,
                 RNG_seed):
        """
        TODO: maybe group arguments together into DataClass to simplify?

        :param RNG_seed: positive int,
            used to initialize the model's RNG for generating
            random variables and random transitions

        See class docstring for other parameters.
        """

        self.compartments = compartments
        self.transition_variables = transition_variables
        self.transition_variable_groups = transition_variable_groups
        self.state_variables = state_variables

        self.sim_state = sim_state
        self.epi_params = epi_params
        self.config = config

        # Create bit generator seeded with given RNG_seed
        self._bit_generator = np.random.MT19937(seed=RNG_seed)
        self.RNG = np.random.Generator(self._bit_generator)

        self.current_day_counter = 0

    def simulate_until_time_period(self, last_simulation_day):
        """
        Advance simulation model time until last_simulation_day

        Advance time by iterating through simulation days,
        which are simulated by iterating through discretized
        timesteps

        Save daily simulation data as history on each EpiCompartment
        instance

        :param last_simulation_day: positive int,
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

        # Attribute lookup shortcuts
        timesteps_per_day = self.config.timesteps_per_day
        sim_state = self.sim_state
        epi_params = self.epi_params

        for timestep in range(timesteps_per_day):

            for tvar in self.transition_variables:
                tvar.current_rate = tvar.get_current_rate(self.sim_state, epi_params)

            for svar in self.state_variables:
                svar.change_in_current_val = svar.get_change_in_current_val(sim_state,
                                                                            epi_params,
                                                                            timesteps_per_day)

            # Obtain transition variable realizations for jointly distributed transition variables
            #   (i.e. when there are multiple transition variable outflows from an epi compartment)
            for tvargroup in self.transition_variable_groups:
                tvargroup.current_realizations_list = tvargroup.get_joint_realization(self.RNG,
                                                                                      timesteps_per_day)
                tvargroup.update_transition_variable_realizations()

            # Obtain transition variable realizations for marginally distributed transition variables
            #   (i.e. when there is only one transition variable outflow from an epi compartment)
            # If transition variable is jointly distributed, then its realization has already
            #   been computed by its transition variable group container previously,
            #   so skip the marginal computation
            for tvar in self.transition_variables:
                if tvar.is_jointly_distributed:
                    continue
                else:
                    tvar.current_realization = tvar.get_realization(self.RNG, timesteps_per_day)

            """
            ###############################################
            ##### IN-PLACE UPDATE OF SIMULATION STATE #####
            ###############################################
            """

            for tvar in self.transition_variables:
                tvar.update_origin_outflow()
                tvar.update_destination_inflow()

            for svar in self.state_variables:
                svar.update_current_val()
                svar.update_sim_state(sim_state)

            for compartment in self.compartments:
                compartment.update_current_val()
                compartment.update_sim_state(sim_state)

                compartment.reset_inflow()
                compartment.reset_outflow()

        # Update state variable history and compartment history
        #   at end of each day, not at end of every discretization timestep,
        #   to be efficient
        for svar in self.state_variables:
            svar.update_history()

        for compartment in self.compartments:
            compartment.update_history()

        # Move to next day in simulation
        self.current_day_counter += 1


def dataclass_instance_from_json(dataclass_ref, json_filepath):
    """
    Create instance of DataClass from class dataclass_ref,
    based on information in json_filepath

    :param dataclass_ref: DataClass class (class, not instance)
        from which to create instance
    :param json_filepath: str,
        path to json file (path includes actual filename
        with suffix ".json") -- all json fields must
        match name and datatype of dataclass_ref instance
        attributes
    :return: DataClass,
        instance of dataclass_ref with attributes dynamically
        assigned by json_filepath file contents
    """

    with open(json_filepath, 'r') as file:
        data = json.load(file)

    # convert lists to numpy arrays to support numpy operations
    #   since json does not have direct support for numpy
    for key, val in data.items():
        if type(val) is list:
            data[key] = np.asarray(val)

    return dataclass_ref(**data)
