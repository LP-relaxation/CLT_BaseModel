from .utils import np, sc, copy, ABC, abstractmethod, dataclass, \
    Optional, Union, Enum, datetime, pd
from collections import defaultdict


class SubpopModelError(Exception):
    """Custom exceptions for simulation model errors."""
    pass


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


def approx_binomial_probability_from_rate(rate: np.ndarray,
                                          interval_length: int) -> np.ndarray:
    """
    Converts a rate (events per time) to the probability of any event
    occurring in the next time interval of length interval_length,
    assuming the number of events occurring in time interval
    follows a Poisson distribution with given rate parameter.

    The probability of 0 events in time_interval_length is
    e^(-rate * time_interval_length), so the probability of any event
    in time_interval_length is 1 - e^(-rate * time_interval_length).

    Rate must be A x L np.ndarray, where A is the number of age groups
    and L is the number of risk groups. Rate is transformed to
    A x L np.ndarray corresponding to probabilities.

    Parameters:
        rate (np.ndarray):
            dimension A x L (number of age groups x number of risk groups),
            rate parameters in a Poisson distribution.
        interval_length (positive int):
            length of time interval in simulation days.

    Returns:
        np.ndarray: array of positive scalars, dimension A x L
    """

    return 1 - np.exp(-rate * interval_length)


@dataclass
class Config:
    """
    Stores simulation configuration values.

    Attributes:
        timesteps_per_day (int):
            number of discretized timesteps within a simulation
            day -- more timesteps_per_day mean smaller discretization
            time intervals, which may cause the model to run slower.
        transition_type (str):
            valid value must be from TransitionTypes,
            specifying the probability distribution of transitions between
            compartments.
        start_real_date (datetime.date):
            actual date that aligns with the beginning of the simulation.
        save_daily_history (bool):
            True if each StateVariable saves state to history after each simulation
            day -- set to False if want speedier performance.
    """

    timesteps_per_day: int = 7
    transition_type: str = TransitionTypes.BINOMIAL
    start_real_date: datetime.time = datetime.datetime.strptime("2024-10-31",
                                                                "%Y-%m-%d").date()
    save_daily_history: bool = True


@dataclass
class SubpopParams(ABC):
    """
    Stores epidemiological parameters for
        a given Subpopulation.
    """
    pass


@dataclass
class InterSubpopRepo(ABC):
    """
    Holds collection of SubpopState instances, with
        actions to query and interact with them.
    """

    subpop_models: Optional[sc.objdict] = None
    travel_proportions: Optional[pd.DataFrame] = None


@dataclass
class SubpopState(ABC):
    """
    Holds current values of Subpop simulation state.
    """

    def sync_to_current_vals(self, lookup_dict):
        for name, item in lookup_dict.items():
            setattr(self, name, item.current_val)


class TransitionVariableGroup:
    """
    Container for TransitionVariable objects to handle joint sampling,
    when there are multiple outflows from a single compartment.

    For example, if all outflows of compartment H are: R and D,
    i.e. from the Hosp compartment, people either go to Recovered
    or Dead, a TransitionVariableGroup that holds both R and D handles
    the correct correlation structure between R and D.

    When an instance is initialized, its get_joint_realization attribute
    is dynamically assigned to a method according to its transition_type
    attribute. This enables all instances to use the same method during
    simulation.

    Attributes:
        origin (Compartment):
            specifies origin of TransitionVariableGroup --
            corresponding populations leave this compartment.
        _transition_type (str):
            only values defined in JointTransitionTypes Enum are valid,
            specifies joint probability distribution of all outflows
            from origin.
        transition_variables (list[TransitionVariable]):
            specifying TransitionVariable instances that outflow from origin --
            order does not matter.
        get_joint_realization (function):
            assigned at initialization, generates realizations according
            to probability distribution given by _transition_type attribute,
            returns either (M x A x L) or ((M+1) x A x L) np.ndarray,
            where M is the length of transition_variables (i.e., number of
            outflows from origin), A is number of age groups, L is number of
            risk groups.
        current_vals_list (list):
            used to store results from get_joint_realization --
            has either M or M+1 arrays of size A x L.

    See __init__ docstring for other attributes.
    """

    def __init__(self,
                 origin,
                 transition_type,
                 transition_variables):
        """
        Args:
            transition_type (str):
                only values defined in TransitionTypes Enum are valid, specifying
                probability distribution of transitions between compartments.

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

        self.current_vals_list = []

    @property
    def transition_type(self) -> JointTransitionTypes:
        return self._transition_type

    def get_total_rate(self) -> np.ndarray:
        """
        Return the age-risk-specific total transition rate,
        which is the sum of the current rate of each transition variable
        in this transition variable group.

        Used to properly scale multinomial probabilities vector so
        that elements sum to 1.

        Returns:
            np.ndarray:
                contains positive floats, has size equal to number
                of age groups x number of risk groups,
                sum of current rates of transition variables in
                transition variable group.
        """

        # axis 0: corresponds to outgoing transition variable
        # axis 1: corresponds to age groups
        # axis 2: corresponds to risk groups
        # --> summing over axis 0 gives the total rate for each age-risk group
        return np.sum(self.get_current_rates_array(), axis=0)

    def get_probabilities_array(self,
                                num_timesteps: int) -> list:
        """
        Returns an array of probabilities used for joint binomial
        (multinomial) transitions (get_multinomial_realization method).

        Returns:
            np.ndarray:
                contains positive floats <= 1, size equal to
                ((length of outgoing transition variables list + 1)
                x number of age groups x number of risk groups) --
                note the "+1" corresponds to the multinomial outcome of staying
                in the same epi compartment (not transitioning to any outgoing
                epi compartment).
        """

        total_rate = self.get_total_rate()

        total_outgoing_probability = approx_binomial_probability_from_rate(total_rate,
                                                                           1 / num_timesteps)

        # Create probabilities_list, where element i corresponds to the
        #   transition variable i's current rate divided by the total rate,
        #   multiplized by the total outgoing probability
        # This generates the probabilities array that parameterizes the
        #   multinomial distribution
        probabilities_list = []

        for transition_variable in self.transition_variables:
            probabilities_list.append((transition_variable.current_rate / total_rate) *
                                      total_outgoing_probability)

        # Append the probability that a person stays in the compartment
        probabilities_list.append(1 - total_outgoing_probability)

        return np.asarray(probabilities_list)

    def get_current_rates_array(self) -> np.ndarray:
        """
        Returns an array of current rates of transition variables in
        self.transition_variables -- ith element in array
        corresponds to current rate of ith transition variable.

        Returns:
            np.ndarray:
                array of positive floats, size equal to (length of outgoing
                transition variables list x number of age groups x number of risk groups).
        """

        current_rates_list = []
        for tvar in self.transition_variables:
            current_rates_list.append(tvar.current_rate)

        return np.asarray(current_rates_list)

    def get_joint_realization(self,
                              RNG: np.random.Generator,
                              num_timesteps: int) -> np.ndarray:
        """
        This function is dynamically assigned based on the Transition
        Variable Group's transition type -- this function is set to
        one of the following methods: get_multinomial_realization,
        get_multinomial_taylor_approx_realization, get_poisson_realization,
        get_multinomial_deterministic_realization,
        get_multinomial_taylor_approx_deterministic_realization,
        get_poisson_deterministic_realization.

        Parameters:
            RNG (np.random.Generator object):
                used to generate random variables and control reproducibility.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.
        """

        pass

    def get_multinomial_realization(self,
                                    RNG: np.random.Generator,
                                    num_timesteps: int) -> np.ndarray:
        """
        Returns an array of transition realizations (number transitioning
        to outgoing compartments) sampled from multinomial distribution.

        Parameters:
            RNG (np.random.Generator object):
                used to generate random variables and control reproducibility.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                contains positive floats, size equal to
                ((length of outgoing transition variables list + 1)
                x number of age groups x number of risk groups) --
                note the "+1" corresponds to the multinomial outcome of staying
                in the same epi compartment (not transitioning to any outgoing
                epi compartment).
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

    def get_multinomial_taylor_approx_realization(self,
                                                  RNG: np.random.Generator,
                                                  num_timesteps: int) -> np.ndarray:
        """
        Returns an array of transition realizations (number transitioning
        to outgoing compartments) sampled from multinomial distribution
        using Taylor Series approximation for probability parameter.

        Parameters:
            RNG (np.random.Generator object):
                used to generate random variables and control reproducibility.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                size equal to (length of outgoing transition variables list + 1)
                x number of age groups x number of risk groups --
                note the "+1" corresponds to the multinomial outcome of staying
                in the same epi compartment (not transitioning to any outgoing
                epi compartment).
        """

        num_outflows = len(self.transition_variables)

        current_rates_array = self.get_current_rates_array()

        total_rate = self.get_total_rate()

        # Multiply current rates array by length of time interval (1 / num_timesteps)
        # Also append additional value corresponding to probability of
        #   remaining in current epi compartment (not transitioning at all)
        # Note: "vstack" function here works better than append function because append
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

    def get_poisson_realization(self,
                                RNG: np.random.Generator,
                                num_timesteps: int) -> np.ndarray:
        """
        Returns an array of transition realizations (number transitioning
        to outgoing compartments) sampled from Poisson distribution.

        Parameters:
            RNG (np.random.Generator object):
                used to generate random variables and control reproducibility.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                contains positive floats, size equal to length of
                (outgoing transition variables list x
                number of age groups x number of risk groups).
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
                            age_group, risk_group] / num_timesteps)

        return realizations_array

    def get_multinomial_deterministic_realization(self,
                                                  RNG: np.random.Generator,
                                                  num_timesteps: int) -> np.ndarray:
        """
        Deterministic counterpart to get_multinomial_realization --
        uses mean (n x p, i.e. total counts x probability array) as realization
        rather than randomly sampling.

        Parameters:
            RNG (np.random.Generator object):
                NOT USED -- only included so that get_realization has
                the same function arguments regardless of transition type.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                contains positive floats, size equal to
                (length of outgoing transition variables list + 1)
                x number of age groups x number of risk groups --
                note the "+1" corresponds to the multinomial outcome of staying
                in the same epi compartment (not transitioning to any outgoing
                epi compartment).
        """

        probabilities_array = self.get_probabilities_array(num_timesteps)
        return np.asarray(self.origin.current_val * probabilities_array, dtype=int)

    def get_multinomial_taylor_approx_deterministic_realization(self,
                                                                RNG: np.random.Generator,
                                                                num_timesteps: int) -> np.ndarray:
        """
        Deterministic counterpart to get_multinomial_taylor_approx_realization --
        uses mean (n x p, i.e. total counts x probability array) as realization
        rather than randomly sampling.

        Parameters:
            RNG (np.random.Generator object):
                NOT USED -- only included so that get_realization has
                the same function arguments regardless of transition type.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                contains positive floats, size equal to
                (length of outgoing transition variables list + 1)
                x number of age groups x number of risk groups --
                note the "+1" corresponds to the multinomial outcome of staying
                in the same epi compartment (not transitioning to any outgoing
                epi compartment).
        """

        current_rates_array = self.get_current_rates_array()
        return np.asarray(self.origin.current_val * current_rates_array / num_timesteps, dtype=int)

    def get_poisson_deterministic_realization(self,
                                              RNG: np.random.Generator,
                                              num_timesteps: int) -> np.ndarray:
        """
        Deterministic counterpart to get_poisson_realization --
        uses mean (rate array) as realization rather than randomly sampling.

        Parameters:
            RNG (np.random.Generator object):
                NOT USED -- only included so that get_realization has
                the same function arguments regardless of transition type.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                contains positive floats, size equal to
                (length of outgoing transition variables list
                x number of age groups x number of risk groups).
        """

        return np.asarray(self.origin.current_val * self.get_current_rates_array() / num_timesteps, dtype=int)

    def reset(self) -> None:
        self.current_vals_list = []

    def update_transition_variable_realizations(self) -> None:
        """
        Updates current_val attribute on all
        TransitionVariable instances contained in this
        transition variable group.
        """

        # Since the ith element in probabilities_array corresponds to the ith transition variable
        #   in transition_variables, the ith element in multinomial_realizations_list
        #   also corresponds to the ith transition variable in transition_variables
        # Update the current realization of the transition variables contained in this group
        for ix in range(len(self.transition_variables)):
            self.transition_variables[ix].current_val = \
                self.current_vals_list[ix, :, :]


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

    Attributes:
        _transition_type (str):
            only values defined in TransitionTypes Enum are valid, specifying
            probability distribution of transitions between compartments.
        get_current_rate (function):
            provides specific implementation for computing current rate
            as a function of current Subpop simulation state and
            epidemiological parameters.
        current_rate (np.ndarray):
            holds output from get_current_rate method -- used to generate
            random variable realizations for transitions between compartments.
        current_val (np.ndarray):
            holds realization of random variable parameterized by current_rate.
        history_vals_list (list[np.ndarray]):
            each element is the same size of current_val, holds
            history of transition variable realizations for age-risk
            groups -- element t corresponds to previous current_val value
            at end of simulation day t.

    See __init__ docstring for other attributes.
    """

    def __init__(self,
                 origin,
                 destination,
                 transition_type,
                 is_jointly_distributed=False):
        """
        Parameters:
            origin (Compartment):
                the compartment from which Transition Variable exits.
            destination (Compartment):
                compartment that the TransitionVariable enters.
            transition_type (str):
                only values defined in TransitionTypes Enum are valid, specifying
                probability distribution of transitions between compartments.
            is_jointly_distributed (bool):
                indicates if transition quantity must be jointly computed
                (i.e. if there are multiple outflows from the origin compartment).
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

        self.current_rate = None
        self.current_val = 0

        self.history_vals_list = []

    @property
    def transition_type(self) -> TransitionTypes:
        return self._transition_type

    @property
    def is_jointly_distributed(self) -> bool:
        return self._is_jointly_distributed

    @abstractmethod
    def get_current_rate(self,
                         state: SubpopState,
                         params: SubpopParams) -> np.ndarray:
        """
        Computes and returns current rate of transition variable,
        based on current state of the simulation and epidemiological parameters.
        Output should be a numpy array of size A x L, where A is
        number of age groups and L is number of risk groups.

        Args:
            state (SubpopState):
                holds Subpop simulation state (current values of StateVariable instances).
            params (SubpopParams):
                holds values of epidemiological parameters.

        Returns:
            np.ndarray:
                holds age-risk transition rate,
                must be same shape as origin.init_val,
                i.e. be size A x L, where A is number of age groups
                and L is number of risk groups.
        """
        pass

    def update_origin_outflow(self) -> None:
        """
        Adds current realization of TransitionVariable to
            its origin Compartment's current_outflow.
            Used to compute total number leaving that
            origin Compartment.
        """

        self.origin.current_outflow += self.current_val

    def update_destination_inflow(self) -> None:
        """
        Adds current realization of TransitionVariable to
            its destination Compartment's current_inflow.
            Used to compute total number leaving that
            destination Compartment.
        """

        self.destination.current_inflow += self.current_val

    def save_history(self) -> None:
        """
        Saves current value to history by appending current_val attribute
            to history_vals_list in place

        Deep copying is CRUCIAL because current_val is a mutable
            np.ndarray -- without deep copying, history_vals_list would
            have the same value for all elements
        """
        self.history_vals_list.append(copy.deepcopy(self.current_val))

    def clear_history(self) -> None:
        """
        Resets history_vals_list attribute to empty list.
        """

        self.history_vals_list = []

    def get_realization(self,
                        RNG: np.random.Generator,
                        num_timesteps: int) -> np.ndarray:
        """
        This method gets assigned to one of the following methods
            based on the TransitionVariable transition type:
            get_binomial_realization, get_binomial_taylor_approx_realization,
            get_poisson_realization, get_binomial_deterministic_realization,
            get_binomial_taylor_approx_deterministic_realization,
            get_poisson_deterministic_realization. This is done so that
            the same method get_realization can be called regardless of
            transition type.

        Parameters:
            RNG (np.random.Generator object):
                used to generate random variables and control reproducibility.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.
        """

        pass

    def get_binomial_realization(self,
                                 RNG: np.random.Generator,
                                 num_timesteps: int) -> np.ndarray:
        """
        Uses RNG to generate binomial random variable with
            number of trials equal to population count in the
            origin Compartment and probability computed from
            a function of the TransitionVariable's current rate
            -- see the approx_binomial_probability_from_rate
            function for details

        Parameters:
            RNG (np.random.Generator object):
                used to generate random variables and control reproducibility.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.
        
        Returns:
            np.ndarray:
                size A x L, where A is number of age groups and
                L is number of risk groups.
        """

        return RNG.binomial(n=np.asarray(self.base_count, dtype=int),
                            p=approx_binomial_probability_from_rate(self.current_rate, 1.0 / num_timesteps))

    def get_binomial_taylor_approx_realization(self,
                                               RNG: np.random.Generator,
                                               num_timesteps: int) -> np.ndarray:
        """
        Uses RNG to generate binomial random variable with
            number of trials equal to population count in the
            origin Compartment and probability equal to
            the TransitionVariable's current_rate / num_timesteps

        Parameters:
            RNG (np.random.Generator object):
                used to generate random variables and control reproducibility.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                size A x L, where A is number of age groups and L
                is number of risk groups.
        """
        return RNG.binomial(n=np.asarray(self.base_count, dtype=int),
                            p=self.current_rate * (1.0 / num_timesteps))

    def get_poisson_realization(self,
                                RNG: np.random.Generator,
                                num_timesteps: int) -> np.ndarray:
        """
        Uses RNG to generate Poisson random variable with
            rate equal to (population count in the
            origin Compartment x the TransitionVariable's
            current_rate / num_timesteps)

        Parameters:
            RNG (np.random.Generator object):
                used to generate random variables and control reproducibility.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                size A x L, where A is number of age groups and
                L is number of risk groups.
        """
        return RNG.poisson(self.base_count * self.current_rate / float(num_timesteps))

    def get_binomial_deterministic_realization(self,
                                               RNG: np.random.Generator,
                                               num_timesteps: int) -> np.ndarray:
        """
        Deterministically returns mean of binomial distribution
            (number of trials x probability), where number of trials
            equals population count in the origin Compartment and
            probability is computed from a function of the TransitionVariable's
            current rate -- see the approx_binomial_probability_from_rate
            function for details

        Parameters:
            RNG (np.random.Generator object):
                NOT USED -- only included so that get_realization has
                the same function arguments regardless of transition type.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                size A x L, where A is number of age groups and
                L is number of risk groups.
        """

        return np.asarray(self.base_count *
                          approx_binomial_probability_from_rate(self.current_rate, 1.0 / num_timesteps),
                          dtype=int)

    def get_binomial_taylor_approx_deterministic_realization(self,
                                                             RNG: np.random.Generator,
                                                             num_timesteps: int) -> np.ndarray:
        """
        Deterministically returns mean of binomial distribution
            (number of trials x probability), where number of trials
            equals population count in the origin Compartment and
            probability equals the TransitionVariable's current rate /
            num_timesteps

        Parameters:
            RNG (np.random.Generator object):
                NOT USED -- only included so that get_realization has
                the same function arguments regardless of transition type.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                size A x L, where A is number of age groups and
                L is number of risk groups.
        """

        return np.asarray(self.base_count * self.current_rate / num_timesteps, dtype=int)

    def get_poisson_deterministic_realization(self,
                                              RNG: np.random.Generator,
                                              num_timesteps: int) -> np.ndarray:
        """
        Deterministically returns mean of Poisson distribution,
            givey by (population count in the origin Compartment x
            TransitionVariable's current rate / num_timesteps)

        Parameters:
            RNG (np.random.Generator object):
                NOT USED -- only included so that get_realization has
                the same function arguments regardless of transition type.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                size A x L, where A is number of age groups and
                L is number of risk groups.
        """

        return np.asarray(self.base_count * self.current_rate / num_timesteps, dtype=int)

    @property
    def base_count(self) -> np.ndarray:
        return self.origin.current_val


class StateVariable:
    """
    Parent class of Compartment, EpiMetric, DynamicVal, and Schedule
    classes. All subclasses have the common attributes "init_val" and "current_val."
    """

    def __init__(self, init_val):
        self.init_val = init_val
        self.current_val = copy.deepcopy(init_val)


class Compartment(StateVariable):
    """
    Class for epidemiological compartments (e.g. Susceptible,
        Exposed, Infected, etc...).

    Inherits attributes from StateVariable.

    Attributes:
        current_val (np.ndarray):
            same size as init_val, holds current value of Compartment
            for age-risk groups.
        current_inflow (np.ndarray):
            same size as current_val, used to sum up all
            transition variable realizations incoming to this compartment
            for age-risk groups.
        current_outflow (np.ndarray):
            same size of current_val, used to sum up all
            transition variable realizations outgoing from this compartment
            for age-risk groups.
        history_vals_list (list[np.ndarray]):
            each element is the same size of current_val, holds
            history of compartment states for age-risk groups --
            element t corresponds to previous current_val value at
            end of simulation day t.
    """

    def __init__(self,
                 init_val):
        super().__init__(init_val)

        self.current_inflow = np.zeros(np.shape(init_val))
        self.current_outflow = np.zeros(np.shape(init_val))

        self.history_vals_list = []

    def update_current_val(self) -> None:
        """
        Updates current_val attribute in-place by adding
            current_inflow (sum of all incoming transition variables'
            realizations) and subtracting current outflow (sum of all
            outgoing transition variables' realizations)
        """
        self.current_val += self.current_inflow - self.current_outflow

    def reset_inflow(self) -> None:
        """
        Resets current_inflow attribute to np.ndarray of zeros.
        """
        self.current_inflow = np.zeros(np.shape(self.current_inflow))

    def reset_outflow(self) -> None:
        """
        Resets current_outflow attribute to np.ndarray of zeros.
        """
        self.current_outflow = np.zeros(np.shape(self.current_outflow))

    def save_history(self) -> None:
        """
        Saves current value to history by appending current_val attribute
            to history_vals_list in place

        NOTE:
            deep copying is CRUCIAL because current_val is a mutable
            np.ndarray -- without deep copying, history_vals_list would
            have the same value for all elements
        """
        self.history_vals_list.append(copy.deepcopy(self.current_val))

    def clear_history(self) -> None:
        """
        Resets history_vals_list attribute to empty list.
        """

        self.history_vals_list = []


class EpiMetric(StateVariable, ABC):
    """
    Abstract base class for epi metrics in epidemiological model.

    This is intended for variables that are aggregate deterministic functions of
    the subpop simulation state (including epi compartment values, other parameters,
    and time.)

    For example, population-level immunity variables should be
    modeled as a EpiMetric subclass, with a concrete
    implementation of the abstract method get_change_in_current_val.

    Inherits attributes from StateVariable.

    Attributes:
        current_val (np.ndarray):
            same size as init_val, holds current value of State Variable
            for age-risk groups.
        change_in_current_val : (np.ndarray):
            initialized to None, but during simulation holds change in
            current value of EpiMetric for age-risk groups
            (size A x L, where A is number of risk groups and L is number
            of age groups).
        history_vals_list (list[np.ndarray]):
            each element is the same size of current_val, holds
            history of transition variable realizations for age-risk
            groups -- element t corresponds to previous current_val value
            at end of simulation day t.

    See __init__ docstring for other attributes.
    """

    def __init__(self,
                 init_val):
        """
        Args:
            init_val (np.ndarray):
                2D array that contains nonnegative floats,
                corresponding to initial value of dynamic val,
                where i,jth entry corresponds to age group i and
                risk group j.
        """

        super().__init__(init_val)

        self.change_in_current_val = None
        self.history_vals_list = []

    @abstractmethod
    def get_change_in_current_val(self,
                                  state: SubpopState,
                                  params: SubpopParams,
                                  num_timesteps: int) -> np.ndarray:
        """
        Computes and returns change in current value of dynamic val,
        based on current state of the simulation and epidemiological parameters.
        ***NOTE: OUTPUT SHOULD ALREADY BE SCALED BY NUM_TIMESTEPS.
        Output should be a numpy array of size A x L, where A
        is number of age groups and L is number of risk groups.

        Args:
            state (SubpopState):
                holds Subpop simulation state (current values of StateVariable
                instances).
            params (SubpopParams):
                holds values of epidemiological parameters.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                size A x L, where A is number of age groups and
                L is number of risk groups.
        """
        pass

    def update_current_val(self) -> None:
        """
        Adds change_in_current_val attribute to current_val attribute
            in-place.
        """

        self.current_val += self.change_in_current_val

    def save_history(self) -> None:
        """
        Saves current value to history by appending current_val attribute
            to history_vals_list in place

        Deep copying is CRUCIAL because current_val is a mutable
            np.ndarray -- without deep copying, history_vals_list would
            have the same value for all elements
        """
        self.history_vals_list.append(copy.deepcopy(self.current_val))

    def clear_history(self) -> None:
        """
        Resets history_vals_list attribute to empty list.
        """

        self.history_vals_list = []


class DynamicVal(StateVariable, ABC):
    """
    Abstract base class for variables that dynamically adjust
    their values based the current values of other StateVariable instances.

    This class should model social distancing (and more broadly,
    staged-alert policies). For example, if we consider a
    case where transmission rates decrease when number infected
    increase above a certain level, we can create a subclass of
    DynamicVal that models a coefficient that modifies transmission
    rates, depending on the epi compartments corresponding to
    infected people.

    Inherits attributes from StateVariable.

    Attributes:
        history_vals_list (list[np.ndarrays]):
            each element is the same size of current_val, holds
            history of transition variable realizations for age-risk
            groups -- element t corresponds to previous current_val value
            at end of simulation day t.

    See __init__ docstring for other attributes.
    """

    def __init__(self,
                 init_val: Optional[Union[np.ndarray, float]] = None,
                 is_enabled: Optional[bool] = False):
        """

        Args:
            init_val (Optional[Union[np.ndarray, float]]):
                starting value(s) at the beginning of the simulation
            is_enabled (Optional[bool]):
                if False, this dynamic value does not get updated
                during the simulation and defaults to its init_val.
                This is designed to allow easy toggling of
                simulations with or without staged alert policies
                and other interventions.
        """

        super().__init__(init_val)
        self.is_enabled = is_enabled
        self.history_vals_list = []

    @abstractmethod
    def update_current_val(self,
                           state: SubpopState,
                           params: SubpopParams) -> None:
        """
        Args:
            state (SubpopState):
                holds Subpop simulation state (current values of StateVariable
                instances).
            params (SubpopParams):
                holds values of epidemiological parameters.
        """

    def save_history(self) -> None:
        """
        Saves current value to history by appending current_val attribute
            to history_vals_list in place

        Deep copying is CRUCIAL because current_val is a mutable
            np.ndarray -- without deep copying, history_vals_list would
            have the same value for all elements
        """
        self.history_vals_list.append(copy.deepcopy(self.current_val))

    def clear_history(self) -> None:
        self.history_vals_list = []


@dataclass
class Schedule(StateVariable, ABC):
    """
    Abstract base class for variables that are functions of real-world
    dates -- for example, contact matrices (which depend on the day of
    the week and whether the current day is a holiday), historical
    vaccination data, and seasonality.

    Inherits attributes from StateVariable.

    See __init__ docstring for other attributes.
    """

    def __init__(self,
                 init_val: Optional[Union[np.ndarray, float]] = None,
                 timeseries_df: Optional[dict] = None):
        """
        Args:
            init_val (Optional[Union[np.ndarray, float]]):
                starting value(s) at the beginning of the simulation
            timeseries_df (Optional[pd.DataFrame] = None):
                has a "date" column with strings in format "YYYY-MM-DD"
                of consecutive calendar days, and other columns
                corresponding to values on those days
        """

        super().__init__(init_val)
        self.timeseries_df = timeseries_df

    @abstractmethod
    def update_current_val(self, current_date: datetime.date) -> None:
        """
        Subpop classes must provide a concrete implementation of
        updating self.current_val in-place.

        Args:
            current_date (date):
                real-world date corresponding to
                model's current simulation day.
        """
        pass


class InteractionTerm(StateVariable, ABC):

    def __init__(self):
        self.history_vals_list = []

    @abstractmethod
    def update_current_val(self,
                           inter_subpop_repo: InterSubpopRepo,
                           subpop_params: SubpopParams) -> None:
        """
        Subpop classes must provide a concrete implementation of
        updating self.current_val in-place.

        Args:
            inter_subpop_repo (InterSubpopRepo):
                manages collection of subpop models with
                methods for querying information.
            travel_proportions (pd.DataFrame):
                each i-j pair is the proportion of people in
                subpopulation i who travel to j (values must be
                in [0,1]), and row and column names correspond
                to names of subpopulations
            subpop_params (SubpopParams):
                holds values of Subpop's epidemiological parameters.
        """

        pass

    def save_history(self) -> None:
        """
        Saves current value to history by appending current_val attribute
            to history_vals_list in place

        Deep copying is CRUCIAL because current_val is a mutable
            np.ndarray -- without deep copying, history_vals_list would
            have the same value for all elements
        """
        self.history_vals_list.append(copy.deepcopy(self.current_val))

    def clear_history(self) -> None:
        self.history_vals_list = []


class MetapopModel(ABC):

    def __init__(self,
                 subpop_models: dict,
                 travel_proportions: pd.DataFrame,
                 name: str = ""):
        
        self.subpop_models = sc.objdict(subpop_models)

        self.assign_inter_subpop_repo(subpop_models, travel_proportions)
        
        self.name = name
        
        for model in subpop_models.values():
            model.metapop_model = self
            model.interaction_terms = model.create_interaction_terms()
            model.state.interaction_terms = model.interaction_terms

    @abstractmethod
    def assign_inter_subpop_repo(self,
                                 subpop_models,
                                 travel_proportions):
        pass

    def extract_states_dict_from_models_dict(self,
                                             models_dict: sc.objdict) -> sc.objdict:
        states_dict = \
            sc.objdict({name: model.state for name, model in models_dict.items()})

        return states_dict

    def simulate_until_time_period(self, last_simulation_day) -> None:
        """
        Advance simulation model time until last_simulation_day.

        Advance time by iterating over each subpopulation for
        each simulation day, which are simulated by iterating
        through discretized timesteps.

        Save daily simulation data as history on each Compartment
        instance.

        Args:
            last_simulation_day (positive int):
                stop simulation at last_simulation_day (i.e. exclusive,
                simulate up to but not including last_simulation_day).
        """

        for subpop_model in self.subpop_models.values():
            subpop_model.simulate_until_time_period(last_simulation_day)


class SubpopModel(ABC):
    """
    Contains and manages all necessary components for
    simulating a compartmental model, including compartments
    epi metrics, dynamic vals, a data container for the current simulation
    state, transition variables and transition variable groups,
    epidemiological parameters, simulation experiment configuration
    parameters, and a random number generator.

    All city-level models, regardless of disease type and
    compartment/transition structure, are instances of this class.

    When creating an instance, the order of elements does not matter
    within compartments, epi_metrics, dynamic_vals,
    transition_variables, and transition_variable_groups.
    The "flow" and "physics" information are stored on the objects.

    Attributes:
        compartments (sc.objdict):
            objdict of all the model's Compartment instances.
        transition_variables (sc.objdict):
            objdict of all the model's TransitionVariable instances.
        transition_variable_groups (sc.objdict):
            objdict of all the model's TransitionVariableGroup instances.
        epi_metrics (sc.objdict):
            objdict of all the model's EpiMetric instances.
        dynamic_vals (sc.objdict):
            objdict of all the model's DynamicVal instances.
        schedules (sc.objdict):
            objdict of all the model's Schedule instances.
        params (SubpopParams):
            data container for the model's epidemiological parameters,
            such as the "Greek letters" characterizing sojourn times
            in compartments.
        config (Config):
            data container for the model's simulation configuration values.
        RNG (np.random.Generator object):
            used to generate random variables and control reproducibility.
        current_simulation_day (int):
            tracks current simulation day -- incremented by +1
            when config.timesteps_per_day discretized timesteps
            have completed.
        current_real_date (datetime.date):
            tracks real-world date -- advanced by +1 day when
            config.timesteps_per_day discretized timesteps
            have completed.
        state (SubpopState):
        metapop_model (Optional[MetapopModel]):
            if not None, is MetapopModel instance associated with
            this SubpopModel.
        metapop_state (Optional[MetapopState]):
            if not None, is the MetapopState instance 
            of the MetapopModel instance associated with this 
            SubpopModel. 
        metapop_params (Optional[MetapopParams]):
            if not None, is the MetapopParams instance
            of the MetapopModel instance associated with this
            SubpopModel.

    See __init__ docstring for other attributes.
    """

    def __init__(self,
                 state: SubpopState,
                 params: SubpopParams,
                 config: Config,
                 RNG: np.random.Generator,
                 name: str = "",
                 metapop_model: MetapopModel = None):

        self.state = copy.deepcopy(state)
        self.params = copy.deepcopy(params)
        self.config = copy.deepcopy(config)

        self.RNG = RNG

        self.current_simulation_day = 0
        self.start_real_date = self.get_start_real_date()
        self.current_real_date = self.start_real_date

        self.metapop_model = None
        self.name = name

        self.interaction_terms = self.create_interaction_terms()
        self.compartments = self.create_compartments()
        self.transition_variables = self.create_transition_variables()
        self.transition_variable_groups = self.create_transition_variable_groups()

        # Some epi metrics depend on transition variables, so
        #   set up epi metrics after transition variables
        self.epi_metrics = self.create_epi_metrics()
        self.dynamic_vals = self.create_dynamic_vals()
        self.schedules = self.create_schedules()

        self.all_state_variables = {**self.interaction_terms,
                                    **self.compartments,
                                    **self.epi_metrics,
                                    **self.dynamic_vals,
                                    **self.schedules}

        # The model's state also has access to the model's
        #   compartments, epi_metrics, dynamic_vals, and schedules --
        #   so that state can easily retrieve each object's
        #   current_val and store it
        self.state.interaction_terms = self.interaction_terms
        self.state.compartments = self.compartments
        self.state.epi_metrics = self.epi_metrics
        self.state.dynamic_vals = self.dynamic_vals
        self.state.schedules = self.schedules

        self.params.total_pop_age_risk = self.compute_total_pop_age_risk()

    def compute_total_pop_age_risk(self) -> np.ndarray:
        """

        Returns:
             total_pop_age_risk (np.ndarray):

        """

        total_pop_age_risk = np.zeros((self.params.num_age_groups,
                                       self.params.num_risk_groups))

        # At initialization (before simulation is run), each
        #   compartment's current val is equivalent to the initial val
        #   specified in the state variables' init val JSON.
        for compartment in self.compartments.values():
            total_pop_age_risk += compartment.current_val

        return total_pop_age_risk

    def get_start_real_date(self):
        """
        Fetches start_real_date from config -- converts to
            proper datetime.date format if originally given as
            string.

        Returns:
            start_real_date (datetime.date):
                real-world date that corresponds to start of
                simulation.
        """

        start_real_date = self.config.start_real_date

        if not isinstance(start_real_date, datetime.date):
            try:
                start_real_date = \
                    datetime.datetime.strptime(start_real_date, "%Y-%m-%d").date()
            except ValueError:
                print("Error: The date format should be YYYY-MM-DD.")

        return start_real_date

    @abstractmethod
    def create_interaction_terms(self) -> sc.objdict:
        pass

    @abstractmethod
    def create_compartments(self) -> sc.objdict:
        pass

    @abstractmethod
    def create_transition_variables(self) -> sc.objdict:
        pass

    @abstractmethod
    def create_transition_variable_groups(self) -> sc.objdict:
        pass

    @abstractmethod
    def create_epi_metrics(self) -> sc.objdict:
        pass

    @abstractmethod
    def create_dynamic_vals(self) -> sc.objdict:
        pass

    @abstractmethod
    def create_schedules(self) -> sc.objdict:
        pass

    def modify_random_seed(self, new_seed_number) -> None:
        """
        Modifies model's RNG attribute in-place to new generator
        seeded at new_seed_number.

        Args:
            new_seed_number (int):
                used to re-seed model's random number generator.
        """

        self._bit_generator = np.random.MT19937(seed=new_seed_number)
        self.RNG = np.random.Generator(self._bit_generator)

    def simulate_until_time_period(self, last_simulation_day) -> None:
        """
        Advance simulation model time until last_simulation_day.

        Advance time by iterating through simulation days,
        which are simulated by iterating through discretized
        timesteps.

        Save daily simulation data as history on each Compartment
        instance.

        Args:
            last_simulation_day (positive int):
                stop simulation at last_simulation_day (i.e. exclusive,
                simulate up to but not including last_simulation_day).
        """

        if self.current_simulation_day > last_simulation_day:
            raise SubpopModelError(f"Current day counter ({self.current_simulation_day}) "
                                   f"exceeds last simulation day ({last_simulation_day}).")

        save_daily_history = self.config.save_daily_history
        timesteps_per_day = self.config.timesteps_per_day

        # last_simulation_day is exclusive endpoint
        while self.current_simulation_day < last_simulation_day:

            self.prepare_daily_state()

            self.simulate_timesteps(timesteps_per_day)

            if save_daily_history:
                self.save_daily_history()

            self.increment_simulation_day()

    def simulate_timesteps(self,
                           num_timesteps: int) -> None:
        """
        Subroutine for simulate_until_time_period.

        Iterates through discretized timesteps to simulate next
        simulation day. Granularity of discretization is given by
        attribute config.timesteps_per_day.

        Properly scales transition variable realizations and changes
        in dynamic vals by specified timesteps per day.

        Parameters:
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.
        """

        for timestep in range(num_timesteps):

            self.update_transition_rates()

            self.sample_transitions()

            self.update_epi_metrics()

            self.update_compartments()

            self.state.sync_to_current_vals(self.epi_metrics)
            self.state.sync_to_current_vals(self.compartments)

    def prepare_daily_state(self) -> None:
        """
        At beginning of each day, update current value of
        interaction terms, schedules, dynamic values --
        note that these are only updated once a day, not
        for every discretized timestep.
        """

        subpop_state = self.state
        subpop_params = self.params
        current_real_date = self.current_real_date

        schedules = self.schedules
        dynamic_vals = self.dynamic_vals
        interaction_terms = self.interaction_terms

        # Update schedules for current day
        for schedule in schedules.values():
            schedule.update_current_val(current_real_date)

        self.state.sync_to_current_vals(self.schedules)

        # Update dynamic values for current day
        for dval in dynamic_vals.values():
            if dval.is_enabled:
                dval.update_current_val(subpop_state, subpop_params)

        self.state.sync_to_current_vals(self.dynamic_vals)

        # Update interaction terms for current day
        for iterm in interaction_terms.values():
            iterm.update_current_val(self.metapop_model.inter_subpop_repo, subpop_params)

        # Sync Subpop simulation state
        self.state.sync_to_current_vals(self.interaction_terms)


    def update_epi_metrics(self):

        state = self.state
        params = self.params
        timesteps_per_day = self.config.timesteps_per_day

        for metric in self.epi_metrics.values():
            metric.change_in_current_val = \
                metric.get_change_in_current_val(state,
                                                 params,
                                                 timesteps_per_day)
            metric.update_current_val()

    def update_transition_rates(self):

        state = self.state
        params = self.params

        for tvar in self.transition_variables.values():
            tvar.current_rate = tvar.get_current_rate(state, params)

    def sample_transitions(self):

        RNG = self.RNG
        timesteps_per_day = self.config.timesteps_per_day

        # Obtain transition variable realizations for jointly distributed transition variables
        #   (i.e. when there are multiple transition variable outflows from an epi compartment)
        for tvargroup in self.transition_variable_groups.values():
            tvargroup.current_vals_list = tvargroup.get_joint_realization(RNG,
                                                                          timesteps_per_day)
            tvargroup.update_transition_variable_realizations()

        # Obtain transition variable realizations for marginally distributed transition variables
        #   (i.e. when there is only one transition variable outflow from an epi compartment)
        # If transition variable is jointly distributed, then its realization has already
        #   been computed by its transition variable group container previously,
        #   so skip the marginal computation
        for tvar in self.transition_variables.values():
            if not tvar.is_jointly_distributed:
                tvar.current_val = tvar.get_realization(RNG, timesteps_per_day)

    def update_compartments(self):

        for tvar in self.transition_variables.values():
            tvar.update_origin_outflow()
            tvar.update_destination_inflow()

        for compartment in self.compartments.values():
            compartment.update_current_val()

            compartment.reset_inflow()
            compartment.reset_outflow()

    def increment_simulation_day(self) -> None:
        """
        Move to next day in simulation
        """

        self.current_simulation_day += 1
        self.current_real_date += datetime.timedelta(days=1)

    def save_daily_history(self) -> None:
        """
        Update history at end of each day, not at end of every
           discretization timestep, to be efficient.
        Update history of state variables other than Schedule
           instances -- schedules do not have history
           TransitionVariableGroup instances also do not
           have history, so do not include.
        """
        for svar in self.interaction_terms.values() + \
                    self.compartments.values() + \
                    self.epi_metrics.values() + \
                    self.dynamic_vals.values():
            svar.save_history()

    def reset_simulation(self) -> None:
        """
        Reset simulation in-place. Subsequent method calls of
        simulate_until_time_period start from day 0, with original
        day 0 state.

        Returns current_simulation_day to 0.
        Restores state values to initial values.
        Clears history on model's state variables.
        Resets transition variables' current_val attribute to 0.

        WARNING:
            DOES NOT RESET THE MODEL'S RANDOM NUMBER GENERATOR TO
            ITS INITIAL STARTING SEED. RANDOM NUMBER GENERATOR WILL CONTINUE
            WHERE IT LEFT OFF.

        Use method modify_random_seed to reset model's RNG to its
        initial starting seed.
        """

        self.current_simulation_day = 0
        self.current_real_date = self.start_real_date

        # AGAIN, MUST BE CAREFUL ABOUT MUTABLE NUMPY ARRAYS -- MUST USE DEEP COPY
        for svar in self.all_state_variables.values():
            setattr(svar, "current_val", copy.deepcopy(svar.init_val))

        self.state.sync_to_current_vals(self.all_state_variables)
        self.clear_history()

    def clear_history(self) -> None:
        """
        Resets history_vals_list attribute of each Compartment,
            EpiMetric, and DynamicVal to an empty list.
        """

        # Schedules do not have history since they are deterministic
        for svar in self.compartments.values() + \
                    self.epi_metrics.values() + \
                    self.dynamic_vals.values():
            svar.clear_history()

        for tvar in self.transition_variables.values():
            tvar.current_rate = None
            tvar.current_val = 0.0

        for tvargroup in self.transition_variable_groups.values():
            tvargroup.current_vals_list = []

    def find_name_by_compartment(self,
                                 target_compartment: Compartment):
        for name, compartment in self.compartments.items():
            if compartment == target_compartment:
                return name

    def display(self) -> None:
        """
        Prints structure of model (compartments and linkages),
            transition variables, epi metrics, schedules,
            and dynamic values.
        """

        # We build origin_dict so that we can print
        #   compartment transitions in an easy-to-read way --
        #   for connections between origin --> destination,
        #   we print all connections with the same origin
        #   consecutively
        origin_dict = defaultdict(list)

        # Each key in origin_dict is a string corresponding to
        #   an origin (Compartment) name
        # Each val in origin_dict is a list of 3-tuples
        # Each 3-tuple has the name of a destination (Compartment)
        #   connected to the given origin, the name of the transition
        #   variable connecting the origin and destination,
        #   and Boolean indicating if the transition variable is jointly
        #   distributed
        for tvar_name, tvar in self.transition_variables.items():
            origin_dict[self.find_name_by_compartment(tvar.origin)].append(
                (self.find_name_by_compartment(tvar.destination),
                 tvar_name, tvar.is_jointly_distributed))

        print(f"\n>>> Displaying SubpopModel {self.name}")

        print("\nCompartments and transition variables")
        print("=====================================")
        for origin_name, origin in self.compartments.items():
            for output in origin_dict[origin_name]:
                if output[2]:
                    print(f"{origin_name} --> {output[0]}, via {output[1]}: jointly distributed")
                else:
                    print(f"{origin_name} --> {output[0]}, via {output[1]}")

        print("\nEpi metrics")
        print("===========")
        for name in self.epi_metrics.keys():
            print(f"{name}")

        print("\nSchedules")
        print("=========")
        for name in self.schedules.keys():
            print(f"{name}")

        print("\nDynamic values")
        print("==============")
        for name, dynamic_val in self.dynamic_vals.items():
            if dynamic_val.is_enabled:
                print(f"{name}: enabled")
            else:
                print(f"{name}: disabled")
        print("\n")



