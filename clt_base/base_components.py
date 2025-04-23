from .utils import np, sc, copy, ABC, abstractmethod, dataclass, \
    Optional, Enum, datetime, pd, TypedDict
from collections import defaultdict


class MetapopModelError(Exception):
    """Custom exceptions for metapopulation simulation model errors."""
    pass


class SubpopModelError(Exception):
    """Custom exceptions for subpopulation simulation model errors."""
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
    occurring in the next time interval of length `interval_length`,
    assuming the number of events occurring in time interval
    follows a Poisson distribution with given rate parameter.

    The probability of 0 events in `interval_length` is
    e^(-`rate` * `interval_length`), so the probability of any event
    in `interval_length` is 1 - e^(-`rate` * `interval_length`).

    Rate must be |A| x |R| `np.ndarray`, where |A| is the number of
    age groups and |R| is the number of risk groups. Rate is transformed to
    |A| x |R| `np.ndarray` corresponding to probabilities.

    Parameters:
        rate (np.ndarray):
            dimension |A| x |R| (number of age groups x number of risk groups),
            rate parameters in a Poisson distribution.
        interval_length (positive int):
            length of time interval in simulation days.

    Returns:
        np.ndarray:
            array of positive scalars, dimension |A| x |R|
    """

    return 1 - np.exp(-rate * interval_length)


@dataclass
class Config:
    """
    Stores simulation configuration values.

    Attributes:
        timesteps_per_day (int):
            number of discretized timesteps within a simulation
            day -- more `timesteps_per_day` mean smaller discretization
            time intervals, which may cause the model to run slower.
        transition_type (str):
            valid value must be from `TransitionTypes`, specifying
            the probability distribution of transitions between
            compartments.
        start_real_date (datetime.date):
            actual date that aligns with the beginning of the simulation.
        save_daily_history (bool):
            set to `True` to save `current_val` of `StateVariable` to history after each
            simulation day -- set to `False` if want speedier performance.
        save_transition_variables_history (bool):
            set to `True` to save `current_val` of `TransitionVariable` to history
            after each TIMESTEP -- note that this makes the simulation execution time
            extremely slow -- set to `False` if want speedier performance.
    """

    timesteps_per_day: int = 7
    transition_type: str = TransitionTypes.BINOMIAL
    start_real_date: datetime.time = datetime.datetime.strptime("2024-10-31",
                                                                "%Y-%m-%d").date()
    save_daily_history: bool = True
    save_transition_variables_history: bool = False


@dataclass
class SubpopParams(ABC):
    """
    Data container for pre-specified and fixed epidemiological
    parameters in model.

    Assume that `SubpopParams` fields are constant or piecewise
    constant throughout the simulation. For variables that
    are more complicated and time-dependent, use an `EpiMetric`
    instead.
    """

    pass


class InterSubpopRepo(ABC):
    """
    Holds collection of `SubpopState` instances, with
        actions to query and interact with them.

    Attributes:
        subpop_models (sc.objdict):
            keys are unique names of subpopulation models,
            values are their respective `SubpopModel` instances --
            this dictionary contains all `SubpopModel` instances
            that comprise a `MetapopModel` instance.
    """

    def __init__(self,
                 subpop_models: Optional[dict] = None):
        self.subpop_models = sc.objdict(subpop_models)

        # The "name" argument for instantiating a SubpopModel
        #   is optional -- but SubpopModel instances must have
        #   names when used in a MetapopModel
        # So, we assign names to each SubpopModel based on
        #   the keys of the dictionary that creates the InterSubpopRepo
        for name, model in subpop_models.items():
            model.name = name

    @abstractmethod
    def compute_shared_quantities(self):
        """
        Subclasses must provide concrete implementation. This method
        is called by the `MetapopModel` instance at the beginning of
        each simulation day, before each `SubpopModel` simulates that day.

        Note: often, `InteractionTerm`s across `SubpopModel`s share similar
        terms in their computation. This `self.compute_shared_quantities`
        method computes such similar terms up front to reduce redundant
        computation.
        """

        pass

    def update_all_interaction_terms(self):
        """
        Updates `SubpopState` of each `SubpopModel` in
        `self.subpop_models` to reflect current values of each
        `InteractionTerm` on that `SubpopModel`.
        """

        for subpop_model in self.subpop_models.values():
            for iterm in subpop_model.interaction_terms.values():
                iterm.update_current_val(self,
                                         subpop_model.params)
            subpop_model.state.sync_to_current_vals(subpop_model.interaction_terms)


@dataclass
class SubpopState(ABC):
    """
    Holds current values of `SubpopModel`'s simulation state.
    """

    def sync_to_current_vals(self, lookup_dict: dict):
        """
        Updates `SubpopState`'s attributes according to
        data in `lookup_dict.` Keys of `lookup_dict` must match
        names of attributes of `SubpopState` instance.
        """

        for name, item in lookup_dict.items():
            setattr(self, name, item.current_val)


class StateVariable:
    """
    Parent class of `InteractionTerm`, `Compartment`, `EpiMetric`,
    `DynamicVal`, and `Schedule` classes. All subclasses have the
    common attributes `self.init_val` and `self.current_val`.

    Attributes:
        init_val (np.ndarray):
            holds initial value of `StateVariable` for age-risk groups.
        current_val (np.ndarray):
            same size as `self.init_val`, holds current value of `StateVariable`
            for age-risk groups.
        history_vals_list (list[np.ndarray]):
            each element is the same size of `self.current_val`, holds
            history of compartment states for age-risk groups --
            element t corresponds to previous `self.current_val` value at
            end of simulation day t.
    """

    def __init__(self, init_val=None):
        self._init_val = init_val
        self.current_val = copy.deepcopy(init_val)
        self.history_vals_list = []

    @property
    def init_val(self):
        return self._init_val

    @init_val.setter
    def init_val(self, value):
        """
        We need to use properties/setters because when we change
            `init_val`, we want `current_val` to be updated too!
        """
        self._init_val = value
        self.current_val = copy.deepcopy(value)

    def save_history(self) -> None:
        """
        Saves current value to history by appending `self.current_val` attribute
            to `self.history_vals_list` in place.

        Deep copying is CRUCIAL because `self.current_val` is a mutable
            `np.ndarray` -- without deep copying, `self.history_vals_list` would
            have the same value for all elements.
        """
        self.history_vals_list.append(copy.deepcopy(self.current_val))

    def reset(self) -> None:
        """
        Resets `self.current_val` to `self.init_val`
        and resets `self.history_vals_list` attribute to empty list.
        """

        self.current_val = copy.deepcopy(self.init_val)
        self.history_vals_list = []


class Compartment(StateVariable):
    """
    Class for epidemiological compartments (e.g. Susceptible,
        Exposed, Infected, etc...).

    Inherits attributes from `StateVariable`.

    Attributes:
        current_inflow (np.ndarray):
            same size as `self.current_val`, used to sum up all
            transition variable realizations incoming to this compartment
            for age-risk groups.
        current_outflow (np.ndarray):
            same size of `self.current_val`, used to sum up all
            transition variable realizations outgoing from this compartment
            for age-risk groups.
    """

    def __init__(self,
                 init_val):
        super().__init__(np.asarray(init_val, dtype=float))

        self.current_inflow = np.zeros(np.shape(init_val))
        self.current_outflow = np.zeros(np.shape(init_val))

    def update_current_val(self) -> None:
        """
        Updates `self.current_val` attribute in-place by adding
            `self.current_inflow` (sum of all incoming transition variables'
            realizations) and subtracting current outflow (sum of all
            outgoing transition variables' realizations).
        """
        self.current_val = self.current_val + self.current_inflow - self.current_outflow

    def reset_inflow(self) -> None:
        """
        Resets `self.current_inflow` attribute to np.ndarray of zeros.
        """
        self.current_inflow = np.zeros(np.shape(self.current_inflow))

    def reset_outflow(self) -> None:
        """
        Resets `self.current_outflow` attribute to np.ndarray of zeros.
        """
        self.current_outflow = np.zeros(np.shape(self.current_outflow))


class TransitionVariable(ABC):
    """
    Abstract base class for transition variables in
    epidemiological model.

    For example, in an S-I-R model, the new number infected
    every iteration (the number going from S to I) in an iteration
    is modeled as a `TransitionVariable` subclass, with a concrete
    implementation of the abstract method `self.get_current_rate`.

    When an instance is initialized, its `self.get_realization` attribute
    is dynamically assigned, just like in the case of
    `TransitionVariableGroup` instantiation.

    Attributes:
        _transition_type (str):
            only values defined in `TransitionTypes` are valid, specifying
            probability distribution of transitions between compartments.
        get_current_rate (function):
            provides specific implementation for computing current rate
            as a function of current subpopulation simulation state and
            epidemiological parameters.
        current_rate (np.ndarray):
            holds output from `self.get_current_rate` method -- used to generate
            random variable realizations for transitions between compartments.
        current_val (np.ndarray):
            holds realization of random variable parameterized by
            `self.current_rate`.
        history_vals_list (list[np.ndarray]):
            each element is the same size of `self.current_val`, holds
            history of transition variable realizations for age-risk
            groups -- element t corresponds to previous `self.current_val`
            value at end of simulation day t.

    See `__init__` docstring for other attributes.
    """

    def __init__(self,
                 origin: Compartment,
                 destination: Compartment,
                 transition_type: TransitionTypes,
                 is_jointly_distributed: str = False):
        """
        Parameters:
            origin (Compartment):
                `Compartment` from which `TransitionVariable` exits.
            destination (Compartment):
                `Compartment` that the `TransitionVariable` enters.
            transition_type (str):
                only values defined in `TransitionTypes` are valid, specifying
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
        Output should be a numpy array of size |A| x |R|, where |A| is the
        number of age groups and |R| is number of risk groups.

        Args:
            state (SubpopState):
                holds subpopulation simulation state
                (current values of `StateVariable` instances).
            params (SubpopParams):
                holds values of epidemiological parameters.

        Returns:
            np.ndarray:
                holds age-risk transition rate,
                must be same shape as origin.init_val,
                i.e. be size |A| x |R|, where |A| is the number of age groups
                and |R| is number of risk groups.
        """
        pass

    def update_origin_outflow(self) -> None:
        """
        Adds current realization of `TransitionVariable` to
            its origin `Compartment`'s current_outflow.
            Used to compute total number leaving that
            origin `Compartment`.
        """

        self.origin.current_outflow = self.origin.current_outflow + self.current_val

    def update_destination_inflow(self) -> None:
        """
        Adds current realization of `TransitionVariable` to
            its destination `Compartment`'s `current_inflow`.
            Used to compute total number leaving that
            destination `Compartment`.
        """

        self.destination.current_inflow = self.destination.current_inflow + self.current_val

    def save_history(self) -> None:
        """
        Saves current value to history by appending `self.current_val`
            attribute to `self.history_vals_list` in place.

        Deep copying is CRUCIAL because `self.current_val` is a mutable
            np.ndarray -- without deep copying, `self.history_vals_list` would
            have the same value for all elements.
        """
        self.history_vals_list.append(copy.deepcopy(self.current_val))

    def reset(self) -> None:
        """
        Resets `self.history_vals_list` attribute to empty list.
        """

        self.current_rate = None
        self.current_val = 0.0
        self.history_vals_list = []

    def get_realization(self,
                        RNG: np.random.Generator,
                        num_timesteps: int) -> np.ndarray:
        """
        This method gets assigned to one of the following methods
            based on the `TransitionVariable` transition type:
            `self.get_binomial_realization`, `self.get_binomial_taylor_approx_realization`,
            `self.get_poisson_realization`, `self.get_binomial_deterministic_realization`,
            `self.get_binomial_taylor_approx_deterministic_realization`,
            `self.get_poisson_deterministic_realization`. This is done so that
            the same method `self.get_realization` can be called regardless of
            transition type.

        Parameters:
            RNG (np.random.Generator object):
                 used to generate stochastic transitions in the model and control
                 reproducibility.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.
        """

        pass

    def get_binomial_realization(self,
                                 RNG: np.random.Generator,
                                 num_timesteps: int) -> np.ndarray:
        """
        Uses `RNG` to generate binomial random variable with
            number of trials equal to population count in the
            origin `Compartment` and probability computed from
            a function of the `TransitionVariable`'s current rate
            -- see the `approx_binomial_probability_from_rate`
            function for details.

        Parameters:
            RNG (np.random.Generator object):
                 used to generate stochastic transitions in the model and control
                 reproducibility.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                size |A| x |R|, where |A| is the number of age groups and
                |R| is number of risk groups.
        """

        return RNG.binomial(n=np.asarray(self.base_count, dtype=int),
                            p=approx_binomial_probability_from_rate(self.current_rate, 1.0 / num_timesteps))

    def get_binomial_taylor_approx_realization(self,
                                               RNG: np.random.Generator,
                                               num_timesteps: int) -> np.ndarray:
        """
        Uses `RNG` to generate binomial random variable with
            number of trials equal to population count in the
            origin `Compartment` and probability equal to
            the `TransitionVariable`'s `current_rate` / `num_timesteps`.

        Parameters:
            RNG (np.random.Generator object):
                 used to generate stochastic transitions in the model and control
                 reproducibility.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                size |A| x |R|, where |A| is the number of age groups and L
                is number of risk groups.
        """
        return RNG.binomial(n=np.asarray(self.base_count, dtype=int),
                            p=self.current_rate * (1.0 / num_timesteps))

    def get_poisson_realization(self,
                                RNG: np.random.Generator,
                                num_timesteps: int) -> np.ndarray:
        """
        Uses `RNG` to generate Poisson random variable with
            rate equal to (population count in the
            origin `Compartment` x the `TransitionVariable`'s
            `current_rate` / `num_timesteps`)

        Parameters:
            RNG (np.random.Generator object):
                 used to generate stochastic transitions in the model and control
                 reproducibility.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                size |A| x |R|, where |A| is the number of age groups and
                |R| is number of risk groups.
        """
        return RNG.poisson(self.base_count * self.current_rate / float(num_timesteps))

    def get_binomial_deterministic_realization(self,
                                               RNG: np.random.Generator,
                                               num_timesteps: int) -> np.ndarray:
        """
        Deterministically returns mean of binomial distribution
            (number of trials x probability), where number of trials
            equals population count in the origin `Compartment` and
            probability is computed from a function of the `TransitionVariable`'s
            current rate -- see the `approx_binomial_probability_from_rate`
            function for details.

        Parameters:
            RNG (np.random.Generator object):
                NOT USED -- only included so that get_realization has
                the same function arguments regardless of transition type.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                size |A| x |R|, where |A| is the number of age groups and
                |R| is number of risk groups.
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
            equals population count in the origin `Compartment` and
            probability equals the `TransitionVariable`'s `current_rate` /
            `num_timesteps`.

        Parameters:
            RNG (np.random.Generator object):
                NOT USED -- only included so that get_realization has
                the same function arguments regardless of transition type.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                size |A| x |R|, where |A| is the number of age groups and
                |R| is number of risk groups.
        """

        return np.asarray(self.base_count * self.current_rate / num_timesteps, dtype=int)

    def get_poisson_deterministic_realization(self,
                                              RNG: np.random.Generator,
                                              num_timesteps: int) -> np.ndarray:
        """
        Deterministically returns mean of Poisson distribution,
            given by (population count in the origin `Compartment` x
            `TransitionVariable`'s `current_rate` / `num_timesteps`).

        Parameters:
            RNG (np.random.Generator object):
                NOT USED -- only included so that get_realization has
                the same function arguments regardless of transition type.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                size |A| x |R|, where |A| is the number of age groups and
                |R| is number of risk groups.
        """

        return np.asarray(self.base_count * self.current_rate / num_timesteps, dtype=int)

    @property
    def base_count(self) -> np.ndarray:
        return self.origin.current_val


class TransitionVariableGroup:
    """
    Container for `TransitionVariable` objects to handle joint sampling,
    when there are multiple outflows from a single compartment.

    For example, if all outflows of compartment `H` are: `R` and `D`,
    i.e. from the hospital, people either recover or die,
    a `TransitionVariableGroup` that holds both `R` and `D` handles
    the correct correlation structure between `R` and `D.`

    When an instance is initialized, its `self.get_joint_realization` attribute
    is dynamically assigned to a method according to its `self.transition_type`
    attribute. This enables all instances to use the same method during
    simulation.

    Attributes:
        origin (Compartment):
            specifies origin of `TransitionVariableGroup` --
            corresponding populations leave this compartment.
        _transition_type (str):
            only values defined in `JointTransitionTypes` are valid,
            specifies joint probability distribution of all outflows
            from origin.
        transition_variables (list[`TransitionVariable`]):
            specifying `TransitionVariable` instances that outflow from origin --
            order does not matter.
        get_joint_realization (function):
            assigned at initialization, generates realizations according
            to probability distribution given by `self._transition_type` attribute,
            returns either (M x |A| x |R|) or ((M+1) x |A| x |R|) np.ndarray,
            where M is the length of `self.transition_variables` (i.e., number of
            outflows from origin), |A| is the number of age groups, |R| is number of
            risk groups.
        current_vals_list (list):
            used to store results from `self.get_joint_realization` --
            has either M or M+1 arrays of size |A| x |R|.

    See `__init__` docstring for other attributes.
    """

    def __init__(self,
                 origin: Compartment,
                 transition_type: TransitionTypes,
                 transition_variables: list[TransitionVariable]):
        """
        Args:
            transition_type (str):
                only values defined in `TransitionTypes` are valid, specifying
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
            numpy array of positive floats with size equal to number
            of age groups x number of risk groups, and with value
            corresponding to sum of current rates of transition variables in
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
        (multinomial) transitions (`get_multinomial_realization` method).

        Returns:
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
        This function is dynamically assigned based on the
        `TransitionVariableGroup`'s `transition_type` -- this function is set to
        one of the following methods: `self.get_multinomial_realization`,
        `self.get_multinomial_taylor_approx_realization`,
        `self.get_poisson_realization`, `self.get_multinomial_deterministic_realization`,
        `self.get_multinomial_taylor_approx_deterministic_realization`,
        `self.get_poisson_deterministic_realization`.

        Parameters:
            RNG (np.random.Generator object):
                 used to generate stochastic transitions in the model and control
                 reproducibility.
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
                 used to generate stochastic transitions in the model and control
                 reproducibility.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                contains positive floats, size equal to
                ((length of outgoing transition variables list + 1)
                x number of age groups x number of risk groups) --
                note the "+1" corresponds to the multinomial outcome of staying
                in the same compartment (not transitioning to any outgoing
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
                 used to generate stochastic transitions in the model and control
                 reproducibility.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                size equal to (length of outgoing transition variables list + 1)
                x number of age groups x number of risk groups --
                note the "+1" corresponds to the multinomial outcome of staying
                in the same compartment (not transitioning to any outgoing
                compartment).
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
                 used to generate stochastic transitions in the model and control
                 reproducibility.
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
        Deterministic counterpart to `self.get_multinomial_realization` --
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
                in the same compartment (not transitioning to any outgoing
                compartment).
        """

        probabilities_array = self.get_probabilities_array(num_timesteps)
        return np.asarray(self.origin.current_val * probabilities_array, dtype=int)

    def get_multinomial_taylor_approx_deterministic_realization(self,
                                                                RNG: np.random.Generator,
                                                                num_timesteps: int) -> np.ndarray:
        """
        Deterministic counterpart to `self.get_multinomial_taylor_approx_realization` --
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
                in the same compartment (not transitioning to any outgoing
                compartment).
        """

        current_rates_array = self.get_current_rates_array()
        return np.asarray(self.origin.current_val * current_rates_array / num_timesteps, dtype=int)

    def get_poisson_deterministic_realization(self,
                                              RNG: np.random.Generator,
                                              num_timesteps: int) -> np.ndarray:
        """
        Deterministic counterpart to `self.get_poisson_realization` --
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

        return np.asarray(self.origin.current_val *
                          self.get_current_rates_array() / num_timesteps, dtype=int)

    def reset(self) -> None:
        self.current_vals_list = []

    def update_transition_variable_realizations(self) -> None:
        """
        Updates current_val attribute on all `TransitionVariable`
        instances contained in this `TransitionVariableGroup`.
        """

        # Since the ith element in probabilities_array corresponds to the ith transition variable
        #   in transition_variables, the ith element in multinomial_realizations_list
        #   also corresponds to the ith transition variable in transition_variables
        # Update the current realization of the transition variables contained in this group
        for ix in range(len(self.transition_variables)):
            self.transition_variables[ix].current_val = \
                self.current_vals_list[ix, :, :]


class EpiMetric(StateVariable, ABC):
    """
    Abstract base class for epi metrics in epidemiological model.

    This is intended for variables that are aggregate deterministic functions of
    the `SubpopState` (including `Compartment` `current_val`'s, other parameters,
    and time.)

    For example, population-level immunity variables should be
    modeled as a `EpiMetric` subclass, with a concrete
    implementation of the abstract method `self.get_change_in_current_val`.

    Inherits attributes from `StateVariable`.

    Attributes:
        current_val (np.ndarray):
            same size as init_val, holds current value of `StateVariable`
            for age-risk groups.
        change_in_current_val : (np.ndarray):
            initialized to None, but during simulation holds change in
            current value of `EpiMetric` for age-risk groups
            (size |A| x |R|, where |A| is the number of risk groups and |R| is number
            of age groups).

    See `__init__` docstring for other attributes.
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

    @abstractmethod
    def get_change_in_current_val(self,
                                  state: SubpopState,
                                  params: SubpopParams,
                                  num_timesteps: int) -> np.ndarray:
        """
        Computes and returns change in current value of dynamic val,
        based on current state of the simulation and epidemiological parameters.

        NOTE:
            OUTPUT SHOULD ALREADY BE SCALED BY NUM_TIMESTEPS.

        Output should be a numpy array of size |A| x |R|, where A
        is number of age groups and |R| is number of risk groups.

        Args:
            state (SubpopState):
                holds subpopulation simulation state (current values of
                `StateVariable` instances).
            params (SubpopParams):
                holds values of epidemiological parameters.
            num_timesteps (int):
                number of timesteps per day -- used to determine time interval
                length for discretization.

        Returns:
            np.ndarray:
                size |A| x |R|, where |A| is the number of age groups and
                |R| is number of risk groups.
        """
        pass

    def update_current_val(self) -> None:
        """
        Adds `self.change_in_current_val` attribute to
            `self.current_val` attribute in-place.
        """

        self.current_val += self.change_in_current_val


class DynamicVal(StateVariable, ABC):
    """
    Abstract base class for variables that dynamically adjust
    their values based the current values of other `StateVariable`
    instances.

    This class should model social distancing (and more broadly,
    staged-alert policies). For example, if we consider a
    case where transmission rates decrease when number infected
    increase above a certain level, we can create a subclass of
    DynamicVal that models a coefficient that modifies transmission
    rates, depending on the epi compartments corresponding to
    infected people.

    Inherits attributes from `StateVariable`.

    See `__init__` docstring for other attributes.
    """

    def __init__(self,
                 init_val: Optional[np.ndarray | float] = None,
                 is_enabled: Optional[bool] = False):
        """

        Args:
            init_val (Optional[np.ndarray | float]):
                starting value(s) at the beginning of the simulation.
            is_enabled (Optional[bool]):
                if `False`, this dynamic value does not get updated
                during the simulation and defaults to its `self.init_val`.
                This is designed to allow easy toggling of
                simulations with or without staged alert policies
                and other interventions.
        """

        super().__init__(init_val)
        self.is_enabled = is_enabled

    @abstractmethod
    def update_current_val(self,
                           state: SubpopState,
                           params: SubpopParams) -> None:
        """
        Args:
            state (SubpopState):
                holds subpopulation simulation state (current values of
                `StateVariable` instances).
            params (SubpopParams):
                holds values of epidemiological parameters.
        """


@dataclass
class Schedule(StateVariable, ABC):
    """
    Abstract base class for variables that are functions of real-world
    dates -- for example, contact matrices (which depend on the day of
    the week and whether the current day is a holiday), historical
    vaccination data, and seasonality.

    Inherits attributes from `StateVariable`.

    See `__init__` docstring for other attributes.
    """

    def __init__(self,
                 init_val: Optional[np.ndarray | float] = None,
                 timeseries_df: Optional[dict] = None):
        """
        Args:
            init_val (Optional[np.ndarray | float]):
                starting value(s) at the beginning of the simulation
            timeseries_df (Optional[pd.DataFrame] = None):
                has a "date" column with strings in format `"YYYY-MM-DD"`
                of consecutive calendar days, and other columns
                corresponding to values on those days
        """

        super().__init__(init_val)
        self.timeseries_df = timeseries_df

    @abstractmethod
    def update_current_val(self,
                           params: SubpopParams,
                           current_date: datetime.date) -> None:
        """
        Subpop classes must provide a concrete implementation of
        updating `self.current_val` in-place.

        Args:
            params (SubpopParams):
                fixed parameters of subpopulation model.
            current_date (date):
                real-world date corresponding to
                model's current simulation day.
        """
        pass


class InteractionTerm(StateVariable, ABC):
    """
    Abstract base class for variables that depend on the state of
    more than one `SubpopModel` (i.e., that depend on more than one
    `SubpopState`). These variables are functions of how subpopulations
    interact.

    In contrast to other state variables, each `InteractionTerm`
    takes in an `InterSubpopRepo` instance to update its `self.current_val`.
    Other state variables that are "local" and depend on
    exactly one subpopulation only need to take in one `SubpopState`
    and one `SubpopParams` instance to update its current value.

    Inherits attributes from `StateVariable`.

    See `__init__` docstring for other attributes.
    """

    @abstractmethod
    def update_current_val(self,
                           inter_subpop_repo: InterSubpopRepo,
                           subpop_params: SubpopParams) -> None:
        """
        Subclasses must provide a concrete implementation of
        updating `self.current_val` in-place.

        Args:
            inter_subpop_repo (InterSubpopRepo):
                manages collection of subpop models with
                methods for querying information.
            subpop_params (SubpopParams):
                holds values of subpopulation's epidemiological parameters.
        """

        pass


class MetapopModel(ABC):
    """
    Abstract base class that bundles `SubpopModel`s linked using
        a travel model.

    Params:
        inter_subpop_repo (InterSubpopRepo):
            Accesses and manages `SubpopState` instances
            of corresponding `SubpopModel`s, and provides
            methods to query current values.

    See `__init__` docstring for other attributes.
    """

    def __init__(self,
                 inter_subpop_repo,
                 name: str = ""):
        """
        Params:
            inter_subpop_repo (InterSubpopRepo):
                manages collection of subpopulation models with
                methods for querying information.
            name (str):
                unique identifier for `MetapopModel`.
        """

        self.subpop_models = inter_subpop_repo.subpop_models

        self.inter_subpop_repo = inter_subpop_repo

        self.name = name

        for model in self.subpop_models.values():
            model.metapop_model = self
            model.interaction_terms = model.create_interaction_terms()
            model.state.interaction_terms = model.interaction_terms

    def extract_states_dict_from_models_dict(self,
                                             models_dict: sc.objdict) -> sc.objdict:
        """
        (Currently unused utility function.)

        Takes objdict of subpop models, where keys are subpop model names and
            values are the subpop model instances, and returns objdict of
            subpop model states, where keys are subpop model names and
            values are the subpop model `SubpopState` instances.
        """

        states_dict = \
            sc.objdict({name: model.state for name, model in models_dict.items()})

        return states_dict

    def simulate_until_day(self,
                           simulation_end_day: int) -> None:
        """
        Advance simulation model time until `simulation_end_day` in
        `MetapopModel`.
        
        NOT just the same as looping through each `SubpopModel`'s
        `simulate_until_day` method. On the `MetapopModel`,
        because `SubpopModel` instances are linked with `InteractionTerm`s
        and are not independent of each other, this `MetapopModel`'s
        `simulate_until_day` method has additional functionality.

        Note: the update order at the beginning of each day is very important!

        - First, each `SubpopModel` updates its daily state (computing
            `Schedule` and `DynamicVal` instances).
        - Second, the `MetapopModel`'s `InterSubpopRepo` computes any shared
            terms used across subpopulations (to reduce computational overhead),
            and then updates each `SubpopModel`'s associated `InteractionTerm`
            instances.
        - Third, each `SubpopModel` simulates discretized timesteps (sampling
            `TransitionVariable`s, updating `EpiMetric`s, and updating `Compartment`s).

        Note: we only update the `InterSubpopRepo` shared quantities
            once a day, not at every timestep -- in other words,
            the travel model state-dependent values are only updated daily
            -- this is to avoid severe computation inefficiency

        Args:
            simulation_end_day (positive int):
                stop simulation at `simulation_end_day` (i.e. exclusive,
                simulate up to but not including `simulation_end_day`).
        """

        if self.current_simulation_day > simulation_end_day:
            raise MetapopModelError(f"Current day counter ({self.current_simulation_day}) "
                                    f"exceeds last simulation day ({simulation_end_day}).")

        # Adding this in case the user manually changes the initial
        #   value or current value of any state variable --
        #   otherwise, the state will not get updated
        # Analogous logic in SubpopModel's `simulate_until_day` method
        for subpop_model in self.subpop_models.values():
            subpop_model.state.sync_to_current_vals(subpop_model.all_state_variables)

        while self.current_simulation_day < simulation_end_day:

            for subpop_model in self.subpop_models.values():
                subpop_model.prepare_daily_state()

            self.inter_subpop_repo.compute_shared_quantities()
            self.inter_subpop_repo.update_all_interaction_terms()

            for subpop_model in self.subpop_models.values():

                save_daily_history = subpop_model.config.save_daily_history
                timesteps_per_day = subpop_model.config.timesteps_per_day

                subpop_model.simulate_timesteps(timesteps_per_day)

                if save_daily_history:
                    subpop_model.save_daily_history()

                subpop_model.increment_simulation_day()

    def display(self):
        """
        Prints structure (compartments and linkages), transition variables,
        epi metrics, schedules, and dynamic values for each `SubpopModel`
        instance in `self.subpop_models`.
        """
        for subpop_model in self.subpop_models.values():
            subpop_model.display()

    def reset_simulation(self):
        """
        Resets `MetapopModel` by resetting and clearing
            history on all `SubpopModel` instances in
            `self.subpop_models`.
        """

        for subpop_model in self.subpop_models.values():
            subpop_model.reset_simulation()

    @property
    def current_simulation_day(self) -> int:
        """
        Returns:
            Current simulation day. The current simulation day of the
            `MetapopModel` should be the same as each individual `SubpopModel`
            in the `MetapopModel`. Otherwise, an error is raised.
        """

        current_simulation_days_list = []

        for subpop_model in self.subpop_models.values():
            current_simulation_days_list.append(subpop_model.current_simulation_day)

        if len(set(current_simulation_days_list)) > 1:
            raise MetapopModelError("Subpopulation models are on different simulation days "
                                    "and are out-of-sync. This may be caused by simulating "
                                    "a subpopulation model independently from the "
                                    "metapopulation model. Fix error and try again.")
        else:
            return current_simulation_days_list[0]

    @property
    def current_real_date(self) -> datetime.date:
        """
        Returns:
            Current real date corresponding to current simulation day.
            The current real date of the `MetapopModel` should be the same as
            each individual `SubpopModel` in the `MetapopModel`.
            Otherwise, an error is raised.
        """

        current_real_dates_list = []

        for subpop_model in self.subpop_models.values():
            current_real_dates_list.append(subpop_model.current_real_date)

        if len(set(current_real_dates_list)) > 1:
            raise MetapopModelError("Subpopulation models are on different real dates "
                                    "and are out-of-sync. This may be caused by simulating "
                                    "a subpopulation model independently from the "
                                    "metapopulation model. Fix error and try again.")
        else:
            return current_real_dates_list[0]


class SubpopModel(ABC):
    """
    Contains and manages all necessary components for
    simulating a compartmental model for a given subpopulation.

    Each `SubpopModel` instance includes compartments,
    epi metrics, dynamic vals, a data container for the current simulation
    state, transition variables and transition variable groups,
    epidemiological parameters, simulation experiment configuration
    parameters, and a random number generator.

    All city-level subpopulation models, regardless of disease type and
    compartment/transition structure, are instances of this class.

    When creating an instance, the order of elements does not matter
    within `self.compartments`, `self.epi_metrics`, `self.dynamic_vals`,
    `self.transition_variables`, and `self.transition_variable_groups`.
    The "flow" and "physics" information are stored on the objects.

    Attributes:
        interaction_terms (sc.objdict[str, InteractionTerm]):
            objdict of all the subpop model's `InteractionTerm` instances.
        compartments (sc.objdict[str, Compartment]):
            objdict of all the subpop model's `Compartment` instances.
        transition_variables (sc.objdict[str, TransitionVariable]):
            objdict of all the subpop model's `TransitionVariable` instances.
        transition_variable_groups (sc.objdict[str, TransitionVariableGroup]):
            objdict of all the subpop model's `TransitionVariableGroup` instances.
        epi_metrics (sc.objdict[str, EpiMetric]):
            objdict of all the subpop model's `EpiMetric` instances.
        dynamic_vals (sc.objdict[str, DynamicVal]):
            objdict of all the subpop model's `DynamicVal` instances.
        schedules (sc.objdict[str, Schedule]):
            objdict of all the subpop model's `Schedule` instances.
        current_simulation_day (int):
            tracks current simulation day -- incremented by +1
            when `config.timesteps_per_day` discretized timesteps
            have completed.
        current_real_date (datetime.date):
            tracks real-world date -- advanced by +1 day when
            `config.timesteps_per_day` discretized timesteps
            have completed.

    See `__init__` docstring for other attributes.
    """

    def __init__(self,
                 state: SubpopState,
                 params: SubpopParams,
                 config: Config,
                 RNG: np.random.Generator,
                 name: str = "",
                 metapop_model: MetapopModel = None):

        """
        Params:
            state (SubpopState):
                holds current values of `SubpopModel`'s state variables.
            params (SubpopParams):
                data container for the model's epidemiological parameters,
                such as the "Greek letters" characterizing sojourn times
                in compartments.
            config (Config):
                data container for the model's simulation configuration values.
            RNG (np.random.Generator):
                 used to generate stochastic transitions in the model and control
                 reproducibility.
            name (str):
                unique identifier of `SubpopModel`.
            metapop_model (Optional[MetapopModel]):
                if not `None`, is the `MetapopModel` instance
                associated with this `SubpopModel`.
        """

        self.state = copy.deepcopy(state)
        self.params = copy.deepcopy(params)
        self.config = copy.deepcopy(config)

        self.RNG = RNG

        self.current_simulation_day = 0
        self.start_real_date = self.get_start_real_date()
        self.current_real_date = self.start_real_date

        self.metapop_model = None
        self.name = name

        self.schedules = self.create_schedules()
        self.interaction_terms = self.create_interaction_terms()
        self.compartments = self.create_compartments()
        self.transition_variables = self.create_transition_variables(self.compartments)
        self.transition_variable_groups = self.create_transition_variable_groups(self.compartments,
                                                                                 self.transition_variables)
        self.epi_metrics = self.create_epi_metrics(self.transition_variables)
        self.dynamic_vals = self.create_dynamic_vals()

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
            np.ndarray:
                |A| x |R| array, where |A| is the number of age groups
                and |R| is the number of risk groups, corresponding to
                total population for that age-risk group (summed
                over all compartments in the subpop model).
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
        Fetches `start_real_date` from `self.config` -- converts to
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
    def create_interaction_terms(self) -> sc.objdict[str, InteractionTerm]:
        pass

    @abstractmethod
    def create_compartments(self) -> sc.objdict[str, Compartment]:
        pass

    @abstractmethod
    def create_transition_variables(self,
                                    compartments_dict: sc.objdict[str, Compartment] = None) \
            -> sc.objdict[str, TransitionVariable]:
        pass

    @abstractmethod
    def create_transition_variable_groups(self,
                                          compartments_dict: sc.objdict[str, Compartment] = None,
                                          transition_variables_dict: sc.objdict[str, TransitionVariable] = None) \
            -> sc.objdict[str, TransitionVariableGroup]:
        pass

    @abstractmethod
    def create_epi_metrics(self,
                           transition_variables_dict: sc.objdict[str, TransitionVariable] = None)\
            -> sc.objdict[str, EpiMetric]:
        pass

    @abstractmethod
    def create_dynamic_vals(self) -> sc.objdict[str, DynamicVal]:
        pass

    @abstractmethod
    def create_schedules(self) -> sc.objdict[str, Schedule]:
        pass

    def modify_random_seed(self, new_seed_number) -> None:
        """
        Modifies model's `self.RNG` attribute in-place to new generator
        seeded at `new_seed_number`.

        Args:
            new_seed_number (int):
                used to re-seed model's random number generator.
        """

        self._bit_generator = np.random.MT19937(seed=new_seed_number)
        self.RNG = np.random.Generator(self._bit_generator)

    def simulate_until_day(self,
                           simulation_end_day: int) -> None:
        """
        Advance simulation model time until `simulation_end_day`.

        Advance time by iterating through simulation days,
        which are simulated by iterating through discretized
        timesteps.

        Save daily simulation data as history on each `Compartment`
        instance.

        Args:
            simulation_end_day (positive int):
                stop simulation at `simulation_end_day` (i.e. exclusive,
                simulate up to but not including `simulation_end_day`).
        """

        if self.current_simulation_day > simulation_end_day:
            raise SubpopModelError(f"Current day counter ({self.current_simulation_day}) "
                                   f"exceeds last simulation day ({simulation_end_day}).")

        save_daily_history = self.config.save_daily_history
        timesteps_per_day = self.config.timesteps_per_day

        # Adding this in case the user manually changes the initial
        #   value or current value of any state variable --
        #   otherwise, the state will not get updated
        self.state.sync_to_current_vals(self.all_state_variables)

        # simulation_end_day is exclusive endpoint
        while self.current_simulation_day < simulation_end_day:

            self.prepare_daily_state()

            self.simulate_timesteps(timesteps_per_day)

            if save_daily_history:
                self.save_daily_history()

            self.increment_simulation_day()

    def simulate_timesteps(self,
                           num_timesteps: int) -> None:
        """
        Subroutine for `self.simulate_until_day`.

        Iterates through discretized timesteps to simulate next
        simulation day. Granularity of discretization is given by
        attribute `self.config.timesteps_per_day`.

        Properly scales transition variable realizations and changes
        in dynamic vals by specified timesteps per day.

        Args:
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

        # Important note: this order of updating is important,
        #   because schedules do not depend on other state variables,
        #   but dynamic vals may depend on schedules
        # Interaction terms may depend on both schedules
        #   and dynamic vals (but interaction terms are updated by
        #   the InterSubpopRepo, not on individual SubpopModel
        #   instances).

        schedules = self.schedules
        dynamic_vals = self.dynamic_vals

        # Update schedules for current day
        for schedule in schedules.values():
            schedule.update_current_val(subpop_params,
                                        current_real_date)

        self.state.sync_to_current_vals(schedules)

        # Update dynamic values for current day
        for dval in dynamic_vals.values():
            if dval.is_enabled:
                dval.update_current_val(subpop_state, subpop_params)

        self.state.sync_to_current_vals(dynamic_vals)

    def update_epi_metrics(self) -> None:
        """
        Update current value attribute on each associated
            `EpiMetric` instance.
        """

        state = self.state
        params = self.params
        timesteps_per_day = self.config.timesteps_per_day

        for metric in self.epi_metrics.values():
            metric.change_in_current_val = \
                metric.get_change_in_current_val(state,
                                                 params,
                                                 timesteps_per_day)
            metric.update_current_val()

    def update_transition_rates(self) -> None:
        """
        Compute current transition rates for each transition variable,
            and store this updated value on each variable's
            current_rate attribute.
        """

        state = self.state
        params = self.params

        for tvar in self.transition_variables.values():
            tvar.current_rate = tvar.get_current_rate(state, params)

    def sample_transitions(self) -> None:
        """
        For each transition variable, sample a random realization
            using its current rate. Handle jointly distributed transition
            variables first (using `TransitionVariableGroup` logic), then
            handle marginally distributed transition variables.
            Use `SubpopModel`'s `RNG` to generate random variables.
        """

        RNG = self.RNG
        timesteps_per_day = self.config.timesteps_per_day
        save_transition_variables_history = self.config.save_transition_variables_history

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

        if save_transition_variables_history:
            for tvar in self.transition_variables.values():
                tvar.save_history()

    def update_compartments(self) -> None:
        """
        Update current value of each `Compartment`, by
            looping through all `TransitionVariable` instances
            and subtracting/adding their current values
            from origin/destination compartments respectively.
        """

        for tvar in self.transition_variables.values():
            tvar.update_origin_outflow()
            tvar.update_destination_inflow()

        for compartment in self.compartments.values():
            compartment.update_current_val()

            # After updating the compartment's current value,
            #   reset its inflow and outflow attributes, to
            #   prepare for the next iteration.
            compartment.reset_inflow()
            compartment.reset_outflow()

    def increment_simulation_day(self) -> None:
        """
        Move day counters to next simulation day, both
            for integer simulation day and real date.
        """

        self.current_simulation_day += 1
        self.current_real_date += datetime.timedelta(days=1)

    def save_daily_history(self) -> None:
        """
        Update history at end of each day, not at end of every
           discretization timestep, to be efficient.
        Update history of state variables other than `Schedule`
           instances -- schedules do not have history.
        """
        for svar in self.interaction_terms.values() + \
                    self.compartments.values() + \
                    self.epi_metrics.values() + \
                    self.dynamic_vals.values():
            svar.save_history()

    def reset_simulation(self) -> None:
        """
        Reset simulation in-place. Subsequent method calls of
        `self.simulate_until_day` start from day 0, with original
        day 0 state.

        Returns `self.current_simulation_day` to 0.
        Restores state values to initial values.
        Clears history on model's state variables.
        Resets transition variables' `current_val` attribute to 0.

        WARNING:
            DOES NOT RESET THE MODEL'S RANDOM NUMBER GENERATOR TO
            ITS INITIAL STARTING SEED. RANDOM NUMBER GENERATOR WILL CONTINUE
            WHERE IT LEFT OFF.

        Use method `self.modify_random_seed` to reset model's `RNG` to its
        initial starting seed.
        """

        self.current_simulation_day = 0
        self.current_real_date = self.start_real_date

        # AGAIN, MUST BE CAREFUL ABOUT MUTABLE NUMPY ARRAYS -- MUST USE DEEP COPY
        for svar in self.all_state_variables.values():
            setattr(svar, "current_val", copy.deepcopy(svar.init_val))

        self.state.sync_to_current_vals(self.all_state_variables)

        for svar in self.all_state_variables.values():
            svar.reset()

        for tvar in self.transition_variables.values():
            tvar.reset()

        for tvargroup in self.transition_variable_groups.values():
            tvargroup.current_vals_list = []

    def find_name_by_compartment(self,
                                 target_compartment: Compartment) -> str:
        """
        Given `Compartment`, returns name of that `Compartment`.

        Args:
            target_compartment (Compartment):
                Compartment object with a name to look up

        Returns:
            str:
                Compartment name, given by the key to look
                it up in the `SubpopModel`'s compartments objdict
        """

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
