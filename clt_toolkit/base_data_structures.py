from abc import ABC
from dataclasses import dataclass
from enum import Enum

import datetime


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


@dataclass(frozen=True)
class SimulationSettings:
    """
    Stores simulation settings.

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
            actual date that aligns with the beginning of the simulation
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


@dataclass(frozen=True)
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
