import numpy as np
import sciris as sc

from dataclasses import dataclass
from typing import Optional

import clt_base as clt


@dataclass
class SIRSubpopParams(clt.SubpopParams):
    """
    Data container for pre-specified and fixed epidemiological
    parameters in SIR model.

    Each field of datatype np.ndarray must be A x L,
    where A is the number of age groups and L is the number of
    risk groups. Note: this means all arrays should be 2D.

    Attributes:
        num_age_groups (positive int):
            number of age groups.
        num_risk_groups (positive int):
            number of risk groups.
        total_pop_age_risk (np.ndarray of positive ints):
            total number in population, summed across all
            age-risk groups.
        beta (positive float): transmission rate.
        I_to_R_rate (positive float):
            rate at which people in I move to R.
    """

    num_age_groups: Optional[int] = None
    num_risk_groups: Optional[int] = None
    total_pop_age_risk: Optional[np.ndarray] = None
    beta: Optional[float] = None
    I_to_R_rate: Optional[float] = None


@dataclass
class SIRSubpopState(clt.SubpopState):
    """
    Data container for pre-specified and fixed set of
    Compartment initial values and EpiMetric initial values
    in SIR model.

    Each field below should be A x L np.ndarray, where
    A is the number of age groups and L is the number of risk groups.
    Note: this means all arrays should be 2D. Even if there is
    1 age group and 1 risk group (no group stratification),
    each array should be 1x1, which is two-dimensional.
    For example, np.array([[100]]) is correct --
    np.array([100]) is wrong.

    Attributes:
        S (np.ndarray of positive floats):
            susceptible compartment for age-risk groups --
            (holds current_val of Compartment "S").
        I (np.ndarray of positive floats):
            infected for age-risk groups
            (holds current_val of Compartment "I").
        R (np.ndarray of positive floats):
            recovered compartment for age-risk groups
            (holds current_val of Compartment "R").
    """

    S: Optional[np.ndarray] = None
    I: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None


class SusceptibleToInfected(clt.TransitionVariable):

    def get_current_rate(self,
                         state: SIRSubpopState,
                         params: SIRSubpopParams) -> np.ndarray:

        return state.I * params.beta / params.total_pop_age_risk


class InfectedToRecovered(clt.TransitionVariable):

    def get_current_rate(self,
                         state: SIRSubpopState,
                         params: SIRSubpopParams) -> np.ndarray:

        return params.I_to_R_rate


class SIRSubpopModel(clt.SubpopModel):

    def __init__(self,
                 state_dict: dict,
                 params_dict: dict,
                 config_dict: dict,
                 RNG: np.random.Generator,
                 name: str = "",
                 wastewater_enabled: bool = False):
        """
        Args:
            state_dict (dict):
                holds current simulation state information,
                such as current values of epidemiological compartments
                and epi metrics -- keys and values respectively
                must match field names and format of FluSubpopState.
            params_dict (dict):
                holds epidemiological parameter values -- keys and
                values respectively must match field names and
                format of FluSubpopParams.
            config_dict (dict):
                holds configuration values -- keys and values
                respectively must match field names and format of
                Config.
            RNG (np.random.Generator):
                numpy random generator object used to obtain
                random numbers.
            name (str):
                name.
            wastewater_enabled (bool):
                if True, includes "wastewater" EpiMetric. Otherwise,
                excludes it.
        """

        # Assign config, params, and state to model-specific
        # types of dataclasses that have epidemiological parameter information
        # and sim state information

        self.wastewater_enabled = wastewater_enabled

        state = clt.make_dataclass_from_dict(SIRSubpopState, state_dict)
        params = clt.make_dataclass_from_dict(SIRSubpopParams, params_dict)
        config = clt.make_dataclass_from_dict(clt.Config, config_dict)

        # IMPORTANT NOTE: as always, we must be careful with mutable objects
        #   and generally use deep copies to avoid modification of the same
        #   object. But in this function call, using deep copies is unnecessary
        #   (redundant) because the parent class SubpopModel's __init__()
        #   creates deep copies.
        super().__init__(state, params, config, RNG, name)

    def create_interaction_terms(self) -> sc.objdict:

        return sc.objdict()

    def create_compartments(self) -> sc.objdict:

        compartments = sc.objdict()

        for name in ("S", "I", "R"):
            compartments[name] = clt.Compartment(getattr(self.state, name))

        return compartments

    def create_dynamic_vals(self) -> sc.objdict:

        dynamic_vals = sc.objdict()

        return dynamic_vals

    def create_schedules(self) -> sc.objdict():

        schedules = sc.objdict()

        return schedules

    def create_transition_variables(self) -> sc.objdict:

        transition_type = self.config.transition_type
        compartments = self.compartments

        transition_variables = sc.objdict()

        S = compartments.S
        I = compartments.I
        R = compartments.R

        transition_variables.S_to_I = SusceptibleToInfected(S, I, transition_type)
        transition_variables.I_to_R = InfectedToRecovered(I, R, transition_type)

        return transition_variables

    def create_transition_variable_groups(self) -> sc.objdict:

        transition_variable_groups = sc.objdict()

        return transition_variable_groups

    def create_epi_metrics(self) -> sc.objdict:

        epi_metrics = sc.objdict()

        return epi_metrics




