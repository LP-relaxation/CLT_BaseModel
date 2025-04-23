###########################################################
######################## SIR-M Model ######################
###########################################################

# The S-I-R model we demonstrate has the following structure:
#   S -> I -> R -> S
# with population-level immunity EpiMetric M

import numpy as np
import sciris as sc
from pathlib import Path

from dataclasses import dataclass
from typing import Optional

import clt_base as clt

# The math for transitions is as follows:
#   - S to I transition rate: I * beta / (total_pop * (1 + risk_reduction * M))
#   - I to R transition rate: I_to_R_rate
#   - R to S transition rate: R_to_S_rate

# The update rule for immunity is
#   - dM/dt = (immune_gain * R_to_S_rate * R) /
#               (total_pop * (1 + immune_saturation * M)) - immune_wane * M

@dataclass
class ToyImmunitySubpopParams(clt.SubpopParams):

    num_age_groups: int = 1,
    num_risk_groups: int = 1,
    total_pop: Optional[int] = None
    beta: Optional[float] = None
    I_to_R_rate: Optional[float] = None
    R_to_S_rate: Optional[float] = None
    immune_gain: Optional[float] = None
    immune_wane: Optional[float] = None
    immune_saturation: Optional[float] = None
    risk_reduction: Optional[float] = None


@dataclass
class ToyImmunitySubpopState(clt.SubpopState):

    S: Optional[int] = None
    I: Optional[int] = None
    R: Optional[int] = None
    M: Optional[float] = None


class SusceptibleToInfected(clt.TransitionVariable):

    def get_current_rate(self,
                         state: ToyImmunitySubpopState,
                         params: ToyImmunitySubpopParams) -> np.ndarray:

        return state.I * params.beta / (params.total_pop *
                                        (1 + params.risk_reduction * state.M))


class InfectedToRecovered(clt.TransitionVariable):

    def get_current_rate(self,
                         state: ToyImmunitySubpopState,
                         params: ToyImmunitySubpopParams) -> np.ndarray:
        return params.I_to_R_rate


class RecoveredToSusceptible(clt.TransitionVariable):

    def get_current_rate(self,
                         state: ToyImmunitySubpopState,
                         params: ToyImmunitySubpopParams) -> np.ndarray:
        return params.R_to_S_rate


class Immunity(clt.EpiMetric):

    def __init__(self, init_val, R_to_S):
        super().__init__(init_val)
        self.R_to_S = R_to_S

    def get_change_in_current_val(self,
                                  state: ToyImmunitySubpopState,
                                  params: ToyImmunitySubpopParams,
                                  num_timesteps: int) -> np.ndarray:
        
        return params.immune_gain * self.R_to_S.current_val / \
               (params.total_pop * (1 + params.immune_saturation * state.M)) - \
               params.immune_wane * state.M / num_timesteps


class ToyImmunitySubpopModel(clt.SubpopModel):

    def __init__(self,
                 compartments_epi_metrics_dict: dict,
                 params_dict: dict,
                 config_dict: dict,
                 RNG: np.random.Generator,
                 name: str = "",
                 wastewater_enabled: bool = False):
        """
        Args:
            compartments_epi_metrics_dict (dict):
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

        state = clt.make_dataclass_from_dict(ToyImmunitySubpopState, compartments_epi_metrics_dict)
        params = clt.make_dataclass_from_dict(ToyImmunitySubpopParams, params_dict)
        config = clt.make_dataclass_from_dict(clt.Config, config_dict)

        # IMPORTANT NOTE: as always, we must be careful with mutable objects
        #   and generally use deep copies to avoid modification of the same
        #   object. But in this function call, using deep copies is unnecessary
        #   (redundant) because the parent class SubpopModel's __init__()
        #   creates deep copies.
        super().__init__(state, params, config, RNG, name)

    def create_interaction_terms(self) -> sc.objdict[str, clt.InteractionTerm]:

        return sc.objdict()

    def create_dynamic_vals(self) -> sc.objdict[str, clt.DynamicVal]:

        dynamic_vals = sc.objdict()

        return dynamic_vals

    def create_schedules(self) -> sc.objdict[str, clt.Schedule]:

        schedules = sc.objdict()

        return schedules

    def create_epi_metrics(self,
                           transition_variables: sc.objdict[str, clt.TransitionVariable]) \
            -> sc.objdict[str, clt.EpiMetric]:

        epi_metrics = sc.objdict()

        epi_metrics["M"] = Immunity(self.state.M, transition_variables.R_to_S)

        return epi_metrics

    def create_compartments(self) -> sc.objdict[str, clt.Compartment]:

        compartments = sc.objdict()

        for name in ("S", "I", "R"):
            compartments[name] = clt.Compartment(getattr(self.state, name))

        return compartments

    def create_transition_variables(self,
            compartments: sc.objdict[str, clt.Compartment] = None) -> sc.objdict[str, clt.TransitionVariable]:

        type = self.config.transition_type

        transition_variables = sc.objdict()

        S = compartments.S
        I = compartments.I
        R = compartments.R

        transition_variables.S_to_I = SusceptibleToInfected(origin=S, destination=I, transition_type=type)
        transition_variables.I_to_R = InfectedToRecovered(origin=I, destination=R, transition_type=type)
        transition_variables.R_to_S = RecoveredToSusceptible(origin=R, destination=S, transition_type=type)

        return transition_variables

    def create_transition_variable_groups(
            self,
            compartments: sc.objdict[str, clt.Compartment] = None,
            transition_variables: sc.objdict[str, clt.TransitionVariable] = None)\
            -> sc.objdict[str, clt.TransitionVariableGroup]:

        return sc.objdict()
