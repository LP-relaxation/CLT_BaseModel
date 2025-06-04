###############################################################
######################## SIHR-M-Mv Model ######################
###############################################################

# This code has Remy's humidity (seasonal forcing) functionality

# ToyImmunitySubpopModel has nonlinear saturation in dM/dt
# LinearSaturationSubpopModel is the same except it has linear
#   saturation in dM/dt (Anass's new proposal)

# See below for the precise write-up

# The SIHR-M-Mv model we demonstrate has the following structure:
#   S -> I -> H -> R -> S
# with population-level immunity EpiMetric M and Mv

# TODO: add vaccination time series as a Schedule --
#   for now it is constant, given in params

import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import sciris as sc
from pathlib import Path

from dataclasses import dataclass
from typing import Optional

import clt_base as clt

# The math for transitions is as follows:
#   - S to I transition rate: I * beta / (total_pop * (1 + inf_induced_inf_risk_constant * M +
#                                                           vax_induced_inf_risk_constant * M_v))
#   - I to H transition rate: I_to_H_rate * I_to_H_adjusted_prop /
#                               (inf_induced_hosp_risk_constant * M + vax_induced_hosp_risk_constant * M_v))
#   - I to R transition rate: I_to_R_rate * (1 - I_to_H_adjusted_prop)
#   - H to R transition rate: H_to_R_rate
#   - R to S transition rate: R_to_S_rate

# The update rule for immunity is
#   - dM/dt = R_to_S_rate * R / (N * (1 + inf_induced_saturation * M + vax_induced_saturation * M_v))
#               - inf_induced_immune_wane * M
#   - dMv/dt = (new vaccinations at time t - delta)/ N - vax_induced_immune_wane

# The update rule for linear saturation immunity is
#   - dM/dt = (R_to_S_rate * R / N) * (1 - inf_induced_saturation * M - vax_induced_saturation * M_v)

@dataclass
class ToyImmunitySubpopParams(clt.SubpopParams):

    num_age_groups: int = 1,
    num_risk_groups: int = 1,
    total_pop: Optional[int] = None
    beta: Optional[float] = None
    humidity_impact: Optional[float] = None
    I_to_H_rate: Optional[float] = None
    I_to_R_rate: Optional[float] = None
    H_to_R_rate: Optional[float] = None
    R_to_S_rate: Optional[float] = None
    I_to_H_adjusted_prop: Optional[float] = None
    inf_induced_saturation: Optional[float] = None
    inf_induced_immune_wane: Optional[float] = None
    vax_induced_saturation: Optional[float] = None
    vax_induced_immune_wane: Optional[float] = None
    inf_induced_inf_risk_constant: Optional[float] = None
    inf_induced_hosp_risk_constant: Optional[float] = None
    vax_induced_inf_risk_constant: Optional[float] = None
    vax_induced_hosp_risk_constant: Optional[float] = None
    vaccines_per_day: Optional[float] = None


@dataclass
class ToyImmunitySubpopState(clt.SubpopState):

    S: Optional[int] = None
    I: Optional[int] = None
    H: Optional[int] = None
    R: Optional[int] = None
    M: Optional[float] = None
    Mv: Optional[float] = None
    absolute_humidity: Optional[float] = None


class SusceptibleToInfected(clt.TransitionVariable):

    def get_current_rate(self,
                         state: ToyImmunitySubpopState,
                         params: ToyImmunitySubpopParams) -> np.ndarray:

        beta_adjusted = params.beta * (1 + params.humidity_impact * np.exp(-180 * state.absolute_humidity))

        return state.I * beta_adjusted / (params.total_pop *
                                        (1 + params.inf_induced_inf_risk_constant * state.M +
                                         params.vax_induced_inf_risk_constant * state.Mv))


class InfectedToHospitalized(clt.TransitionVariable):

    def get_current_rate(self,
                         state: ToyImmunitySubpopState,
                         params: ToyImmunitySubpopParams) -> np.ndarray:
        return params.I_to_H_rate * params.I_to_H_adjusted_prop / \
               (1 + params.inf_induced_hosp_risk_constant * state.M +
                params.vax_induced_hosp_risk_constant * state.Mv)


class InfectedToRecovered(clt.TransitionVariable):

    def get_current_rate(self,
                         state: ToyImmunitySubpopState,
                         params: ToyImmunitySubpopParams) -> np.ndarray:

        return params.I_to_R_rate * (1 - params.I_to_H_adjusted_prop)


class HospitalizedToRecovered(clt.TransitionVariable):

    def get_current_rate(self,
                         state: ToyImmunitySubpopState,
                         params: ToyImmunitySubpopParams) -> np.ndarray:

        return params.H_to_R_rate * params.I_to_H_adjusted_prop


class RecoveredToSusceptible(clt.TransitionVariable):

    def get_current_rate(self,
                         state: ToyImmunitySubpopState,
                         params: ToyImmunitySubpopParams) -> np.ndarray:
        return params.R_to_S_rate


class InfInducedImmunity(clt.EpiMetric):

    def __init__(self, init_val, R_to_S):
        super().__init__(init_val)
        self.R_to_S = R_to_S

    def get_change_in_current_val(self,
                                  state: ToyImmunitySubpopState,
                                  params: ToyImmunitySubpopParams,
                                  num_timesteps: int) -> np.ndarray:
        
        return self.R_to_S.current_val / (params.total_pop *
                                          (1 + params.inf_induced_saturation * state.M +
                                           params.vax_induced_saturation * state.Mv)) - \
               params.inf_induced_immune_wane * state.M / num_timesteps


class VaxInducedImmunity(clt.EpiMetric):

    def get_change_in_current_val(self,
                                  state: ToyImmunitySubpopState,
                                  params: ToyImmunitySubpopParams,
                                  num_timesteps: int) -> np.ndarray:

        return params.vaccines_per_day / (params.total_pop * num_timesteps) - \
               params.vax_induced_immune_wane * state.Mv / num_timesteps


class InfInducedImmunityLinearSaturation(clt.EpiMetric):

    def __init__(self, init_val, R_to_S):
        super().__init__(init_val)
        self.R_to_S = R_to_S

    def get_change_in_current_val(self,
                                  state: ToyImmunitySubpopState,
                                  params: ToyImmunitySubpopParams,
                                  num_timesteps: int) -> np.ndarray:

        return (self.R_to_S.current_val / params.total_pop) * \
               (1 - params.inf_induced_saturation * state.M - params.vax_induced_saturation * state.Mv) - \
               params.inf_induced_immune_wane * state.M / num_timesteps


class VaxInducedImmunityLinearSaturation(clt.EpiMetric):

    def get_change_in_current_val(self,
                                  state: ToyImmunitySubpopState,
                                  params: ToyImmunitySubpopParams,
                                  num_timesteps: int) -> np.ndarray:
        return params.vaccines_per_day / (params.total_pop * num_timesteps) - \
               params.vax_induced_immune_wane * state.Mv / num_timesteps


class AbsoluteHumidity(clt.Schedule):

    def __init__(self,
                 init_val: Optional[np.ndarray | float] = None,
                 filepath: Optional[str] = None):
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

        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"], format='%m/%d/%y').dt.date
        self.time_series_df = df

    def update_current_val(self, params, current_date: datetime.date) -> None:
        self.current_val = self.time_series_df.loc[
            self.time_series_df["date"] == current_date, "humidity"].values[0]


class ToyImmunitySubpopModel(clt.SubpopModel):

    def __init__(self,
                 compartments_epi_metrics_dict: dict,
                 params_dict: dict,
                 config_dict: dict,
                 RNG: np.random.Generator,
                 absolute_humidity_filepath: str,
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
            absolute_humidity_filepath (str):
                filepath (ending in ".csv") corresponding to
                absolute humidity data -- see `AbsoluteHumidity`
                class for CSV file specifications
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

        self.absolute_humidity_filepath = absolute_humidity_filepath

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
        schedules["absolute_humidity"] = AbsoluteHumidity(filepath=self.absolute_humidity_filepath)

        return schedules

    def create_epi_metrics(self,
                           transition_variables: sc.objdict[str, clt.TransitionVariable]) \
            -> sc.objdict[str, clt.EpiMetric]:

        epi_metrics = sc.objdict()

        epi_metrics["M"] = InfInducedImmunity(self.state.M, transition_variables.R_to_S)
        epi_metrics["Mv"] = VaxInducedImmunity(self.state.Mv)

        return epi_metrics

    def create_compartments(self) -> sc.objdict[str, clt.Compartment]:

        compartments = sc.objdict()

        for name in ("S", "I", "H", "R"):
            compartments[name] = clt.Compartment(getattr(self.state, name))

        return compartments

    def create_transition_variables(self,
            compartments: sc.objdict[str, clt.Compartment] = None) -> sc.objdict[str, clt.TransitionVariable]:

        type = self.config.transition_type

        transition_variables = sc.objdict()

        S = compartments.S
        I = compartments.I
        H = compartments.H
        R = compartments.R

        transition_variables.S_to_I = SusceptibleToInfected(origin=S, destination=I, transition_type=type)
        transition_variables.I_to_R = InfectedToRecovered(origin=I, destination=R, transition_type=type)
        transition_variables.I_to_H = InfectedToHospitalized(origin=I, destination=H, transition_type=type)
        transition_variables.H_to_R = HospitalizedToRecovered(origin=H, destination=R, transition_type=type)
        transition_variables.R_to_S = RecoveredToSusceptible(origin=R, destination=S, transition_type=type)

        return transition_variables

    def create_transition_variable_groups(
            self,
            compartments: sc.objdict[str, clt.Compartment] = None,
            transition_variables: sc.objdict[str, clt.TransitionVariable] = None)\
            -> sc.objdict[str, clt.TransitionVariableGroup]:

        return sc.objdict()


class LinearSaturationSubpopModel(ToyImmunitySubpopModel):

    def create_epi_metrics(self,
                           transition_variables: sc.objdict[str, clt.TransitionVariable]) \
            -> sc.objdict[str, clt.EpiMetric]:

        epi_metrics = sc.objdict()

        epi_metrics["M"] = InfInducedImmunityLinearSaturation(self.state.M, transition_variables.R_to_S)
        epi_metrics["Mv"] = VaxInducedImmunityLinearSaturation(self.state.Mv)

        return epi_metrics