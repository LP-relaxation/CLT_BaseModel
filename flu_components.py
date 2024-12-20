import datetime
import pandas as pd

import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
import base_components as base

import sciris as sc

base_path = Path(__file__).parent / "flu_demo_input_files"


# Note: for dataclasses, Optional is used to help with static type checking
# -- it means that an attribute can either hold a value with the specified
# datatype or it can be None


@dataclass
class FluFixedParams(base.FixedParams):
    """
    Data container for pre-specified and fixed epidemiological
    parameters in FluModel flu model. Along with FluSimState,
    is passed to get_current_rate and get_change_in_current_val.

    Assume that FluFixedParams fields are constant or piecewise
    constant throughout the simulation. For variables that
    are more complicated and time-dependent, use a EpiMetric
    instead.

    Each field of datatype np.ndarray must be A x L,
    where A is the number of age groups and L is the number of
    risk groups. Note: this means all arrays should be 2D.
    See FluSimState docstring for important formatting note
    on 2D arrays.

    TODO:
        when adding multiple strains, need to add subscripts
        to math of attributes and add strain-specific description

    Attributes:
        num_age_groups (positive int):
            number of age groups.
        num_risk_groups (positive int):
            number of risk groups.
        beta_baseline (positive float): transmission rate.
        total_population_val (np.ndarray of positive ints):
            total number in population, summed across all
            age-risk groups.
        humidity_impact (positive float):
            coefficient that determines how much absolute
            humidity affects beta_baseline.
        immunity_hosp_increase_factor (positive float):
            factor by which population-level immunity
            against hospitalization grows after each
            case that recovers.
        immunity_inf_increase_factor (positive float):
            factor by which population-level immunity
            against infection grows after each case
                that recovers.
        immunity_saturation (np.ndarray of positive floats):
            constant(s) modeling saturation of antibody
            production of individuals.
        waning_factor_hosp (positive float):
            rate at which infection-induced immunity
            against hospitalization wanes.
        waning_factor_inf (positive float):
            rate at which infection-induced immunity
            against infection wanes.
        hosp_risk_reduction (positive float in [0,1]):
            reduction in hospitalization risk from
            infection-induced immunity.
        inf_risk_reduction (positive float in [0,1]):
            reduction in infection risk
            from infection-induced immunity.
        death_risk_reduction (positive float in [0,1]):
            reduction in death risk from infection-induced immunity.
        R_to_S_rate (positive float):
            rate at which people in R move to S.
        E_to_I_rate (positive float):
            rate at which people in E move to I (both
            IP and IA, infected pre-symptomatic and infected
            asymptomatic)
        IP_to_IS_rate (positive float):
            rate a which people in IP (infected pre-symptomatic)
            move to IS (infected symptomatic)
        IS_to_R_rate (positive float):
            rate at which people in IS (infected symptomatic)
            move to R.
        IA_to_R_rate (positive float):
            rate at which people in IA (infected asymptomatic)
            move to R
        IS_to_H_rate (positive float):
            rate at which people in IS (infected symptomatic)
            move to H.
        H_to_R_rate (positive float):
            rate at which people in H move to R.
        H_to_D_rate (positive float):
            rate at which people in H move to D.
        E_to_IA_prop (np.ndarray of positive floats in [0,1]):
            proportion exposed who are asymptomatic based on
            age-risk groups.
        IS_to_H_adjusted_prop (np.ndarray of positive floats in [0,1]):
            rate-adjusted proportion infected who are hospitalized
            based on age-risk groups.
        H_to_D_adjusted_prop (np.ndarray of positive floats in [0,1]):
            rate-adjusted proportion hospitalized who die based on
            age-risk groups.
        IP_relative_inf (positive float):
            relative infectiousness of pre-symptomatic to symptomatic
            people (IP to IS compartment).
        IA_relative_inf (positive float):
            relative infectiousness of asymptomatic to symptomatic
            people (IA to IS compartment).
    """

    num_age_groups: Optional[int] = None
    num_risk_groups: Optional[int] = None
    beta_baseline: Optional[float] = None
    total_population_val: Optional[np.ndarray] = None
    humidity_impact: Optional[float] = None
    immunity_hosp_increase_factor: Optional[float] = None
    immunity_inf_increase_factor: Optional[float] = None
    immunity_saturation: Optional[np.ndarray] = None
    waning_factor_hosp: Optional[float] = None
    waning_factor_inf: Optional[float] = None
    hosp_risk_reduction: Optional[float] = None
    inf_risk_reduction: Optional[float] = None
    death_risk_reduction: Optional[float] = None
    R_to_S_rate: Optional[float] = None
    E_to_I_rate: Optional[float] = None
    IP_to_IS_rate: Optional[float] = None
    IS_to_R_rate: Optional[float] = None
    IA_to_R_rate: Optional[float] = None
    IS_to_H_rate: Optional[float] = None
    H_to_R_rate: Optional[float] = None
    H_to_D_rate: Optional[float] = None
    E_to_IA_prop: Optional[np.ndarray] = None
    IS_to_H_adjusted_prop: Optional[np.ndarray] = None
    H_to_D_adjusted_prop: Optional[np.ndarray] = None
    IP_relative_inf: Optional[float] = None
    IA_relative_inf: Optional[float] = None


@dataclass
class FluSimState(base.SimState):
    """
    Data container for pre-specified and fixed set of
    EpiCompartment initial values and EpiMetric initial values
    in FluModel flu model.

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
            (holds current_val of EpiCompartment "S").
        E (np.ndarray of positive floats):
            exposed compartment for age-risk groups --
            (holds current_val of EpiCompartment "E").
        IP (np.ndarray of positive floats):
            infected pre-symptomatic compartment for age-risk groups
            (holds current_val of EpiCompartment "IP").
        IS (np.ndarray of positive floats):
            infected symptomatic compartment for age-risk groups
            (holds current_val of EpiCompartment "IS").
        IA (np.ndarray of positive floats):
            infected asymptomatic compartment for age-risk groups
            (holds current_val of EpiCompartment "IA").
        H (np.ndarray of positive floats):
            hospital compartment for age-risk groups
            (holds current_val of EpiCompartment "H").
        R (np.ndarray of positive floats):
            recovered compartment for age-risk groups
            (holds current_val of EpiCompartment "R").
        D (np.ndarray of positive floats):
            dead compartment for age-risk groups
            (holds current_val of EpiCompartment "D").
        pop_immunity_hosp (np.ndarray of positive floats):
            infection-induced population-level immunity against
            hospitalization, for age-risk groups (holds current_val
            of EpiMetric "pop_immunity_hosp").
        pop_immunity_inf (np.ndarray of positive floats):
            infection-induced population-level immunity against
            infection, for age-risk groups (holds current_val
            of EpiMetric "pop_immunity_inf").
        absolute_humidity (positive float):
            grams of water vapor per cubic meter g/m^3,
            used as seasonality parameter that influences
            transmission rate beta_baseline.
        flu_contact_matrix (np.ndarray of positive floats):
            A x L x A x L array, where A is the number of age
            groups and L is the number of risk groups --
            element (a, l, a', l') corresponds to the number of
            contacts that a person in age-risk group a,l
            has with people in age-risk group a', l'.
        beta_reduct (float in [0,1]):
            starting value of DynamicVal "beta_reduct" on
            starting day of simulation -- this DynamicVal
            emulates a simple staged-alert policy
    """

    S: Optional[np.ndarray] = None
    E: Optional[np.ndarray] = None
    IP: Optional[np.ndarray] = None
    IS: Optional[np.ndarray] = None
    IA: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    D: Optional[np.ndarray] = None
    pop_immunity_hosp: Optional[np.ndarray] = None
    pop_immunity_inf: Optional[np.ndarray] = None
    absolute_humidity: Optional[float] = None
    flu_contact_matrix: Optional[np.ndarray] = None
    beta_reduct: Optional[float] = 0.0


class SusceptibleToExposed(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        force_of_immunity = (1 + fixed_params.inf_risk_reduction * sim_state.pop_immunity_inf)

        # We subtract absolute_humidity because higher humidity means less transmission
        beta_humidity_adjusted = (1 - sim_state.absolute_humidity * fixed_params.humidity_impact) * \
                                 fixed_params.beta_baseline

        # Compute I / N -> original shape is (A, L)
        # Expand ratio for broadcasting -> new shape is (1, 1, A, L)
        I_N_ratio_expanded = ((
                                      sim_state.IS + sim_state.IP * fixed_params.IP_relative_inf + sim_state.IA * fixed_params.IA_relative_inf)
                              / fixed_params.total_population_val)[None, None, :, :]

        # Expand force_of_immunity for broadcasting -> new shape is (A, L, 1, 1)
        force_of_immunity_expanded = force_of_immunity[:, :, None, None]

        # Element-wise multiplication and division by M_expanded
        # Sum over a' and l' (last two dimensions) -> result has shape (A, L)
        summand = np.sum(sim_state.flu_contact_matrix * I_N_ratio_expanded / force_of_immunity_expanded, axis=(2, 3))

        return (1 - sim_state.beta_reduct) * beta_humidity_adjusted * summand


class RecoveredToSusceptible(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups), fixed_params.R_to_S_rate)


class ExposedToAsymp(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       fixed_params.E_to_I_rate * fixed_params.E_to_IA_prop)


class ExposedToPresymp(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       fixed_params.E_to_I_rate * (1 - fixed_params.E_to_IA_prop))


class PresympToSymp(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       fixed_params.IP_to_IS_rate)


class SympToRecovered(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       (1 - fixed_params.IS_to_H_adjusted_prop) * fixed_params.IS_to_R_rate)


class AsympToRecovered(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       fixed_params.IA_to_R_rate)


class HospToRecovered(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.full((fixed_params.num_age_groups, fixed_params.num_risk_groups),
                       (1 - fixed_params.H_to_D_adjusted_prop) * fixed_params.H_to_R_rate)


class SympToHosp(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.asarray(fixed_params.IS_to_H_rate * fixed_params.IS_to_H_adjusted_prop /
                          (1 + fixed_params.hosp_risk_reduction * sim_state.pop_immunity_hosp))


class HospToDead(base.TransitionVariable):
    def get_current_rate(self,
                         sim_state: FluSimState,
                         fixed_params: FluFixedParams):
        return np.asarray(fixed_params.H_to_D_adjusted_prop * fixed_params.H_to_D_rate /
                          (1 + fixed_params.death_risk_reduction * sim_state.pop_immunity_hosp))


class PopulationImmunityHosp(base.EpiMetric):

    def __init__(self, init_val, R_to_S):
        super().__init__(init_val)
        self.R_to_S = R_to_S

    def get_change_in_current_val(self,
                                  sim_state: FluSimState,
                                  fixed_params: FluFixedParams,
                                  num_timesteps: int):
        # Note: I'm not actually sure all these precision
        #   precautions are necessary... I initially added this
        #   because I thought some floating point errors were
        #   responsible for a bug (the problem actually came
        #   from a different source).

        # Ensure consistent float64 precision
        factor = np.float64(fixed_params.immunity_hosp_increase_factor)
        susceptible = np.float64(self.R_to_S.current_val)
        population = np.float64(fixed_params.total_population_val)
        saturation = np.float64(fixed_params.immunity_saturation)
        pop_immunity = np.float64(sim_state.pop_immunity_hosp)
        waning_factor = np.float64(fixed_params.waning_factor_hosp)
        num_timesteps = np.float64(num_timesteps)

        # Break down calculations
        gain_numerator = factor * susceptible
        gain_denominator = population * (1 + saturation * pop_immunity)
        immunity_gain = gain_numerator / gain_denominator

        immunity_loss = waning_factor * pop_immunity

        # Final result
        result = (immunity_gain - immunity_loss) / num_timesteps

        return np.asarray(result, dtype=np.float64)


class PopulationImmunityInf(base.EpiMetric):
    def __init__(self, init_val, R_to_S):
        super().__init__(init_val)
        self.R_to_S = R_to_S

    def get_change_in_current_val(self,
                                  sim_state: FluSimState,
                                  fixed_params: FluFixedParams,
                                  num_timesteps: int):
        # Convert all parameters to consistent float64 for high precision
        increase_factor = np.float64(fixed_params.immunity_inf_increase_factor)
        R_to_S = np.float64(self.R_to_S.current_val)
        total_population = np.float64(fixed_params.total_population_val)
        saturation = np.float64(fixed_params.immunity_saturation)
        population_immunity = np.float64(sim_state.pop_immunity_inf)
        waning_factor = np.float64(fixed_params.waning_factor_inf)
        num_timesteps = np.float64(num_timesteps)

        # Break down calculations for better readability and to avoid compounded rounding errors
        gain_numerator = increase_factor * R_to_S
        gain_denominator = total_population * (1 + saturation * population_immunity)
        immunity_gain = gain_numerator / gain_denominator

        immunity_loss = waning_factor * population_immunity

        # Compute result with full precision
        result = (immunity_gain - immunity_loss) / num_timesteps

        # Ensure the result is a NumPy array
        return np.asarray(result, dtype=np.float64)


class BetaReduct(base.DynamicVal):

    def __init__(self, init_val, is_enabled):
        super().__init__(init_val, is_enabled)
        self.permanent_lockdown = False

    def update_current_val(self, sim_state, fixed_params):
        if np.sum(sim_state.I) / np.sum(fixed_params.total_population_val) > 0.05:
            self.current_val = .5
            self.permanent_lockdown = True
        else:
            if not self.permanent_lockdown:
                self.current_val = 0.0


def absolute_humidity_func(current_date: datetime.date) -> float:
    """
    Note: this is a dummy function loosely based off of
    the absolute humidity data from Kaiming and Shraddha's
    new burden averted draft.

    TODO: replace this function with real humidity function

    The following calculation is used to achieve the correct
        upside-down parabola with the right min and max
        values and location
        max_value = 12.5
        0.00027 = (max_value - k) / ((0 - h) ** 2)

    Args:
        current_date (datetime.date):
            datetime.date object corresponding to
            real-world date

    Returns:
        float:
            nonnegative float between 3.4 and 12.5
            corresponding to absolute humidity
            that day of the year
    """

    # Convert datetime.date to integer between 1 and 365
    #   corresponding to day of the year
    day_of_year = current_date.timetuple().tm_yday

    # Minimum humidity occurs in January and December
    # Maximum humidity occurs in July
    return 12.5 - 0.00027 * (day_of_year % 365 - 180) ** 2


class AbsoluteHumidity(base.Schedule):
    def update_current_val(self, current_date: datetime.date) -> None:
        self.current_val = absolute_humidity_func(current_date)


class FluContactMatrix(base.Schedule):
    """
    Attributes:
        timeseries_df (pd.DataFrame):
            has a "date" column with strings in format "YYYY-MM-DD"
            of consecutive calendar days, and other columns
            named "is_school_day" (bool) and "is_work_day" (bool)
            corresponding to type of day.
        total_contact_matrix (np.ndarray):
            (A x L) x (A x L) np.ndarray, where A is the number
            of age groups and L is the number of risk groups.

    See parent class docstring for other attributes.
    """

    def __init__(self,
                 init_val: Optional[Union[np.ndarray, float]] = None):
        super().__init__(init_val)

        df = pd.read_csv(base_path / "school_work_calendar.csv", index_col=0)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        self.time_series_df = df

        self.total_contact_matrix = np.array([[2.5, 0.5], [2, 1.5]]).reshape((2, 1, 2, 1))
        self.school_contact_matrix = np.array([[0.5, 0], [0.05, 0.1]]).reshape((2, 1, 2, 1))
        self.work_contact_matrix = np.array([[0, 0], [0, 0.0]]).reshape((2, 1, 2, 1))

    def update_current_val(self, current_date: datetime.date) -> None:
        """
        Subclasses must provide a concrete implementation of
        updating self.current_val in-place

        Args:
            current_date (datetime.date):
                real-world date corresponding to
                model's current simulation day
        """

        df = self.time_series_df

        try:
            current_row = df[df["date"] == current_date].iloc[0]
        except IndexError:
            print(f"Error: {current_date} is not in the Calendar's time_series_df.")

        self.current_val = self.total_contact_matrix - \
                           (1 - current_row["is_school_day"]) * self.school_contact_matrix - \
                           (1 - current_row["is_work_day"]) * self.work_contact_matrix


class FluSubpopModel(base.SubpopModel):
    """
    Class for creating ImmunoSEIRS flu model with predetermined fixed
    structure -- initial values and epidemiological structure are
    populated by user-specified JSON files.

    Key method create_transmission_model returns a SubpopModel
    instance with S-E-I-H-R-D compartments and pop_immunity_inf
    and pop_immunity_hosp epi metrics. 
    
    The structure is as follows:
        - S = R_to_S - S_to_E
        - E = S_to_E - E_to_IP - E_to_IA
        - I = new_infected - IS_to_R - IS_to_H
        - H = IS_to_H - H_to_R - H_to_D
        - R = IS_to_R + H_to_R - R_to_S
        - D = H_to_D

    The following are TransitionVariable instances:
        - R_to_S is a RecoveredToSusceptible instance
        - S_to_E is a SusceptibleToExposed instance
        - IP_to_IS is a PresympToSymp instance
        - IS_to_H is a SympToHosp instance
        - IS_to_R is a SympToRecovered instance
        - H_to_R is a HospToRecovered instance
        - H_to_D is a HospToDead instance

    There are three TransitionVariableGroups:
        - E_out (handles E_to_IP and E_to_IA)
        - IS_out (handles IS_to_H and IS_to_R)
        - H_out (handles H_to_R and H_to_D)

    The following are EpiMetric instances:
        - pop_immunity_inf is a PopulationImmunityInf instance
        - pop_immunity_hosp is a PopulationImmunityHosp instance

    Transition rates and update formulas are specified in
        corresponding classes.

    Attributes:
        config (Config):
            holds configuration values.
        fixed_params (FluFixedParams):
            holds epidemiological parameter values, read-in
            from user-specified JSON.
        sim_state (FluSimState):
            holds current simulation state information,
            such as current values of epidemiological compartments
            and epi metrics, read in from user-specified JSON.
        transition_variable_lookup (dict):
            maps string to corresponding TransitionVariable.
        transition_variable_group_lookup (dict):
            maps string to corresponding TransitionVariableGroup.
        compartment_lookup (dict):
            maps string to corresponding EpiCompartment,
            using the value of the EpiCompartment's "name" attribute.
        epi_metric_lookup (dict):
            maps string to corresponding EpiMetric,
            using the value of the EpiMetric's "name" attribute.
    """

    def __init__(self,
                 config_filepath: Optional[str] = None,
                 fixed_params_filepath: Optional[str] = None,
                 state_vars_init_vals_filepath: Optional[str] = None):
        """
        Create Config, FluFixedParams, and FluSimState instances
        using values from respective JSON files, and save these instances
        on the FluSubpopModel to construct a model.

        If any filepath is not specified, then user must manually assign
        the respective attribute (config, fixed_params, or sim_state)
        before using constructor to create a model.

        Attributes:
            config_filepath (Optional[str]):
                path to config JSON file (path includes actual filename
                with suffix ".json") -- all JSON fields must match
                name and datatype of Config instance attributes.
            fixed_params_filepath (Optional[str]):
                path to epidemiological parameters JSON file
                (path includes actual filename with suffix ".json")
                -- all JSON fields must match name and datatype of
                FixedParams instance attributes.
            state_vars_init_vals_filepath (Optional[str]):
                path to epidemiological compartments JSON file
                (path includes actual filename with suffix ".json")
                -- all JSON fields must match name and datatype of
                StateVariable instance attributes -- these initial
                values are used to populate sim_state attribute.
        """

        # Assign config, fixed_params, and sim_state to model-specific
        # types of dataclasses that have epidemiological parameter information
        # and sim state information
        if config_filepath:
            self.config = self.make_dataclass_from_json(base.Config,
                                                        config_filepath)

        if fixed_params_filepath:
            self.fixed_params = self.make_dataclass_from_json(FluFixedParams,
                                                              fixed_params_filepath)

        if state_vars_init_vals_filepath:
            self.sim_state = \
                self.make_dataclass_from_json(FluSimState,
                                              state_vars_init_vals_filepath)

        self.compartment_lookup = sc.objdict()
        self.transition_variable_lookup = sc.objdict()
        self.transition_variable_group_lookup = sc.objdict()
        self.epi_metric_lookup = sc.objdict()
        self.dynamic_val_lookup = sc.objdict()
        self.schedule_lookup = sc.objdict()

        self.setup_model()

        self.compartments = self.compartment_lookup.values()
        self.transition_variables = self.transition_variable_lookup.values()
        self.transition_variable_groups = self.transition_variable_group_lookup.values()
        self.epi_metrics = self.epi_metric_lookup.values()
        self.dynamic_vals = self.dynamic_val_lookup.values()
        self.schedules = self.schedule_lookup.values()

        config = self.config

        self.current_simulation_day = 0

        if isinstance(config.start_real_date, datetime.date):
            self.start_real_date = config.start_real_date
        else:
            try:
                self.start_real_date = \
                    datetime.datetime.strptime(config.start_real_date, "%Y-%m-%d").date()
            except ValueError:
                print("Error: The date format should be YYYY-MM-DD.")

        self.current_real_date = self.start_real_date

    def setup_model(self):

        # Setup objects for model
        self.setup_epi_compartments()
        self.setup_transition_variables()
        self.setup_transition_variable_groups()

        # Some epi metrics depend on transition variables, so
        #   set up epi metrics after transition variables
        self.setup_epi_metrics()
        self.setup_dynamic_vals()
        self.setup_schedules()

    def setup_epi_compartments(self) -> None:
        """
        Create EpiCompartment instances S-E-I-H-R-D (6 compartments total)
        and add them to compartment_lookup for dictionary access
        """

        for name in ("S", "E", "IP", "IS", "IA", "H", "R", "D"):
            self.compartment_lookup[name] = base.EpiCompartment(getattr(self.sim_state, name))

    def setup_dynamic_vals(self) -> None:
        """
        Create all DynamicVal instances and add them to dynamic_val_lookup attribute
            for dictionary access
        """

        self.dynamic_val_lookup["beta_reduct"] = BetaReduct(init_val=0.0,
                                                            is_enabled=False)

    def setup_schedules(self) -> None:
        """
        Create all Schedule instances and add them to schedule_lookup attribute
            for dictionary access
        """

        self.schedule_lookup["absolute_humidity"] = AbsoluteHumidity()
        self.schedule_lookup["flu_contact_matrix"] = FluContactMatrix()

    def setup_transition_variables(self) -> None:
        """
        Create all TransitionVariable instances (7 transition variables total)
            and add them to transition_variable_lookup attribute
            for dictionary access
        """

        transition_type = self.config.transition_type

        transition_variable_lookup = self.transition_variable_lookup
        compartment_lookup = self.compartment_lookup

        S = compartment_lookup.S
        E = compartment_lookup.E
        IP = compartment_lookup.IP
        IS = compartment_lookup.IS
        IA = compartment_lookup.IA
        H = compartment_lookup.H
        R = compartment_lookup.R
        D = compartment_lookup.D

        transition_variable_lookup.R_to_S = RecoveredToSusceptible(R, S, transition_type)
        transition_variable_lookup.S_to_E = SusceptibleToExposed(S, E, transition_type)
        transition_variable_lookup.IP_to_IS = PresympToSymp(IP, IS, transition_type)
        transition_variable_lookup.IA_to_R = AsympToRecovered(IA, R, transition_type)
        transition_variable_lookup.E_to_IP = ExposedToPresymp(E, IP, transition_type, True)
        transition_variable_lookup.E_to_IA = ExposedToAsymp(E, IA, transition_type, True)
        transition_variable_lookup.IS_to_R = SympToRecovered(IS, R, transition_type, True)
        transition_variable_lookup.IS_to_H = SympToHosp(IS, H, transition_type, True)
        transition_variable_lookup.H_to_R = HospToRecovered(H, R, transition_type, True)
        transition_variable_lookup.H_to_D = HospToDead(H, D, transition_type, True)

    def setup_transition_variable_groups(self) -> None:
        """
        Create all transition variable groups described in docstring (2 transition
        variable groups total) and add them to transition_variable_group_lookup attribute
        for dictionary access
        """

        # Shortcuts for attribute access
        transition_variable_lookup = self.transition_variable_lookup
        transition_variable_group_lookup = self.transition_variable_group_lookup
        compartment_lookup = self.compartment_lookup

        transition_type = self.config.transition_type

        transition_variable_group_lookup.E_out = base.TransitionVariableGroup(compartment_lookup.E,
                                                                              transition_type,
                                                                              (transition_variable_lookup.E_to_IP,
                                                                               transition_variable_lookup.E_to_IA))

        transition_variable_group_lookup.IS_out = base.TransitionVariableGroup(compartment_lookup.IS,
                                                                               transition_type,
                                                                               (transition_variable_lookup.IS_to_R,
                                                                                transition_variable_lookup.IS_to_H))

        transition_variable_group_lookup.H_out = base.TransitionVariableGroup(compartment_lookup.H,
                                                                              transition_type,
                                                                              (transition_variable_lookup.H_to_R,
                                                                               transition_variable_lookup.H_to_D))

    def setup_epi_metrics(self) -> None:
        """
        Create all epi metric described in docstring (2 state
        variables total) and add them to epi_metric_lookup attribute
        for dictionary access
        """

        self.epi_metric_lookup.pop_immunity_inf = \
            PopulationImmunityInf(getattr(self.sim_state, "pop_immunity_inf"),
                                  self.transition_variable_lookup.R_to_S)

        self.epi_metric_lookup.pop_immunity_hosp = \
            PopulationImmunityHosp(getattr(self.sim_state, "pop_immunity_hosp"),
                                   self.transition_variable_lookup.R_to_S)
