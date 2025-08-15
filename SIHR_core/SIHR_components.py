###########################################################
######################## SIHR Model #######################
###########################################################

# This code demonstrates how to inherit from clt_base
#   to create a customized model for one subpopulation
#   -- this is intended as an intermediate tutorial for
#   code users (and as somewhat of an introductory tutorial
#   to inheritance and abstract methods).

# The S-I-H-R model we demonstrate has the following structure:
#   S -> I -> H -> R
#        I ------> R
# So people in the I (infected) compartment either move
#   to H or R (go to the hospital or recover)

# Imports
# numpy is our standard array manipulation package.
# sciris is a nice package that StarSim uses --
#   we will use it for "object dictionaries" --
#   to access values in dictionaries using dot
#   notation, which is very convenient.
# dataclasses provide favorable ways to store data.
# typing (specifically Optional) gives us
#   the ability to allow optional arguments
#   for instantiating dataclasses.
# And finally, don't forget to import the clt_base
#   package -- we'll need this to create our model!

import numpy as np
import sciris as sc
from pathlib import Path

from dataclasses import dataclass
from typing import Optional

import clt_toolkit as clt

# Generally, when creating a custom model, we need
#   to define the following subclasses
#   - Exactly 1 subclass of `clt.SubpopParams` -- to hold our model's
#       fixed parameters
#   - Exactly 1 subclass of `clt.SubpopState` -- to hold our model's
#       simulation state
#   - (Potentially multiple) subclasses of `clt.TransitionVariable`
#       -- one for each type of transition variable --
#       to manage transitions between epidemiological compartments
#   - (Potentially 0 or multiple) subclasses of `clt.StateVariable` --
#       specifically, subclasses of `clt.EpiMetric`, `clt.DynamicVal`,
#       and `clt.Schedule` for any epi metrics, dynamic values,
#       or schedules in our model. Note that we in general
#       do NOT need subclasses of `clt.Compartment` unless
#       we want to create a special compartment with advanced
#       functionality not provided in the `clt.Compartment`
#       class already.
#   - Exactly 1 subclass of `clt.SubpopModel` -- to wrap everything
#       together in our simulation model!

# To keep this S-I-H-R model and demo simple, we do not have epi metrics,
#   dynamic values, or schedules, so we will not need to create
#   associated subclasses.

# STEP ZERO (before coding) is to write down our model structure
#   and write down any mathematical formulas. We also should
#   write down the names (strings) of all the STUFF in our model.
# In the actual code, we will name our variables according to the
#   names we plan out here.
# We have 4 compartments: "S", "I", "H", "R". (We will create 4
#   `clt.Compartment` instances and give them these names respectively.)
# We have 4 transition variables (we will create 4 `clt.TransitionVariable`
#   subclasses with these names):
#   - "SusceptibleToInfected" (for transitions from "S" to "I")
#   - "InfectedToHospitalized" (for transitions from "I" to "H")
#   - "HospitalizedToRecovered" (for transitions from "H" to "R")
#   - "InfectedToRecovered" (for transitions from "I" to "R")
# We have the following fixed/constant parameters (in addition to
#   "num_age_groups" and "num_risk" groups):
#   - "total_pop_age_risk", "beta", "I_to_H_rate", "I_to_R_rate",
#       "H_to_R_rate", "I_to_H_prop"
# The math for transitions is as follows:
#   - "S" to "I" transition rate: "I" * "beta" / "total_pop_age_risk"
#   - "I" to "H" transition rate: "I_to_H_rate" * "I_to_H_prop"
#   - "I" to "R" transition rate: "I_to_R_rate" * 1 - "I_to_H_prop"
#   - "H" to "R" transition rate: "H_to_R_rate"

# First, we create our `clt.SubpopParams` subclass.
# We need to add the `@dataclass` decorator before the class
#   definition to use the convenient dataclass data storage
#   functionality.
# We specify the names of each field -- these strings
#   must be unique and descriptive names for fixed parameters
#   in our model.
# In general, we recommend adding the fields "num_age_groups"
#   and "num_risk_groups" -- as the inclusion of these fields
#   makes certain computations more convenient. Many arrays
#   in the model will have size (`num_age_groups` x `num_risk_groups`).
# We must define the data type of each field. Here we use
#   `Optional` to also indicate a default value for
#   entries if they are not specified.
# Read more about `dataclasses` for more syntax details.
@dataclass
class SIHRSubpopParams(clt.SubpopParams):
    """
    Data container for pre-specified and fixed epidemiological
    parameters in SIHR model.

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
        I_to_H_rate (positive float):
            rate at which people in I move to H --
            units of people per day.
        I_to_R_rate (positive float):
            rate at which people in I move to R --
            units of people per day.
        H_to_R_rate (positive float):
            rate at which people in H move to R --
            units of people per day.
        I_to_H_prop (np.ndarray):
            contains values in [0,1] corresponding to
            probability of going to hospital given
            infection, for a specific age-risk group
            (age is given by row index, risk is
            given by column index).
    """

    num_age_groups: Optional[int] = None
    num_risk_groups: Optional[int] = None
    total_pop_age_risk: Optional[np.ndarray] = None
    beta: Optional[float] = None
    I_to_H_rate: Optional[float] = None
    I_to_R_rate: Optional[float] = None
    H_to_R_rate: Optional[float] = None
    I_to_H_prop: Optional[np.ndarray] = None

# Next, we create our `clt.SubpopState` subclass.
# We also need the `@dataclass` decorator here and we
#   also need to specify the datatype of each field.
@dataclass
class SIHRSubpopState(clt.SubpopState):
    """
    Data container for pre-specified and fixed set of
    Compartment initial values and EpiMetric initial values
    in SIHR model.

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
        H (np.ndarray of positive floats):
            hospitalized compartment for age-risk groups
            (holds current_val of Compartment "H").
        R (np.ndarray of positive floats):
            recovered compartment for age-risk groups
            (holds current_val of Compartment "R").
    """

    S: Optional[np.ndarray] = None
    I: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None


# For each transition variable, we create a subclass of
#   `clt.TransitionVariable.`
# Note that `clt.TransitionVariable` is an abstract
#   base class, and has an abstract method
#   `get_current_rate` that we need to concretely
#   implement in our subclass.
# The method `get_current_rate` takes in `state`
#   and `params`, which are respectively instances of
#   `SubpopState` (specifically, `SIHRSubpopState`)
#   and `SubpopParams` (specifically `SIHRSubpopParams`).
#   The reason we pass these arguments is for easy
#   access of parameters and simulation state.
# For each transition variable subclass, we need
#   to provide a concrete implementation of
#   `get_current_rate` and return the actual current rate
#   (which is a function of the parameters and
#   simulation state). We follow the mathematical formulas
#   we have previously defined.

class SusceptibleToInfected(clt.TransitionVariable):

    def get_current_rate(self,
                         state: SIHRSubpopState,
                         params: SIHRSubpopParams) -> np.ndarray:

        return state.I * params.beta / params.total_pop_age_risk


class InfectedToHospitalized(clt.TransitionVariable):

    def get_current_rate(self,
                         state: SIHRSubpopState,
                         params: SIHRSubpopParams) -> np.ndarray:
        return params.I_to_H_rate * params.I_to_H_prop


class HospitalizedToRecovered(clt.TransitionVariable):

    def get_current_rate(self,
                         state: SIHRSubpopState,
                         params: SIHRSubpopParams) -> np.ndarray:
        return params.H_to_R_rate


class InfectedToRecovered(clt.TransitionVariable):

    def get_current_rate(self,
                         state: SIHRSubpopState,
                         params: SIHRSubpopParams) -> np.ndarray:

        return params.I_to_R_rate * (1 - params.I_to_H_prop)

# Finally, we put everything together in a `SubpopModel` :)
# We create a subclass of `clt.SubpopModel` -- here we
#   named our subclass `SIHRSubpopModel.`
# We will go through each function step-by-step.


class SIHRSubpopModel(clt.SubpopModel):

    # If we look at the `clt.SubpopModel` base class,
    #   we see that `clt.SubpopModel`'s `__init__` function
    #   requires the arguments `state` (a `SubpopState`),
    #   `params` (a `SubpopParams`), `simulation_settings` (a `SimulationSettings`)
    #   and `RNG` (an `np.random.Generator`).
    # For *our* custom model, we add additional functionality
    #   to `__init__` to allow the user to specify dictionaries
    #   containing model information, and then generate the
    #   `SubpopState`, `SubpopParams`, `SimulationSettings` instances
    #   from those dictionaries.
    # Note that customizing our own `__init__` is TOTALLY OPTIONAL!
    #   If a subclass does not have its own `__init__` defined,
    #   it simply inherits its parent's `__init__`.
    # So, we create an `__init__` function for our subclass.
    # Within our `__init__`, we use the helper function
    #   `clt.make_dataclass_from_dict` to convert dictionaries
    #   to actual base model objects.
    # Then we call `super().__init__` to call the original
    #   initialization commands in the base/parent class.
    # This puts everything together in a model!
    def __init__(self,
                 compartments_epi_metrics_dict: dict,
                 params_dict: dict,
                 simulation_settings_dict: dict,
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
            simulation_settings_dict (dict):
                holds simulation settings -- keys and values
                respectively must match field names and format of
                SimulationSettings.
            RNG (np.random.Generator):
                numpy random generator object used to obtain
                random numbers.
            name (str):
                name.
            wastewater_enabled (bool):
                if True, includes "wastewater" EpiMetric. Otherwise,
                excludes it.
        """

        # Assign simulation settings, params, and state to model-specific
        # types of dataclasses that have epidemiological parameter information
        # and sim state information

        self.wastewater_enabled = wastewater_enabled

        state = clt.make_dataclass_from_dict(SIHRSubpopState, compartments_epi_metrics_dict)
        params = clt.make_dataclass_from_dict(SIHRSubpopParams, params_dict)
        simulation_settings = clt.make_dataclass_from_dict(clt.SimulationSettings, simulation_settings_dict)

        # IMPORTANT NOTE: as always, we must be careful with mutable objects
        #   and generally use deep copies to avoid modification of the same
        #   object. But in this function call, using deep copies is unnecessary
        #   (redundant) because the parent class SubpopModel's __init__()
        #   creates deep copies.
        super().__init__(state, params, simulation_settings, RNG, name)

    # `clt.SubpopModel` is an abstract base class, and has multiple
    #   abstract methods that we must implement:
    #   - `create_interaction_terms`
    #   - `create_dynamic_vals`
    #   - `create_schedules`
    #   - `create_epi_metrics`
    #   - `create_compartments`
    #   - `create_transition_variables`
    #   - `create_transition_variable_groups`
    # We do not use `InteractionTerm` instances because these
    #   are for `MetapopModel`s comprised of multiple
    #   `SubpopModel`s. We still need to implement the
    #   `create_interaction_terms` method, however, but we simply
    #   return an empty `sc.objdict`.
    # Similarly, we return an empty `sc.objdict` for
    #   `create_dynamic_vals`, `create_schedules`, and
    #   `create_epi_metrics` because we do not have any
    #   such objects in our simple model.

    def create_interaction_terms(self) -> sc.objdict[str, clt.InteractionTerm]:

        return sc.objdict()

    def create_dynamic_vals(self) -> sc.objdict[str, clt.DynamicVal]:

        dynamic_vals = sc.objdict()

        return dynamic_vals

    def create_schedules(self) -> sc.objdict[str, clt.Schedule]:

        schedules = sc.objdict()

        return schedules

    def create_epi_metrics(
            self,
            transition_variables: sc.objdict[str, clt.TransitionVariable]) \
            -> sc.objdict[str, clt.EpiMetric]:

        epi_metrics = sc.objdict()

        return epi_metrics

    def create_compartments(self) -> sc.objdict[str, clt.Compartment]:
        """
        Create a `Compartment` instance for each compartment
        ("S", "I", "H", "R"). Store each instance in an `sc.objdict`,
        where the keys are the names (strings) of each compartment,
        and the values are the compartment instances themselves.
        Return the dictionary.
        """

        compartments = sc.objdict()

        for name in ("S", "I", "H", "R"):
            compartments[name] = clt.Compartment(getattr(self.state, name))

        return compartments

    def create_transition_variables(
            self,
            compartments: sc.objdict[str, clt.Compartment] = None) -> sc.objdict[str, clt.TransitionVariable]:
        """
        For each transition (we have 4 total), create an instance
        of the associated `clt.TransitionVariable` subclass. To
        initialize each `TransitionVariable`, we need to specify the
        origin `Compartment`, the destination `Compartment`, and the
        transition type.

        For example, for the transition between the "S" and "I"
        compartments, create an instance of `SusceptibleToInfected`
        (which we created in the code above). We need to pass the
        corresponding `Compartment` instances to the `origin`
        and `destination` arguments, in addition to specifying the
        `transition_type`.
        """

        # Grab the `transition_type` specified in `simulation_settings`
        type = self.simulation_settings.transition_type

        transition_variables = sc.objdict()

        S = compartments.S
        I = compartments.I
        H = compartments.H
        R = compartments.R

        transition_variables.S_to_I = SusceptibleToInfected(origin=S, destination=I, transition_type=type)
        transition_variables.I_to_R = InfectedToRecovered(origin=I, destination=R, transition_type=type)
        transition_variables.I_to_H = InfectedToHospitalized(origin=I, destination=H, transition_type=type)
        transition_variables.H_to_R = HospitalizedToRecovered(origin=H, destination=R, transition_type=type)

        return transition_variables

    def create_transition_variable_groups(
            self,
            compartments: sc.objdict[str, clt.Compartment] = None,
            transition_variables: sc.objdict[str, clt.TransitionVariable] = None)\
            -> sc.objdict[str, clt.TransitionVariableGroup]:
        """
        When there are multiple transitions out of a single compartment,
        we need a `TransitionVariableGroup` to handle the jointly distributed
        transition logic properly.

        In our model, the "I" compartment has two outgoing arcs: one to "H"
        and one to "R".

        We create a `TransitionVariableGroup` saved as `"I_out"` in the
        transition variable group dictionary. We need to specify the `Compartment`
        instance corresponding to `origin`, the `transition_type`, and also
        a tuple (or list) of `TransitionVariable` instances that make up this
        joint distribution.
        """

        transition_type = self.simulation_settings.transition_type

        transition_variable_groups = sc.objdict()

        transition_variable_groups.I_out = clt.TransitionVariableGroup(origin=compartments.I,
                                                                       transition_type=transition_type,
                                                                       transition_variables=
                                                                       (transition_variables.I_to_R,
                                                                        transition_variables.I_to_H))

        return transition_variable_groups
