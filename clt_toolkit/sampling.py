import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional

from .base_components import MetapopModel
from .utils import updated_dataclass, daily_sum_over_timesteps

from enum import Enum

from collections import defaultdict


class ParamShapes(str, Enum):
    """
    Defines allowed structural shapes for parameter sampling.

    Specifies which population dimension(s) a parameter varies across.
      - "age": an array of length A (one value per age group)
      - "age_risk": a 2D array of shape (A, R) (values per age × risk group)
      - "scalar": a single value applied to all subpopulations

    Used in `UniformSamplingSpec.param_shapes` to reduce the need for
    manually expanding arrays.
    """

    age = "age"
    age_risk = "age_risk"
    scalar = "scalar"


@dataclass(frozen=True)
class UniformSamplingSpec:
    """
    Holds Uniform distribution info to randomly sample a
    subpop model's `SubpopParams` attribute.

    Attributes:
        lower_bound ([np.ndarray | float]):
            Lower bound(s) of the uniform distribution. Can be a scalar,
            shape (A,) array, or shape (A, R) array depending on `param_shape`.
        upper_bound ([np.ndarray | float]):
            Upper bound(s) of the uniform distribution. Must have the same shape
            as `lower_bound`.
        param_shape (ParamShapes):
            Describes how the parameter varies across subpopulations
            (scalar, by age, or by age and risk).
        num_decimals (positive int):
            Optional number of decimals to keep after rounding -- default is 2.
    """

    lower_bound: Optional[np.ndarray | float] = None
    upper_bound: Optional[np.ndarray | float] = None
    param_shape: Optional[ParamShapes] = None
    num_decimals: Optional[int] = 2


def sample_uniform_matrix(lb: [np.ndarray | float],
                          ub: [np.ndarray | float],
                          RNG: np.random.Generator,
                          A: int,
                          R: int,
                          param_shape: str,
                          ) -> [np.ndarray | float]:
    """
    Sample a matrix X of shape (A,R) such that
    X[a,r] ~ (independent) Uniform(low[a,r], high[a,r]).
    We assume each element is independent, so we do not assume
    any correlation structure:

    Parameters:
        lb (np.ndarray of shape (A,) or (A, R) or float):
            Array or scalar of lower bounds
        ub (np.ndarray of shape (A,) or (A, R) or float):
            Array or scalar of upper bounds
        RNG (np.random.Generator):
            Used to generate Uniform random variables.


    Returns:
        X (np.ndarray of shape (A,) or (A, R) or float):
            Random matrix or scalar realization where each
            element is independently sampled from a Uniform distribution
            with parameters given element-wise by `lb`
            and `ub` -- `X` is same shape as `lb` and `ub`.
    """

    # Use linear transformation of Uniform random variable!
    # Sample standard Uniforms ~[0,1] and apply transformation below
    #   to get Uniforms ~[low, high] element-wise :)

    if param_shape == "age":
        if (np.shape(lb) != (A,) or
                np.shape(ub) != (A,)):
            raise ValueError("With dependence on age, lower bounds and \n"
                             "upper bounds must be arrays of shape (A,). \n"
                             "Fix inputs and try again.")

            U = RNG.uniform(size=lb.shape)
            X = lb + (ub - lb) * U
    elif param_shape == "AR":
        if (np.shape(lb) != (A, R) or
                np.shape(ub) != (A, R)):
            raise ValueError("With dependence on age-risk, lower bounds and \n"
                             "upper bounds must be arrays of shape (A,R). \n"
                             "Fix inputs and try again.")
        U = RNG.uniform(size=lb.shape)
        X = lb + (ub - lb) * U
    elif param_shape == "scalar":
        if not np.isscalar(lb) or not np.isscalar(ub):
            raise ValueError("With dependence type scalar, lower bounds and \n"
                             "upper bounds must be scalars. Fix inputs and try again.")
        U = RNG.uniform()
        X = lb + (ub - lb) * U

    return X


def sample_uniform_metapop_params(metapop_model: MetapopModel,
                                  sampling_RNG: np.random.Generator,
                                  sampling_info: dict[str, dict[str, UniformSamplingSpec]]) \
        -> dict[str, dict[str, np.ndarray]]:
    """
    Draw parameter realizations from uniform distributions for a
    metapopulation model.

    Parameters:
        metapop_model (MetapopModel):
            The metapop model whose subpopulation parameters are sampled.
        sampling_RNG (np.random.Generator):
            Random number generator for Uniform sampling.
        sampling_info (dict[str, dict[str, UniformSamplingSpec]]):
            Nested dictionary with sampling information.
            - Outer keys:
                Either "all_subpop" (apply to all subpopulations)
                or the name of a subpopulation, matching the `name`
                attribute of a `SubpopModel` in `metapop_model`.
            - Inner keys:
                Parameter names corresponding to attributes of the
                `SubpopParams` class associated with the subpop models
                in `metapop_model`.
            - Values:
                `UniformSamplingSpec` objects defining lower/upper bounds
                and shape of the parameter.

    Returns:
        pending_param_updates (dict[str, dict[str, np.ndarray | float]]):
            Nested dictionary of sampled parameter values.
            - Outer keys: subpop names -- similar to description for
                outer keys of `sampling_info` argument. But unlike `sampling_info`,
                there is no `"all_subpop"` key here -- if a parameter applies to
                all subpopulations, the same sampled value appears under each
                subpopulation key.
            - Inner keys: parameter names -- same as description for
                inner keys of `sampling_info` argument.
            - Values: sampled parameters (scalar, 1D array, or 2D array) according to
            the shape specified in `UniformSamplingSpec.param_shape`.
    """

    # These dimensions should be the same across subpopulations
    #   -- so just grab the values from the 1st subpop model
    num_age_groups = metapop_model._subpop_models_ordered[0].params.num_age_groups
    num_risk_groups = metapop_model._subpop_models_ordered[0].params.num_risk_groups

    # We do not want to call `updated_dataclass` repeatedly
    #   when we update a single parameter field for each subpopulation,
    #   because this creates a NEW instance (since the dataclass
    #   is frozen and cannot be edited).
    # Instead, we to call `updated_dataclass` once for each
    #   subpop model.
    # So, for each subpop model, we save the parameters that
    #   need to be changed (to reflect the sampling outcomes)
    #   in a dictionary, hence the nested dictionaries.
    pending_param_updates = defaultdict(dict)  # {subpop_id: {param_name: new_value}}

    for subpop_name, params_dict in sampling_info.items():

        for param_name, param_spec in params_dict.items():
            sample = sample_uniform_matrix(param_spec.lower_bound,
                                           param_spec.upper_bound,
                                           sampling_RNG,
                                           num_age_groups,
                                           num_risk_groups,
                                           param_spec.param_shape)

            sample = np.round(sample, param_spec.num_decimals)

        if subpop_name == "all_subpop":  # sample is the same across all subpop models
            for subpop_id in metapop_model.subpop_models.keys():
                pending_param_updates[subpop_id][param_name] = sample
        else:
            pending_param_updates[subpop_name][param_name] = sample

    return pending_param_updates


def aggregate_daily_tvar_history(metapop_model: MetapopModel,
                                 transition_var_name: str) -> np.ndarray:
    """
    Sum the history values of a given transition variable
    across all subpopulations and across timesteps per day,
    so that we have the total number that transitioned compartments
    in a day.

    Parameters:
        metapop_model (MetapopModel):
            The metapopulation model containing subpopulations.
        transition_var_name (str):
            Name of the transition variable to sum.

    Returns:
        total (np.ndarray):
            Array of shape (num_days, A, R) containing the sum across all subpopulations,
            where A = number of age groups, and R = number of risk groups.
            Each element contains the total number of individuals who transitioned that
            day for the given age and risk group.
    """
    # Convert each subpop's history list to a NumPy array
    all_arrays = [
        np.asarray(getattr(subpop, transition_var_name).history_vals_list)
        for subpop in metapop_model.subpop_models.values()
    ]

    # Stack along new subpop dimension (axis=0) and sum across subpops
    total = np.sum(np.stack(all_arrays, axis=0), axis=0)

    # Each subpopulation model should have the same simulation settings, so
    #   just grab the first subpop model
    num_timesteps = metapop_model.subpop_models[0].simulation_settings.timesteps_per_day

    # Transition variable history contains values recorded at each TIMESTEP.
    # To get DAILY totals, we sum blocks of `num_timesteps` consecutive timesteps.
    return daily_sum_over_timesteps(total, num_timesteps)