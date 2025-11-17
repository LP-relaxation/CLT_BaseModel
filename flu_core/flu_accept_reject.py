###########################################################
######################## IMPORTS ##########################
###########################################################

import numpy as np
import pandas as pd

import clt_toolkit as clt
from .flu_components import FluMetapopModel

import json


def compute_rsquared(reference_timeseries: list[np.ndarray],
                     simulated_timeseries: list[np.ndarray]) -> float:
    if len(reference_timeseries) != len(simulated_timeseries):
        raise ValueError("Reference time series and simulated time series \n"
                         "must have same length.")

    reference_timeseries = np.asarray(reference_timeseries)
    simulated_timeseries = np.asarray(simulated_timeseries)

    ybar = reference_timeseries.mean(axis=0)

    ss_residual = np.sum(np.square(simulated_timeseries - reference_timeseries))
    ss_total = np.sum(np.square(reference_timeseries - ybar))

    return 1 - ss_residual / ss_total


def accept_reject_admits(metapop_model: FluMetapopModel,
                         sampling_RNG: np.random.Generator,
                         sampling_info: dict[str, dict[str, clt.UniformSamplingSpec]],
                         total_daily_target_admits: list[np.ndarray],
                         num_days: int = 50,
                         target_accepted_reps: int = int(1e2),
                         max_reps: int = int(1e3),
                         early_stop_percent: float = 0.5,
                         target_rsquared: float = 0.75):
    """
    Accept-reject sampler for a metapopulation model.

    This function repeatedly samples parameters from uniform distributions
    (as specified in `spec`) and simulates the model until the R-squared between
    simulated total admits and reference data exceeds `target_rsquared`.
    Accepted parameter sets and simulation states are saved as JSON files.

    Parameters:
        metapop_model (flu.FluMetapopModel):
            The metapopulation model to simulate and sample parameters for.
        sampling_RNG (np.random.Generator):
            Random number generator used for uniform sampling.
        sampling_info (dict[str, dict[str, clt.UniformSamplingSpec]]):
            See `clt_toolkit / sampling / sample_uniform_metapop_params / sampling_info`
            parameter for description.
        total_daily_target_admits (list[np.ndarray]):
            "Target" time series of total admits (across subpopulations)
            for computing R-squared -- we would like parameters and
            sample paths that give simulated admits close to
            `total_daily_target_admits`. Must have length equal to `num_days`.
        num_days (int, default=50):
            Total number of days to simulate for accepted parameter sets.
        target_accepted_reps (int, default=100):
            Target number of accepted parameter sets (replicates) to collect.
        max_reps (int, default=1000):
            Maximum number of sampling attempts before stopping.
        early_stop_percent (float, default=0.5):
            Fraction of `num_days` to simulate initially for early R-squared check.
        target_rsquared (float, default=0.75):
            Minimum R-squared required between simulated and reference admits for acceptance.

    Notes:
    - Early stopping is performed at `num_days * early_stop_percent` to
        quickly reject poor parameter samples.
    - Accepted samples (and the state of the simulation at day
        `num_days`) are saved to JSON files per subpopulation.
        Note that for efficiency, NOT ALL PARAMETERS ARE SAVED!
        Only the parameters that are randomly sampled (and thus are
        different between replications).
    - Running this function can be slow -- test this function with a small
        number of replications or simulation days to start.
    """

    if target_accepted_reps > max_reps:
        max_reps = 10 * target_accepted_reps

    num_days_early_stop = int(num_days * early_stop_percent)

    reps_counter = 0
    accepted_reps_counter = 0

    while reps_counter < max_reps and accepted_reps_counter < target_accepted_reps:

        reps_counter += 1

        metapop_model.reset_simulation()

        param_samples = clt.sample_uniform_metapop_params(metapop_model,
                                                          sampling_RNG,
                                                          sampling_info)

        # Save IS to H transition variable history
        # But do not save daily (compartment) history for efficiency
        for subpop_name, updates_dict in param_samples.items():
            metapop_model.modify_subpop_params(subpop_name, updates_dict)
            metapop_model.modify_simulation_settings({"transition_variables_to_save": ["ISH_to_HR", "ISH_to_HD"],
                                                      "save_daily_history": False})

        metapop_model.simulate_until_day(num_days_early_stop)
        total_simulated_admits = clt.aggregate_daily_tvar_history(metapop_model, ["ISH_to_HR", "ISH_to_HD"])
        current_rsquared = compute_rsquared(reference_timeseries=total_daily_target_admits[:num_days_early_stop],
                                            simulated_timeseries=total_simulated_admits)
        if current_rsquared < target_rsquared:
            continue

        else:
            metapop_model.simulate_until_day(num_days)
            total_simulated_admits = clt.aggregate_daily_tvar_history(metapop_model, ["ISH_to_HR", "ISH_to_HD"])
            current_rsquared = compute_rsquared(reference_timeseries=total_daily_target_admits,
                                                simulated_timeseries=total_simulated_admits)
            if current_rsquared < target_rsquared:
                continue
            else:
                accepted_reps_counter += 1

                for subpop_name, subpop in metapop_model.subpop_models.items():
                    with open("subpop_" + str(subpop_name) + "_rep_" + str(accepted_reps_counter) +
                              "_accepted_sample_params.json", "w") as f:
                        json.dump(clt.serialize_dataclass(param_samples[subpop_name]), f, indent=4)
                    with open("subpop_" + str(subpop_name) + "_rep_" + str(accepted_reps_counter) +
                              "_accepted_state.json", "w") as f:
                        json.dump(clt.serialize_dataclass(subpop.state), f, indent=4)


