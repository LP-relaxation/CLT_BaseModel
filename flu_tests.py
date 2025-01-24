# WARNING:
#   Currently excluding Poisson transition types from automatic
#   test run. Poisson transition types are not always well-behaved
#   -- this is because the Poisson distribution is unbounded.
#   For example, values of beta_baseline that are too high
#   can result in negative compartment populations.
#   Similarly, if num_timesteps is small, this can result
#   in negative compartment populations. Sometimes tests fail because
#   these choices of parameter values and parameter initial values
#   are unsuitable for well-behaved Poisson random variables.

import flu_model as flu
import clt_base as clt

import numpy as np
import pandas as pd
import copy
import pytest

from pathlib import Path

base_path = Path(__file__).parent / "flu_demo_input_files"

config_filepath = base_path / "config.json"
params_filepath = base_path / "common_params.json"
compartments_epi_metrics_init_vals_filepath = base_path / "compartments_epi_metrics_init_vals.json"
calendar_filepath = base_path / "school_work_calendar.csv"
travel_proportions_filepath = base_path / "travel_proportions.csv"

state_dict = clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath)
params_dict = clt.load_json_new_dict(params_filepath)
config_dict = clt.load_json_new_dict(config_filepath)
calendar_df = pd.read_csv(calendar_filepath, index_col=0)
travel_proportions_df = pd.read_csv(travel_proportions_filepath)

flu_model = flu.FluSubpopModel(state_dict,
                               params_dict,
                               config_dict,
                               calendar_df,
                               np.random.default_rng(88888))

def create_models_all_transition_types_list(RNG_seed):
    models_list = []

    for transition_type in clt.TransitionTypes:

        if "poisson" not in transition_type:
            #  Need deep copy -- otherwise changing "transition_type" on
            #   model_constructor.config changes config attribute for all
            #   models in models_list
            new_config_dict = copy.deepcopy(config_dict)
            new_config_dict["transition_type"] = transition_type

            models_list.append(flu.FluSubpopModel(state_dict,
                                                  params_dict,
                                                  new_config_dict,
                                                  calendar_df,
                                                  np.random.default_rng(RNG_seed)))

    return models_list


starting_random_seed = 123456789123456789

flu_model_variations_list = \
    create_models_all_transition_types_list(starting_random_seed)


def test_correct_object_count():
    """
    Based on this model, there should be 6 epi compartments,
        7 transition variables, 2 transition variable groups,
        and 3 epi metrics
    """

    assert len(flu_model.compartments) == 8
    assert len(flu_model.transition_variables) == 10
    assert len(flu_model.transition_variable_groups) == 3

    if flu_model.wastewater_enabled:
        assert len(flu_model.epi_metrics) == 3
    else:
        assert len(flu_model.epi_metrics) == 2

    assert len(flu_model.dynamic_vals) == 1


def test_model_constructor_no_unintended_sharing():
    """
    Regression test: there was a previous bug where the same
        SubpopState object was being shared across multiple models created
        by subsequent creation calls on the model constructor.
        This is remedied using deep copies -- we make sure that
        the model constructor always creates a transmission model
        with its own distinct/independent SubpopState OBJECT,
        even if the actual initial SubpopState VALUES are the same
        across models.

    This test makes sure that SubpopState objects across models
        created by the same constructor are indeed distinct/independent.
    """

    initial_state_dict = copy.deepcopy(state_dict)

    first_model = flu.FluSubpopModel(state_dict,
                                     params_dict,
                                     config_dict,
                                     calendar_df,
                                     np.random.default_rng(1))

    second_model = flu.FluSubpopModel(state_dict,
                                      params_dict,
                                      config_dict,
                                      calendar_df,
                                      np.random.default_rng(1))

    first_model.simulate_until_time_period(100)

    # The initial state of the second model should still be the same
    #   initial state -- it should not have been affected by simulating
    #   the first model

    for key, value in initial_state_dict.items():
        if isinstance(value, (np.ndarray, list)):
            try:
                assert (getattr(second_model.state, key) ==
                        initial_state_dict[key])
            # if it's an array, have to check equality of each element --
            #   Python will complain that the truth value of an array is ambiguous
            except ValueError:
                assert (getattr(second_model.state, key) ==
                        initial_state_dict[key]).all()


def test_model_constructor_reproducible_results():
    """
    If the flu model constructor creates two identical models
        with the same starting random number seed, they should give
        the same results. Specifically, if the first model is simulated
        before the second model is created, the results should still
        be the same.

    Also a way of ensuring there is no unintended object sharing
        or unintended mutability issues with model constructors.
        Specifically, simulating a model created from a constructor
        should not modify objects on that constructor.
    """

    first_model = flu.FluSubpopModel(state_dict,
                                     params_dict,
                                     config_dict,
                                     calendar_df,
                                     np.random.default_rng(1))

    first_model.simulate_until_time_period(100)

    first_model_history_dict = {}
    first_model_compartments = first_model.compartments

    for name in first_model_compartments.keys():
        first_model_history_dict[name] = getattr(first_model_compartments, name).history_vals_list

    second_model = flu.FluSubpopModel(state_dict,
                                      params_dict,
                                      config_dict,
                                      calendar_df,
                                      np.random.default_rng(1))
    second_model.simulate_until_time_period(100)

    second_model_history_dict = {}
    second_model_compartments = second_model.compartments

    for name in second_model_compartments.keys():
        second_model_history_dict[name] = getattr(second_model_compartments, name).history_vals_list

    for name in first_model_compartments.keys():
        assert np.array_equal(np.array(getattr(first_model_compartments, name).history_vals_list),
                              np.array(getattr(second_model_compartments, name).history_vals_list))


def test_num_timesteps():
    """
    If "timesteps_per_day" in Config increases (number of timesteps per day
        increases), then step sizes are smaller.

    Using binomial deterministic transitions, realizations will be smaller
        for more timesteps per day.
    """

    new_config = copy.deepcopy(flu_model.config)
    new_config.timesteps_per_day = 2
    new_config.transition_type = "binomial_deterministic"

    few_timesteps_model = flu.FluSubpopModel(state_dict,
                                             params_dict,
                                             config_dict,
                                             calendar_df,
                                             np.random.default_rng(starting_random_seed))

    few_timesteps_model.prepare_daily_state()
    few_timesteps_model.simulate_timesteps(1)

    new_config = copy.deepcopy(flu_model.config)
    new_config.timesteps_per_day = 20
    new_config.transition_type = "binomial_deterministic"

    many_timesteps_model = flu.FluSubpopModel(state_dict,
                                              params_dict,
                                              config_dict,
                                              calendar_df,
                                              np.random.default_rng(starting_random_seed))

    many_timesteps_model.prepare_daily_state()
    many_timesteps_model.simulate_timesteps(1)

    for name in few_timesteps_model.transition_variables.keys():
        assert (few_timesteps_model.transition_variables[name].current_val >=
                many_timesteps_model.transition_variables[name].current_val).all()

# breakpoint()


# wastewater test
@pytest.mark.parametrize("model", flu_model_variations_list)
def test_wastewater_when_beta_zero(model):
    """
    If the transmission rate beta_baseline = 0, then viral load should be zero
    """
    if model.wastewater_enabled:
        model.reset_simulation()
        model.params.beta_baseline = 0
        model.simulate_until_time_period(300)

        ww_history = model.epi_metrics["wastewater"].history_vals_list
        tol = 1e-6
        assert np.sum(np.abs(ww_history) < tol) == len(ww_history)


@pytest.mark.parametrize("model", flu_model_variations_list)
def test_no_transmission_when_beta_zero(model):
    """
    If the transmission rate beta_baseline = 0, then S should not decrease over time
    """

    model.reset_simulation()
    model.params.beta_baseline = 0
    model.simulate_until_time_period(300)

    S_history = model.compartments["S"].history_vals_list

    assert np.sum((np.diff(np.sum(S_history, axis=(1, 2))) >= 0)) == len(S_history) - 1


@pytest.mark.parametrize("model", flu_model_variations_list)
def test_dead_compartment_monotonic(model):
    """
    People do not rise from the dead; the dead compartment
        should not decrease over time
    """

    model.reset_simulation()
    model.params.beta = 2
    model.simulate_until_time_period(300)

    D_history = model.compartments["D"].history_vals_list

    assert np.sum(np.diff(np.sum(D_history, axis=(1, 2))) >= 0) == len(D_history) - 1


@pytest.mark.parametrize("model", flu_model_variations_list)
def test_population_is_constant(model):
    """
    The total population (summed over all compartments and age-risk groups)
        should be constant over time, equal to the initial total population.
    """

    model.reset_simulation()
    model.params.beta = 0.25

    for day in range(300):
        model.simulate_until_time_period(day)

        current_sum_all_compartments = 0
        for compartment in model.compartments.values():
            current_sum_all_compartments += np.sum(compartment.current_val)

        assert np.abs(current_sum_all_compartments -
                      np.sum(model.params.total_pop_age_risk)) < 1e-6


@pytest.mark.parametrize("model", flu_model_variations_list)
def test_reset_simulation_reproducible_results(model):
    """
    Resetting the random number generator and simulating should
        give the same results as the initial run.
    """

    model.reset_simulation()
    model.modify_random_seed(starting_random_seed)
    model.simulate_until_time_period(100)

    original_model_history_dict = {}

    for name, compartment in model.compartments.items():
        original_model_history_dict[name] = \
            copy.deepcopy(compartment.history_vals_list)

    reset_model_history_dict = {}

    model.reset_simulation()
    model.modify_random_seed(starting_random_seed)
    model.simulate_until_time_period(100)

    for name, compartment in model.compartments.items():
        reset_model_history_dict[name] = \
            copy.deepcopy(compartment.history_vals_list)

    for name in model.compartments.keys():
        assert np.array_equal(np.array(original_model_history_dict[name]),
                              np.array(reset_model_history_dict[name]))


@pytest.mark.parametrize("model", flu_model_variations_list)
def test_compartments_integer_population(model):
    """
    Compartment populations should be integer-valued.
    """

    model.reset_simulation()
    model.modify_random_seed(starting_random_seed)

    for day in [1, 10, 100]:
        model.simulate_until_time_period(day)

        for compartment in model.compartments.values():
            assert (compartment.current_val ==
                    np.asarray(compartment.current_val, dtype=int)).all()


@pytest.mark.parametrize("model", flu_model_variations_list)
def test_transition_format(model):
    """
    Transition variables' transition rates and
        current value should be A x L, where
        A is the number of risk groups and L is the
        number of age groups.

    Transition rates should also be floats, even though the
        transition variable realization is integer
        (so that population counts in compartments
        always stay integer). Transition rates should be
        floats to prevent premature rounding. Binomial
        and Poisson random variables are always integer,
        but their deterministic equivalents may not be
        under our implementation -- so we round them
        after the fact.
    """

    A = model.params.num_age_groups
    L = model.params.num_risk_groups

    model.reset_simulation()
    model.modify_random_seed(starting_random_seed)

    for day in [1, 10, 100]:
        model.simulate_until_time_period(day)

        for tvar in model.transition_variables.values():
            assert np.shape(tvar.current_rate) == (A, L)
            assert np.shape(tvar.current_val) == (A, L)

            for element in tvar.current_rate.flatten():
                assert isinstance(element, float)
