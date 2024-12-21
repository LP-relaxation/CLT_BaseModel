from flu_components import FluSubpopModel

import base_components as base
import numpy as np
import copy
import pytest

from pathlib import Path

base_path = Path(__file__).parent / "flu_demo_input_files"

config_filepath = base_path / "config.json"
fixed_params_filepath = base_path / "fixed_params.json"
init_vals_filepath = base_path / "state_variables_init_vals.json"

flu_model_constructor = FluSubpopModel(config_filepath,
                                            fixed_params_filepath,
                                            init_vals_filepath)

flu_model = flu_model_constructor.create_transmission_model(RNG_seed=88888888)


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


def create_models_all_transition_types_list(model_constructor, RNG_seed):
    models_list = []

    for transition_type in clt.TransitionTypes:

        if "poisson" not in transition_type:
            #  Need deep copy -- otherwise changing "transition_type" on
            #   model_constructor.config changes config attribute for all
            #   models in models_list
            model_constructor.config = copy.deepcopy(model_constructor.config)

            model_constructor.config.transition_type = transition_type

            models_list.append(model_constructor.create_transmission_model(RNG_seed))

    return models_list


starting_random_seed = 123456789123456789

flu_model_variations_list = \
    create_models_all_transition_types_list(flu_model_constructor,
                                            starting_random_seed)


def test_correct_object_count():
    """
    Based on this model, there should be 6 epi compartments,
        7 transition variables, 2 transition variable groups,
        and 3 epi metrics
    """

    assert len(flu_model.compartments) == 8
    assert len(flu_model.transition_variables) == 10
    assert len(flu_model.transition_variable_groups) == 3
    assert len(flu_model.epi_metrics) == 3
    assert len(flu_model.dynamic_vals) == 1


def test_model_constructor_no_unintended_sharing():
    """
    Regression test: there was a previous bug where the same
        SimState object was being shared across multiple models created
        by subsequent creation calls on the model constructor.
        This is remedied using deep copies -- we make sure that
        the model constructor always creates a transmission model
        with its own distinct/independent SimState OBJECT,
        even if the actual initial SimState VALUES are the same
        across models.

    This test makes sure that SimState objects across models
        created by the same constructor are indeed distinct/independent.
    """

    first_model = flu_model_constructor.create_transmission_model(RNG_seed=1)

    second_model = flu_model_constructor.create_transmission_model(RNG_seed=1)

    first_model.simulate_until_time_period(100)

    # The initial state of the second model should still be the same
    #   initial state -- it should not have been affected by simulating
    #   the first model
    for key in vars(second_model.sim_state).keys():
        try:
            assert (getattr(second_model.sim_state, key) ==
                    getattr(flu_model_constructor.sim_state, key))
        # if it's an array, have to check equality of each element --
        #   Python will complain that the truth value of an array is ambiguous
        except ValueError:
            assert (getattr(second_model.sim_state, key) ==
                    getattr(flu_model_constructor.sim_state, key)).all()


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

    first_model = flu_model_constructor.create_transmission_model(RNG_seed=1)
    first_model.simulate_until_time_period(100)

    first_model_history_dict = {}

    for compartment in first_model.compartments:
        first_model_history_dict[compartment.name] = compartment.history_vals_list

    second_model = flu_model_constructor.create_transmission_model(RNG_seed=1)
    second_model.simulate_until_time_period(100)

    second_model_history_dict = {}

    for compartment in second_model.compartments:
        second_model_history_dict[compartment.name] = compartment.history_vals_list

    for compartment_name in first_model_history_dict.keys():
        assert np.array_equal(np.array(first_model_history_dict[compartment_name]),
                              np.array(second_model_history_dict[compartment_name]))


def test_num_timesteps():
    """
    If "timesteps_per_day" in Config increases (number of timesteps per day
        increases), then step sizes are smaller.

    Using binomial deterministic transitions, realizations will be smaller
        for more timesteps per day.
    """

    flu_model_constructor.config.transition_type = "binomial_deterministic"

    flu_model_constructor.config.timesteps_per_day = 2

    few_timesteps_model = flu_model_constructor.create_transmission_model(RNG_seed=88888888)
    few_timesteps_model.prepare_daily_state()
    few_timesteps_model.simulate_timesteps(1)

    flu_model_constructor.config.timesteps_per_day = 20

    many_timesteps_model = flu_model_constructor.create_transmission_model(RNG_seed=88888888)
    many_timesteps_model.prepare_daily_state()
    many_timesteps_model.simulate_timesteps(1)

    for tvar in few_timesteps_model.transition_variables:
        assert (tvar.current_val >=
                many_timesteps_model.lookup_by_name[tvar.name].current_val).all()

# wastewater test
@pytest.mark.parametrize("model", flu_model_variations_list)
def test_wastewater_when_beta_zero(model):
    """
    If the transmission rate beta_baseline = 0, then viral load should be zero
    """
    model.reset_simulation()
    model.fixed_params.beta_baseline = 0
    model.simulate_until_time_period(300)

    ww_history = model.lookup_by_name["wastewater"].history_vals_list
    tol = 1e-6
    assert np.sum(np.abs(ww_history) < tol) == len(ww_history)



@pytest.mark.parametrize("model", flu_model_variations_list)
def test_no_transmission_when_beta_zero(model):
    """
    If the transmission rate beta_baseline = 0, then S should not decrease over time
    """

    model.reset_simulation()
    model.fixed_params.beta_baseline = 0
    model.simulate_until_time_period(300)

    S_history = model.lookup_by_name["S"].history_vals_list

    assert np.sum((np.diff(np.sum(S_history, axis=(1, 2))) >= 0)) == len(S_history) - 1


@pytest.mark.parametrize("model", flu_model_variations_list)
def test_dead_compartment_monotonic(model):
    """
    People do not rise from the dead; the dead compartment
        should not decrease over time
    """

    model.reset_simulation()
    model.fixed_params.beta = 2
    model.simulate_until_time_period(300)

    D_history = model.lookup_by_name["D"].history_vals_list

    assert np.sum(np.diff(np.sum(D_history, axis=(1, 2))) >= 0) == len(D_history) - 1


@pytest.mark.parametrize("model", flu_model_variations_list)
def test_population_is_constant(model):
    """
    The total population (summed over all compartments and age-risk groups)
        should be constant over time, equal to the initial total population.
    """

    model.reset_simulation()
    model.fixed_params.beta = 0.25

    for day in range(300):
        model.simulate_until_time_period(day)

        current_sum_all_compartments = 0
        for compartment in model.compartments:
            current_sum_all_compartments += np.sum(compartment.current_val)

        assert np.abs(current_sum_all_compartments -
                      np.sum(model.fixed_params.total_population_val)) < 1e-6


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

    for compartment in model.compartments:
        original_model_history_dict[compartment.name] = \
            copy.deepcopy(compartment.history_vals_list)

    reset_model_history_dict = {}

    model.reset_simulation()
    model.modify_random_seed(starting_random_seed)
    model.simulate_until_time_period(100)

    for compartment in model.compartments:
        reset_model_history_dict[compartment.name] = \
            copy.deepcopy(compartment.history_vals_list)

    for compartment_name in original_model_history_dict.keys():
        assert np.array_equal(np.array(original_model_history_dict[compartment_name]),
                              np.array(reset_model_history_dict[compartment_name]))


@pytest.mark.parametrize("model", flu_model_variations_list)
def test_compartments_integer_population(model):
    """
    Compartment populations should be integer-valued.
    """

    model.reset_simulation()
    model.modify_random_seed(starting_random_seed)

    for day in [1, 10, 100]:
        model.simulate_until_time_period(day)

        for compartment in model.compartments:
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

    A = model.fixed_params.num_age_groups
    L = model.fixed_params.num_risk_groups

    model.reset_simulation()
    model.modify_random_seed(starting_random_seed)

    for day in [1, 10, 100]:
        model.simulate_until_time_period(day)

        for tvar in model.transition_variables:
            assert np.shape(tvar.current_rate) == (A, L)
            assert np.shape(tvar.current_val) == (A, L)

            for element in tvar.current_rate.flatten():
                assert isinstance(element, float)

