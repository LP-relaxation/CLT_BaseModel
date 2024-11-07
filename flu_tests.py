import numpy as np
from flu_components import FluModelConstructor
from plotting import create_basic_compartment_history_plot

import base_components as base
import numpy as np
import copy
import pytest

from pathlib import Path

base_path = Path(__file__).parent / "flu_demo_input_files"

config_filepath = base_path / "config.json"
fixed_params_filepath = base_path / "fixed_params.json"
init_vals_filepath = base_path / "state_variables_init_vals.json"

flu_model_constructor = FluModelConstructor(config_filepath,
                                            fixed_params_filepath,
                                            init_vals_filepath)


def create_models_all_transition_types_list(model_constructor, RNG_seed):
    models_list = []

    for transition_type in base.TransitionTypes:
        model_constructor.config.transition_type = transition_type

        models_list.append(model_constructor.create_transmission_model(RNG_seed))

        return models_list


starting_random_seed = np.random.SeedSequence()

flu_model_variations_list = create_models_all_transition_types_list(flu_model_constructor, starting_random_seed)


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
    People do not rise from the dead; the deaths compartment
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
    model.fixed_params.beta = 2

    for day in range(300):
        model.simulate_until_time_period(day)

        current_sum_all_compartments = 0
        for compartment in model.compartments:
            current_sum_all_compartments += np.sum(compartment.current_val)

        assert np.abs(current_sum_all_compartments - np.sum(model.fixed_params.total_population_val)) < 1e-6


@pytest.mark.parametrize("model", flu_model_variations_list)
def test_constructor_methods(model):
    """
    Based on this model, there should be 6 epi compartments,
        7 transition variables, 2 transition variable groups,
        and 2 epi metrics
    """

    model.reset_simulation()

    assert len(model.compartments) == 6
    assert len(model.transition_variables) == 7
    assert len(model.transition_variable_groups) == 2
    assert len(model.epi_metrics) == 2
    assert len(model.dynamic_vals) == 1


@pytest.mark.parametrize("model", flu_model_variations_list)
def test_reproducible_RNG(model):
    """
    Resetting the random number generator and simulating should
        give the same results as the initial run.
    """

    model.modify_random_seed(starting_random_seed)
    model.reset_simulation()
    model.simulate_until_time_period(300)

    original_model_history_dict = {}

    for compartment in model.compartments:
        original_model_history_dict[compartment.name] = copy.deepcopy(compartment.history_vals_list)

    reset_model_history_dict = {}

    model.reset_simulation()
    model.modify_random_seed(starting_random_seed)
    model.simulate_until_time_period(300)

    for compartment in model.compartments:
        reset_model_history_dict[compartment.name] = copy.deepcopy(compartment.history_vals_list)

    for compartment_name in original_model_history_dict.keys():
        assert np.array_equal(np.array(original_model_history_dict[compartment_name]),
                              np.array(reset_model_history_dict[compartment_name]))


def test_contact_matrix_vectorization(model)