from ImmunoSEIRSModel import ImmunoSEIRSModel
from PlotTools import create_basic_compartment_history_plot

import numpy as np

import copy
import pytest

from pathlib import Path

base_path = Path(__file__).parent

random_seed = np.random.SeedSequence()

model_1age_1risk = ImmunoSEIRSModel(base_path / "instance1_1age_1risk_test" / "epi_params.json",
                                    base_path / "instance1_1age_1risk_test" / "config.json",
                                    base_path / "instance1_1age_1risk_test" / "epi_compartments_state_vars_init_vals.json",
                                    base_path / "instance1_1age_1risk_test" / "state_variables_init_vals.json",
                                    random_seed)

model_2age_2risk = ImmunoSEIRSModel(base_path / "instance2_2age_2risk_test" / "epi_params.json",
                                    base_path / "instance2_2age_2risk_test" / "config.json",
                                    base_path / "instance2_2age_2risk_test" / "epi_compartments_state_vars_init_vals.json",
                                    base_path / "instance2_2age_2risk_test" / "state_variables_init_vals.json",
                                    random_seed)

def create_models_all_transition_types_list(model):

    models_list = []

    for transition_type in ("binomial", "binomial_deterministic",
                            "binomial_taylor_approx", "binomial_taylor_approx_deterministic",
                            "poisson", "poisson_deterministic"):

        model_variation = copy.deepcopy(model)
        model_variation.modify_model_transition_type(transition_type)

        models_list.append(model_variation)

        return models_list

model_1age_1risk_variations_list = create_models_all_transition_types_list(model_1age_1risk)

model_2age_2risk_variations_list = create_models_all_transition_types_list(model_2age_2risk)

@pytest.mark.parametrize("model",
                         model_1age_1risk_variations_list + model_2age_2risk_variations_list)
def test_beta(model):
    '''
    If the transmission rate beta = 0, then S should not decrease over time
    '''

    model.reset_simulation()
    model.epi_params.beta = 0
    model.simulate_until_time_period(last_simulation_day=365)

    S_history = model.name_to_epi_compartment_dict["S"].history_vals_list

    assert np.sum((np.diff(np.sum(S_history, axis=(1,2))) >= 0)) == len(S_history) - 1

@pytest.mark.parametrize("model",
                         model_1age_1risk_variations_list + model_2age_2risk_variations_list)
def test_deaths(model):
    '''
    People do not rise from the dead; the deaths compartment
        should not decrease over time
    '''

    model.reset_simulation()
    model.epi_params.beta = 2
    model.simulate_until_time_period(last_simulation_day=365)

    D_history = model.name_to_epi_compartment_dict["D"].history_vals_list

    assert np.sum(np.diff(np.sum(D_history, axis=(1,2))) >= 0) == len(D_history) - 1

@pytest.mark.parametrize("model",
                         model_1age_1risk_variations_list + model_2age_2risk_variations_list)
def test_population_is_constant(model):
    '''
    The total population (summed over all compartments and age-risk groups)
        should be constant over time, equal to the initial total population.
    '''

    model.reset_simulation()
    model.epi_params.beta = 2

    for day in range(365):
        model.simulate_until_time_period(day)

        current_sum_all_compartments = 0
        for compartment in model.name_to_epi_compartment_dict.values():
            current_sum_all_compartments += np.sum(compartment.current_val)

        assert np.abs(current_sum_all_compartments - np.sum(model.epi_params.total_population_val)) < 1e-6

@pytest.mark.parametrize("model",
                         model_1age_1risk_variations_list + model_2age_2risk_variations_list)
def test_constructor_methods(model):
    '''
    Based on this model, there should be 6 epi compartments,
        7 transition variables, 2 transition variable groups,
        and 2 state variables
    '''

    model.reset_simulation()

    assert len(model.name_to_epi_compartment_dict) == 6
    assert len(model.name_to_transition_variable_dict) == 7
    assert len(model.name_to_transition_variable_group_dict) == 2
    assert len(model.name_to_state_variable_dict) == 2

    assert "I_out" in model.name_to_transition_variable_group_dict
    assert "H_out" in model.name_to_transition_variable_group_dict

@pytest.mark.parametrize("model",
                         model_1age_1risk_variations_list + model_2age_2risk_variations_list)
def test_attribute_assignment(model):
    '''
    Confirm that each epi compartment, transition variable, transition variable group,
        and state variable are assigned as attributes to the model
    '''

    model.reset_simulation()

    for compartment_name in ("S", "E", "I", "H", "R", "D"):
        assert compartment_name in vars(model)

    for tvar_name in ("new_susceptible", "new_exposed", "new_infected", "new_recovered_home", "new_hosp",
                      "new_recovered_hosp", "new_dead"):
        assert tvar_name in vars(model)

    for tvargroup_name in ("I_out", "H_out"):
        assert tvargroup_name in vars(model)

    for svar_name in ("population_immunity_hosp", "population_immunity_inf"):
        assert svar_name in vars(model)

@pytest.mark.parametrize("model",
                         model_1age_1risk_variations_list + model_2age_2risk_variations_list)
def test_reproducible_RNG(model):
    '''
    Resetting the random number generator and simulating should
        give the same results as the initial run.
    '''

    model.reset_simulation()
    model.modify_model_RNG_seed(random_seed)
    model.simulate_until_time_period(100)

    original_model_history_dict = {}

    for compartment_name in model.name_to_epi_compartment_dict.keys():
        original_model_history_dict[compartment_name] = getattr(model, compartment_name).history_vals_list

    reset_model_history_dict = {}

    model.reset_simulation()
    model.modify_model_RNG_seed(random_seed)
    model.simulate_until_time_period(100)

    for compartment_name in model.name_to_epi_compartment_dict.keys():
        reset_model_history_dict[compartment_name] = getattr(model, compartment_name).history_vals_list

    for compartment_name in original_model_history_dict.keys():
        assert np.array_equal(np.array(original_model_history_dict[compartment_name]),\
            np.array(reset_model_history_dict[compartment_name]))