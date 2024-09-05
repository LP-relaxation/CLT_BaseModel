from ImmunoSEIRSModel_SingleAgeGroup import ImmunoSEIRSModel
from BaseModel import SimulationParams

import numpy as np

import copy
import pytest

random_seed = np.random.SeedSequence()

def test_beta():
    '''
    If the transmission rate beta = 0, then S should not decrease over time
    '''

    model = ImmunoSEIRSModel(random_seed)
    model.epi_params.beta = 0
    model.simulate_until_time_period(last_simulation_day=365)

    S_history = model.name_to_epi_compartment_dict["S"].history_vals_list

    assert np.sum((np.diff(np.sum(S_history, axis=1)) >= 0)) == len(S_history) - 1

def test_deaths():
    '''
    People do not rise from the dead; the deaths compartment
        should not decrease over time
    '''

    model = ImmunoSEIRSModel(random_seed)
    model.epi_params.beta = 2
    model.simulate_until_time_period(last_simulation_day=365)

    D_history = model.name_to_epi_compartment_dict["D"].history_vals_list

    assert np.sum((np.diff(np.sum(D_history, axis=1)) >= 0)) == len(D_history) - 1

def test_population_is_constant():
    '''
    The total population (summed over all compartments and age-risk groups)
        should be constant over time, equal to the initial total population.
    '''

    model = ImmunoSEIRSModel(random_seed)
    model.epi_params.beta = 2

    for day in range(365):
        model.simulate_until_time_period(day)

        current_sum_all_compartments = 0
        for compartment in model.name_to_epi_compartment_dict.values():
            current_sum_all_compartments += np.sum(compartment.current_val)

        assert np.abs(current_sum_all_compartments - np.sum(model.epi_params.total_population_val)) < 1e-6

def test_constructor_methods():
    '''
    Based on this model, there should be 6 epi compartments,
        7 transition variables, 2 transition variable groups,
        and 2 state variables
    '''

    model = ImmunoSEIRSModel(random_seed)

    assert len(model.name_to_epi_compartment_dict) == 6
    assert len(model.name_to_transition_variable_dict) == 7
    assert len(model.name_to_transition_variable_group_dict) == 2
    assert len(model.name_to_state_variable_dict) == 2

    assert "I_out" in model.name_to_transition_variable_group_dict
    assert "H_out" in model.name_to_transition_variable_group_dict

def test_attribute_assignment():
    '''
    Confirm that each epi compartment, transition variable, transition variable group,
        and state variable are assigned as attributes to the model
    '''

    model = ImmunoSEIRSModel(random_seed)

    for compartment_name in ("S", "E", "I", "H", "R", "D"):
        assert compartment_name in vars(model)

    for tvar_name in ("new_susceptible", "new_exposed", "new_infected", "new_recovered_home", "new_hosp",
                      "new_recovered_hosp", "new_dead"):
        assert tvar_name in vars(model)

    for tvargroup_name in ("I_out", "H_out"):
        assert tvargroup_name in vars(model)

    for svar_name in ("population_immunity_hosp", "population_immunity_inf"):
        assert svar_name in vars(model)

def test_reproducible_RNG():
    '''
    Resetting the random number generator and simulating should
        give the same results as the initial run.
    '''

    model = ImmunoSEIRSModel(random_seed)
    model.epi_params.beta = 2
    model.simulate_until_time_period(100)

    model_copy = copy.deepcopy(model)
    model_copy.reset_simulation()
    model_copy.bit_generator = np.random.MT19937(seed=random_seed)
    model_copy.RNG = np.random.Generator(model_copy.bit_generator)
    model_copy.simulate_until_time_period(100)

    for compartment_name in model.name_to_epi_compartment_dict.keys():
        assert getattr(model, compartment_name).history_vals_list == getattr(model_copy, compartment_name).history_vals_list
