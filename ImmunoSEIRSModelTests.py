from ImmunoSEIRSModel import ImmunoSEIRSModel
from BaseModel import SimulationParams

import numpy as np

import pytest

simulation_params = SimulationParams(timesteps_per_day=7)

model = ImmunoSEIRSModel(is_stochastic=False)

model.add_epi_params_from_json("ImmunoSEIRSEpiParams.json")
model.add_simulation_params(simulation_params)

model.add_epi_compartment("S", np.array([int(1e6) - 2e4]), ["new_susceptible"], ["new_exposed"])
model.add_epi_compartment("E", np.array([1e4]), ["new_exposed"], ["new_infected"])
model.add_epi_compartment("I", np.array([1e4]), ["new_infected"], ["new_recovered_home", "new_hosp"])
model.add_epi_compartment("H", np.array([0.0]), ["new_hosp"], ["new_recovered_hosp", "new_dead"])
model.add_epi_compartment("R", np.array([0.0]), ["new_recovered_home", "new_recovered_hosp"], ["new_susceptible"])
model.add_epi_compartment("D", np.array([0.0]), ["new_dead"], [])

model.add_state_variable("population_immunity_hosp", np.array([0.5]))
model.add_state_variable("population_immunity_inf", np.array([0.5]))

def test_beta():
    '''
    If the transmission rate beta = 0, then S should not decrease over time
    '''

    model.reset()
    model.epi_params.beta = 0
    model.simulate_until_time_period(last_simulation_day=365)

    S_history = model.name_to_epi_compartment_dict["S"].history_vals_list

    assert np.sum((np.diff(np.sum(S_history, axis=1)) >= 0)) == len(S_history) - 1

def test_deaths():
    '''
    People do not rise from the dead; the deaths compartment
        should not decrease over time
    '''

    model.reset()
    model.epi_params.beta = 2
    model.simulate_until_time_period(last_simulation_day=365)

    D_history = model.name_to_epi_compartment_dict["D"].history_vals_list

    assert np.sum((np.diff(np.sum(D_history, axis=1)) >= 0)) == len(D_history) - 1

def test_population_is_constant():
    '''
    The total population (summed over all compartments and age-risk groups)
        should be constant over time, equal to the initial total population.
    '''

    model.reset()
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
        7 transition variables, and 2 transition variable groups
    '''

    assert len(model.name_to_epi_compartment_dict) == 6
    assert len(model.name_to_transition_variable_dict) == 7
    assert len(model.name_to_transition_variable_group_dict) == 2

    assert "I_out" in model.name_to_transition_variable_group_dict
    assert "H_out" in model.name_to_transition_variable_group_dict