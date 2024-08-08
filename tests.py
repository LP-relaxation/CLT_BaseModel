from BaseModel import EpiCompartment, TransitionVariable, BaseModel, SimulationParams, EpiParams
from SEIRModel import SimpleModel
import PlotTools
import numpy as np
import matplotlib.pyplot as plt
import time
import unittest

class TestDeterministicLogic(unittest.TestCase):

    def setUp(self) -> None:

        S = EpiCompartment("S", np.array([8500000.0 - 20.0]), ["new_susceptible"], ["new_exposed"])
        E = EpiCompartment("E", np.array([0.0]), ["new_exposed"], ["new_infected"])
        I = EpiCompartment("I", np.array([20.0]), ["new_infected"], ["new_recovered"])
        R = EpiCompartment("R", np.array([0.0]), ["new_recovered"], ["new_susceptible"])

        new_susceptible = TransitionVariable("new_susceptible", "deterministic", R)
        new_exposed = TransitionVariable("new_exposed", "deterministic", S)
        new_infected = TransitionVariable("new_infected", "deterministic", E)
        new_recovered = TransitionVariable("new_recovered", "deterministic", I)

        self.epi_compartments_list = [S, E, I, R]

        self.transition_variables_list = [new_susceptible, new_exposed, new_infected, new_recovered]

        self.epi_params = EpiParams()
        self.epi_params.beta = 0.65
        self.epi_params.phi = 1
        self.epi_params.gamma = 0.2
        self.epi_params.kappa = 0.331
        self.epi_params.eta = 0.05
        self.epi_params.total_population_val = np.array([8500000.0])

        self.simulation_params = SimulationParams(timesteps_per_day=7)

    def test_beta(self):

        self.epi_params.beta = 0

        simple_model = SimpleModel(self.epi_compartments_list,
                                   self.transition_variables_list,
                                   self.epi_params,
                                   self.simulation_params)

        simple_model.simulate_until_time_period(last_simulation_day=365)

        self.assertTrue(np.abs(np.sum(simple_model.name_to_epi_compartment_dict["E"].current_val) < 1e-6))
        self.assertTrue(np.abs(np.sum(simple_model.name_to_epi_compartment_dict["E"].history_vals_list) < 1e-6))

    def test_is_population_constant(self):

        simple_model = SimpleModel(self.epi_compartments_list,
                                   self.transition_variables_list,
                                   self.epi_params,
                                   self.simulation_params)

        for day in range(365):
            simple_model.simulate_until_time_period(last_simulation_day=day)

            current_sum_all_compartments = 0
            for compartment in simple_model.epi_compartments_list:
                current_sum_all_compartments += np.sum(compartment.current_val)

            self.assertTrue(np.abs(current_sum_all_compartments - np.sum(self.epi_params.total_population_val)) < 1e-6)

if __name__ == '__main__':
    print("Running unit tests")
    unittest.main()


