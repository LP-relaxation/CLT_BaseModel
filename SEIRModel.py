from BaseModel import EpiCompartment, BaseModel, SimulationParams, ImmunityTracker
import PlotTools
import numpy as np
import matplotlib.pyplot as plt
import time

from collections import namedtuple


def compute_next_S_val(previous_S_val, newly_susceptible_val, newly_exposed_val):
    return previous_S_val + newly_susceptible_val - newly_exposed_val


def compute_next_E_val(previous_E_val, newly_exposed_val, newly_infected_val):
    return previous_E_val + newly_exposed_val - newly_infected_val




start_time = time.time()

list_of_epi_compartments = [EpiCompartment("S", np.array([8500000])),
                            EpiCompartment("E", np.array([0])),
                            EpiCompartment("I", np.array([20])),
                            EpiCompartment("R", np.array([0]))]

# EpiParams and SimulationParams will eventually be read in from a file
epi_params = namedtuple("EpiParams",
                        ["beta", "phi", "gamma", "kappa", "eta", "total_population_val", "starting_simulation_day"])
epi_params.beta = 0.65
epi_params.phi = 1
epi_params.gamma = 0.2
epi_params.kappa = 0.331
epi_params.eta = 0.05
epi_params.total_population_val = np.array([8500000])

simulation_params = SimulationParams(timesteps_per_day=7)

class SimpleModel(BaseModel):

    def simulate_discretized_timesteps(self):

        epi_params = self.epi_params
        beta = epi_params.beta
        phi = epi_params.phi
        gamma = epi_params.gamma
        kappa = epi_params.kappa
        eta = epi_params.eta
        total_population_val = epi_params.total_population_val

        num_timesteps = self.simulation_params.timesteps_per_day

        S = self.S.current_day_val
        E = self.E.current_day_val
        I = self.I.current_day_val
        R = self.R.current_day_val

        for timestep in range(num_timesteps):
            # Generate (possibly random) transition variables
            newly_exposed_val = (beta * phi * S * I / total_population_val) / num_timesteps
            newly_infected_val = kappa * E / num_timesteps
            newly_recovered_val = gamma * I / num_timesteps
            newly_susceptible_val = eta * R / num_timesteps

            # newly_exposed_val = np.random.binomial(S, (beta * phi * I / total_population_val) / num_timesteps)
            # newly_infected_val = np.random.binomial(E, kappa / num_timesteps)
            # newly_recovered_val = np.random.binomial(I, gamma / num_timesteps)
            # newly_susceptible_val = np.random.binomial(R, eta / num_timesteps)

            # Update counts in each compartment using discretized timestep
            S = S + newly_susceptible_val - newly_exposed_val
            E = E + newly_exposed_val - newly_infected_val
            I = I + newly_infected_val - newly_recovered_val
            R = R + newly_recovered_val - newly_susceptible_val

            self.S.current_timestep_val = S
            self.E.current_timestep_val = E
            self.I.current_timestep_val = I
            self.R.current_timestep_val = R


print(time.time() - start_time)

start_time = time.time()

simple_model = SimpleModel(list_of_epi_compartments, None, epi_params, simulation_params)

simple_model.simulate_until_time_period(last_simulation_day=365)

print(time.time() - start_time)

PlotTools.create_basic_compartment_history_plot(simple_model)