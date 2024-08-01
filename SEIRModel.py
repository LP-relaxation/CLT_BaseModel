from BaseModel import EpiCompartment, TransitionVariable, BaseModel, SimulationParams, EpiParams
import PlotTools
import numpy as np
import matplotlib.pyplot as plt
import time

from collections import namedtuple


def compute_newly_exposed_rate(I, beta, phi, total_population_val, num_timesteps):
    return (beta * phi * I / total_population_val) / num_timesteps


def compute_newly_infected_rate(kappa, num_timesteps):
    return kappa / num_timesteps


def compute_newly_recovered_rate(gamma, num_timesteps):
    return gamma / num_timesteps


def compute_newly_susceptible_rate(eta, num_timesteps):
    return eta / num_timesteps


class SimpleModel(BaseModel):

    def update_discretized_rates(self):
        num_timesteps = self.simulation_params.timesteps_per_day

        self.newly_exposed.current_rate = compute_newly_exposed_rate(
            I=self.I.current_val,
            beta=self.epi_params.beta,
            phi=self.epi_params.phi,
            total_population_val=self.epi_params.total_population_val,
            num_timesteps=num_timesteps
        )

        self.newly_infected.current_rate = compute_newly_infected_rate(
            kappa=self.epi_params.kappa,
            num_timesteps=num_timesteps
        )

        self.newly_recovered.current_rate = compute_newly_recovered_rate(
            gamma=self.epi_params.gamma,
            num_timesteps=num_timesteps
        )

        self.newly_susceptible.current_rate = compute_newly_susceptible_rate(
            eta=self.epi_params.eta,
            num_timesteps=num_timesteps
        )


start_time = time.time()

# Could maybe make list_of_epi_compartments read from .json and list_of_transition_variables read from .csv?
# EpiParams and SimulationParams will eventually be read in from a file

list_of_epi_compartments = [EpiCompartment("S", np.array([8500000.0 - 20.0]), ["newly_susceptible"], ["newly_exposed"]),
                            EpiCompartment("E", np.array([0.0]), ["newly_exposed"], ["newly_infected"]),
                            EpiCompartment("I", np.array([20.0]), ["newly_infected"], ["newly_recovered"]),
                            EpiCompartment("R", np.array([0.0]), ["newly_recovered"], ["newly_susceptible"])]

list_of_transition_variables = [TransitionVariable("newly_susceptible", "deterministic", "R"),
                                TransitionVariable("newly_exposed", "deterministic", "S"),
                                TransitionVariable("newly_infected", "deterministic", "E"),
                                TransitionVariable("newly_recovered", "deterministic", "I")]

epi_params = EpiParams()
epi_params.beta = 0.65
epi_params.phi = 1
epi_params.gamma = 0.2
epi_params.kappa = 0.331
epi_params.eta = 0.05
epi_params.total_population_val = np.array([8500000.0])

simulation_params = SimulationParams(timesteps_per_day=7)

print(time.time() - start_time)

start_time = time.time()

simple_model = SimpleModel(list_of_epi_compartments,
                           list_of_transition_variables,
                           epi_params,
                           simulation_params)

simple_model.simulate_until_time_period(last_simulation_day=180)
simple_model.simulate_until_time_period(last_simulation_day=365)

print(time.time() - start_time)

PlotTools.create_basic_compartment_history_plot(simple_model)
