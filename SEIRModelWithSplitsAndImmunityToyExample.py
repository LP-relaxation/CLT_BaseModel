from BaseModel import EpiCompartment, TransitionVariable, BaseModel, SimulationParams, EpiParams
import PlotTools
import numpy as np
import matplotlib.pyplot as plt
import time

from collections import namedtuple


def compute_new_exposed_rate(I, beta, phi, total_population_val, num_timesteps):
    return (beta * phi * I / total_population_val) / num_timesteps


def compute_new_infected_rate(kappa, num_timesteps):
    return kappa / num_timesteps


def compute_new_recovered_rate(gamma, num_timesteps):
    return gamma / num_timesteps


def compute_new_susceptible_rate(eta, num_timesteps):
    return eta / num_timesteps

def compute_new_hospitalized_rate(zeta, num_timesteps):
    return zeta / num_timesteps

def compute_new_dead_rate(nu, num_timesteps):
    return nu / num_timesteps


class SimpleModel(BaseModel):

    def update_discretized_rates(self):
        num_timesteps = self.simulation_params.timesteps_per_day

        self.new_exposed.current_rate = compute_new_exposed_rate(
            I=self.I.current_val,
            beta=self.epi_params.beta,
            phi=self.epi_params.phi,
            total_population_val=self.epi_params.total_population_val,
            num_timesteps=num_timesteps
        )

        self.new_infected.current_rate = compute_new_infected_rate(
            kappa=self.epi_params.kappa,
            num_timesteps=num_timesteps
        )

        self.new_recovered.current_rate = compute_new_recovered_rate(
            gamma=self.epi_params.gamma,
            num_timesteps=num_timesteps
        )

        self.new_susceptible.current_rate = compute_new_susceptible_rate(
            eta=self.epi_params.eta,
            num_timesteps=num_timesteps
        )

        self.new_hospitalized.current_rate = compute_new_hospitalized_rate(
            zeta = self.epi_params.zeta,
            num_timesteps=num_timesteps
        )

        self.new_dead.current_rate = compute_new_dead_rate(
            nu = self.epi_params.nu,
            num_timesteps=num_timesteps
        )


start_time = time.time()

# Could maybe make epi_compartments_list read from .json and transition_variables_list read from .csv?
# EpiParams and SimulationParams will eventually be read in from a file

S = EpiCompartment("S", np.array([8500000.0 - 20.0]), ["new_susceptible"], ["new_exposed"])
E = EpiCompartment("E", np.array([0.0]), ["new_exposed"], ["new_infected"])
I = EpiCompartment("I", np.array([20.0]), ["new_infected"], ["new_recovered"])
H = EpiCompartment("H", np.array([0.0]), ["new_hospitalized"], ["new_dead", "new_recovered"])
D = EpiCompartment("D", np.array([0.0]), ["new_dead"], [])
R = EpiCompartment("R", np.array([0.0]), ["new_recovered"], ["new_susceptible"])

new_susceptible = TransitionVariable("new_susceptible", "deterministic", R)
new_exposed = TransitionVariable("new_exposed", "deterministic", S)
new_infected = TransitionVariable("new_infected", "deterministic", E)
new_hospitalized = TransitionVariable("new_hospitalized", "deterministic", I)
new_dead = TransitionVariable("new_dead", "deterministic", H)
new_recovered = TransitionVariable("new_recovered", "deterministic", H)

epi_compartments_list = [S, E, I, H, D, R]

transition_variables_list = [new_susceptible, new_exposed, new_infected, new_hospitalized, new_dead, new_recovered]

epi_params = EpiParams()
epi_params.beta = 0.65
epi_params.phi = 1
epi_params.gamma = 0.2 # recovery rate
epi_params.kappa = 0.331
epi_params.eta = 0.05 # new susceptible rate
epi_params.zeta = 0.03 # hospitalization rate
epi_params.nu = 0.01 # death rate
epi_params.total_population_val = np.array([8500000.0])

simulation_params = SimulationParams(timesteps_per_day=7)

print(time.time() - start_time)

start_time = time.time()

simple_model = SimpleModel(epi_compartments_list,
                           transition_variables_list,
                           epi_params,
                           simulation_params)

simple_model.simulate_until_time_period(last_simulation_day=180)
simple_model.simulate_until_time_period(last_simulation_day=365)

print(time.time() - start_time)

PlotTools.create_basic_compartment_history_plot(simple_model)
