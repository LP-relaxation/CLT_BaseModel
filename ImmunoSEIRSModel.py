from BaseModel import EpiCompartment, TransitionVariable, TransitionVariableGroup, \
    BaseModel, SimulationParams, EpiParams
import PlotTools
import numpy as np
import matplotlib.pyplot as plt
import time
import json

from collections import namedtuple


def compute_change_in_immunity(current_val,
                               R_val,
                               immunity_increase_factor,
                               total_population_val,
                               saturation_constant,
                               waning_factor):
    return (immunity_increase_factor * R_val) / (total_population_val * (1 + saturation_constant * current_val)) \
           - waning_factor * current_val

def compute_new_exposed_rate(I_val,
                             immunity_against_inf,
                             efficacy_against_inf,
                             beta,
                             total_population_val):
    return beta * I_val / (
            total_population_val * (1 + efficacy_against_inf * immunity_against_inf))


def compute_new_infected_rate(sigma):
    return sigma


def compute_new_recovered_home_rate(mu, gamma):
    return (1 - mu) * gamma


def compute_new_hosp_rate(zeta,
                          mu,
                          immunity_against_hosp,
                          efficacy_against_hosp):
    return zeta * mu / (1 + efficacy_against_hosp * immunity_against_hosp)


def compute_new_recovered_hosp_rate(nu, gamma_hosp):
    return (1 - nu) * gamma_hosp


def compute_new_dead_rate(pi,
                          nu,
                          immunity_against_hosp,
                          efficacy_against_death):
    return pi * nu / (1 + efficacy_against_death * immunity_against_hosp)


def compute_new_susceptible_rate(eta):
    return eta


class ImmunoSEIRModel(BaseModel):

    def compute_change_in_state_variables(self):
        self.population_immunity_hosp.change_in_current_val = compute_change_in_immunity(
            current_val=self.population_immunity_hosp.current_val,
            R_val=self.R.current_val,
            immunity_increase_factor=self.epi_params.immunity_hosp_increase_factor,
            total_population_val=self.epi_params.total_population_val,
            saturation_constant=self.epi_params.immunity_hosp_saturation_constant,
            waning_factor=self.epi_params.waning_factor_hosp
        )

        self.population_immunity_inf.change_in_current_val = compute_change_in_immunity(
            current_val=self.population_immunity_inf.current_val,
            R_val=self.R.current_val,
            immunity_increase_factor=self.epi_params.immunity_inf_increase_factor,
            total_population_val=self.epi_params.total_population_val,
            saturation_constant=self.epi_params.immunity_inf_saturation_constant,
            waning_factor=self.epi_params.waning_factor_inf
        )

    def compute_transition_rates(self):
        self.new_exposed.current_rate = compute_new_exposed_rate(
            beta=self.epi_params.beta,
            I_val=self.I.current_val,
            immunity_against_inf=self.population_immunity_inf.current_val,
            efficacy_against_inf=self.epi_params.efficacy_immunity_inf,
            total_population_val=self.epi_params.total_population_val)

        self.new_infected.current_rate = compute_new_infected_rate(
            sigma=self.epi_params.sigma
        )

        self.new_recovered_home.current_rate = compute_new_recovered_home_rate(
            mu=self.epi_params.mu,
            gamma=self.epi_params.gamma
        )

        self.new_hosp.current_rate = compute_new_hosp_rate(
            zeta=self.epi_params.zeta,
            mu=self.epi_params.mu,
            immunity_against_hosp=self.population_immunity_hosp.current_val,
            efficacy_against_hosp=self.epi_params.efficacy_immunity_hosp
        )

        self.new_recovered_hosp.current_rate = compute_new_recovered_hosp_rate(
            nu=self.epi_params.nu,
            gamma_hosp=self.epi_params.gamma_hosp
        )

        self.new_dead.current_rate = compute_new_dead_rate(
            pi=self.epi_params.pi,
            nu=self.epi_params.nu,
            immunity_against_hosp=self.population_immunity_hosp.current_val,
            efficacy_against_death=self.epi_params.efficacy_immunity_death
        )

        self.new_susceptible.current_rate = compute_new_susceptible_rate(
            eta=self.epi_params.eta
        )

simulation_params = SimulationParams(timesteps_per_day=7)

simple_model = ImmunoSEIRModel(transition_type="binomial")

simple_model.add_epi_params_from_json("ImmunoSEIRSEpiParams.json")
simple_model.add_simulation_params(simulation_params)

simple_model.add_epi_compartment("S", np.array([int(1e6) - 2e4]), ["new_susceptible"], ["new_exposed"])
simple_model.add_epi_compartment("E", np.array([1e4]), ["new_exposed"], ["new_infected"])
simple_model.add_epi_compartment("I", np.array([1e4]), ["new_infected"], ["new_recovered_home", "new_hosp"])
simple_model.add_epi_compartment("H", np.array([0.0]), ["new_hosp"], ["new_recovered_hosp", "new_dead"])
simple_model.add_epi_compartment("R", np.array([0.0]), ["new_recovered_home", "new_recovered_hosp"], ["new_susceptible"])
simple_model.add_epi_compartment("D", np.array([0.0]), ["new_dead"], [])

simple_model.add_state_variable("population_immunity_hosp", np.array([0.5]))
simple_model.add_state_variable("population_immunity_inf", np.array([0.5]))

simple_model.simulate_until_time_period(last_simulation_day=100)

PlotTools.create_basic_compartment_history_plot(simple_model)
