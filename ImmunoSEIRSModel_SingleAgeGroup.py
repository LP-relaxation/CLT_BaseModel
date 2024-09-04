from BaseModel import BaseModel, SimulationParams
import PlotTools
import numpy as np
import matplotlib.pyplot as plt
import time
import json

from collections import namedtuple


def get_change_in_immunity(current_val,
                               R_val,
                               immunity_increase_factor,
                               total_population_val,
                               saturation_constant,
                               waning_factor):
    return np.asarray((immunity_increase_factor * R_val) / (total_population_val * (1 + saturation_constant * current_val)) \
           - waning_factor * current_val)


def get_new_exposed_rate(I_val,
                             immunity_against_inf,
                             efficacy_against_inf,
                             beta,
                             total_population_val):
    return np.asarray(beta * I_val / (total_population_val * (1 + efficacy_against_inf * immunity_against_inf)))


def get_new_infected_rate(sigma):
    return np.asarray(sigma)


def get_new_hosp_rate(zeta,
                          mu,
                          immunity_against_hosp,
                          efficacy_against_hosp):
    return np.asarray(zeta * mu / (1 + efficacy_against_hosp * immunity_against_hosp))


def get_new_dead_rate(pi,
                          nu,
                          immunity_against_hosp,
                          efficacy_against_death):
    return np.asarray(pi * nu / (1 + efficacy_against_death * immunity_against_hosp))


class ImmunoSEIRSModel(BaseModel):

    def __init__(self, RNG_seed=np.random.SeedSequence()):

        super().__init__(RNG_seed)

        self.add_epi_params_from_json("ImmunoSEIRS_EpiParams.json")
        self.add_simulation_params_from_json("ImmunoSEIRS_SimulationParams.json")

        self.add_epi_compartment("S", np.array([int(1e6) - 2e4]), ["new_susceptible"], ["new_exposed"])
        self.add_epi_compartment("E", np.array([1e4]), ["new_exposed"], ["new_infected"])
        self.add_epi_compartment("I", np.array([1e4]), ["new_infected"], ["new_recovered_home", "new_hosp"])
        self.add_epi_compartment("H", np.array([0.0]), ["new_hosp"], ["new_recovered_hosp", "new_dead"])
        self.add_epi_compartment("R", np.array([0.0]), ["new_recovered_home", "new_recovered_hosp"], ["new_susceptible"])
        self.add_epi_compartment("D", np.array([0.0]), ["new_dead"], [])

        self.add_state_variable("population_immunity_hosp", np.array([0.5]))
        self.add_state_variable("population_immunity_inf", np.array([0.5]))

    def update_change_in_state_variables(self):

        epi_params = self.epi_params

        self.population_immunity_hosp.change_in_current_val = get_change_in_immunity(
            current_val=self.population_immunity_hosp.current_val,
            R_val=self.R.current_val,
            immunity_increase_factor=epi_params.immunity_hosp_increase_factor,
            total_population_val=epi_params.total_population_val,
            saturation_constant=epi_params.immunity_hosp_saturation_constant,
            waning_factor=epi_params.waning_factor_hosp
        )

        self.population_immunity_inf.change_in_current_val = get_change_in_immunity(
            current_val=self.population_immunity_inf.current_val,
            R_val=self.R.current_val,
            immunity_increase_factor=epi_params.immunity_inf_increase_factor,
            total_population_val=epi_params.total_population_val,
            saturation_constant=epi_params.immunity_inf_saturation_constant,
            waning_factor=epi_params.waning_factor_inf
        )

    def update_transition_rates(self):

        epi_params = self.epi_params

        self.new_exposed.current_rate = get_new_exposed_rate(
            I_val=self.I.current_val,
            immunity_against_inf=self.population_immunity_inf.current_val,
            efficacy_against_inf=epi_params.efficacy_immunity_inf,
            total_population_val=epi_params.total_population_val,
            beta=epi_params.beta)

        self.new_hosp.current_rate = get_new_hosp_rate(
            zeta=epi_params.zeta,
           mu=epi_params.mu,
            immunity_against_hosp=self.population_immunity_hosp.current_val,
            efficacy_against_hosp=epi_params.efficacy_immunity_hosp
        )

        self.new_dead.current_rate = get_new_dead_rate(
            pi=epi_params.pi,
            nu=epi_params.nu,
            immunity_against_hosp=self.population_immunity_hosp.current_val,
            efficacy_against_death=epi_params.efficacy_immunity_death
        )

        self.new_infected.current_rate = np.expand_dims(epi_params.sigma, axis=0)

        self.new_recovered_home.current_rate = np.expand_dims((1 - epi_params.mu) * epi_params.gamma, axis=0)

        self.new_recovered_hosp.current_rate = np.expand_dims((1 - epi_params.nu) * epi_params.gamma_hosp, axis=0)

        self.new_susceptible.current_rate = np.expand_dims(epi_params.eta, axis=0)


start = time.time()

simple_model = ImmunoSEIRSModel()

simple_model.simulate_until_time_period(last_simulation_day=365)

print(time.time() - start)

PlotTools.create_basic_compartment_history_plot(simple_model)
