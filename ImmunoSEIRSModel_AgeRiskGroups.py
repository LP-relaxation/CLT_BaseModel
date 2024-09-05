from BaseModel import BaseModel, SimulationParams
import PlotTools
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import pandas as pd

from collections import namedtuple

# global values
VARIANT_DIM = 2
VACCINE_DIM = 1

def get_change_in_immunity_H1(immune_H1_val,
                              immune_H3_val,
                              immune_V_val,
                               R_val,
                               immunity_increase_factor,
                               total_population_val,
                               saturation_matrix, #saturation_matrix, # convert to saturation matrix age_dim * [variant_dim + vaccine_dim] saturation_constant,
                               waning_factor):

    num_age_group, num_risk_group = immune_H1_val.shape

    res = np.zeros(shape=(num_age_group,num_risk_group))


    for age_index in range(num_age_group):
        for risk_index in range(num_risk_group):
            denominator = 1
            denominator += saturation_matrix[age_index,risk_index,0] * immune_H1_val[age_index,risk_index]
            denominator += saturation_matrix[age_index,risk_index,1] * immune_H3_val[age_index,risk_index]
            denominator += saturation_matrix[age_index,risk_index,2] * immune_V_val[age_index,risk_index]
            denominator *= total_population_val[age_index,risk_index]
            res[age_index, risk_index] = (immunity_increase_factor * R_val[age_index,risk_index]) / denominator  - waning_factor * immune_H1_val[age_index,risk_index]

    return res


def get_change_in_immunity_H3(immune_H1_val,
                              immune_H3_val,
                              immune_V_val,
                               R_val,
                               immunity_increase_factor,
                               total_population_val,
                               saturation_matrix, #saturation_matrix, # convert to saturation matrix age_dim * [variant_dim + vaccine_dim] saturation_constant,
                               waning_factor):
    num_age_group, num_risk_group = immune_H1_val.shape

    res = np.zeros(shape=(num_age_group, num_risk_group))

    for age_index in range(num_age_group):
        for risk_index in range(num_risk_group):
            denominator = 1
            denominator += saturation_matrix[age_index, risk_index, 0] * immune_H1_val[age_index,risk_index]
            denominator += saturation_matrix[age_index, risk_index, 1] * immune_H3_val[age_index,risk_index]
            denominator += saturation_matrix[age_index, risk_index, 2] * immune_V_val[age_index,risk_index]
            denominator *= total_population_val[age_index, risk_index]
            res[age_index, risk_index] = (immunity_increase_factor * R_val[age_index,risk_index]) / denominator - waning_factor * \
                                         immune_H3_val[age_index, risk_index]

    return res

def get_change_in_immunity_V(immune_V_val,
                               vaccine_doses,
                               index_date,
                               vaccine_in_effect_delay,
                               immunity_increase_factor,
                               waning_factor):
    # assume uniform increase
    index_vaccine_in_effect_date = index_date - vaccine_in_effect_delay
    if index_vaccine_in_effect_date < 0:
        return -waning_factor * immune_V_val
    else:
        return immunity_increase_factor * vaccine_doses[index_vaccine_in_effect_date] - waning_factor * immune_V_val




def get_new_exposed_rate(I_val,
                             immunity_against_inf_H1,
                             immunity_against_inf_H3,
                             immunity_against_inf_V,
                             efficacy_against_inf,
                             beta,
                             contact_matrix,
                             total_population_val):
    num_age_group, num_risk_group = I_val.shape
    res = np.zeros(shape=(num_age_group, num_risk_group))
    for age_index in range(num_age_group):
        for risk_index in range(num_risk_group):
            immune_efficacy = 1
            immune_efficacy += efficacy_against_inf[0] * immunity_against_inf_H1[age_index,risk_index]
            immune_efficacy += efficacy_against_inf[1] * immunity_against_inf_H3[age_index, risk_index]
            immune_efficacy += efficacy_against_inf[2] * immunity_against_inf_V[age_index, risk_index]
            for age_index2 in range(num_age_group):
                for risk_index2 in range(num_risk_group):
                    res[age_index,risk_index] += beta * contact_matrix[age_index,age_index2] * I_val[age_index2,risk_index2] \
                                                 / total_population_val[age_index2,risk_index2] / immune_efficacy


    return res


def get_new_infected_rate(sigma):
    return np.asarray(sigma)


def get_new_hosp_rate(zeta,
                          mu,
                      immunity_against_hosp_H1,
                      immunity_against_hosp_H3,
                      immunity_against_hosp_V,
                          efficacy_against_hosp):
    num_age_group, num_risk_group = immunity_against_hosp_H1.shape
    res = np.ones(shape=(num_age_group,num_risk_group)) * zeta * mu
    for age_index in range(num_age_group):
        for risk_index in range(num_risk_group):
            immune_efficacy = 1
            immune_efficacy += efficacy_against_hosp[0] * immunity_against_hosp_H1[age_index,risk_index]
            immune_efficacy += efficacy_against_hosp[1] * immunity_against_hosp_H3[age_index, risk_index]
            immune_efficacy += efficacy_against_hosp[2] * immunity_against_hosp_V[age_index, risk_index]

            res[age_index,risk_index] = res[age_index,risk_index] / immune_efficacy

    return res


def get_new_dead_rate(pi,
                          nu,
                          immunity_against_hosp_H1,
                          immunity_against_hosp_H3,
                          immunity_against_hosp_V,
                          efficacy_against_death):
    num_age_group, num_risk_group = immunity_against_hosp_H1.shape
    res = np.ones(shape=(num_age_group, num_risk_group)) * pi * nu
    for age_index in range(num_age_group):
        for risk_index in range(num_risk_group):
            immune_efficacy = 1
            immune_efficacy += efficacy_against_death[0] * immunity_against_hosp_H1[age_index,risk_index]
            immune_efficacy += efficacy_against_death[1] * immunity_against_hosp_H3[age_index, risk_index]
            immune_efficacy += efficacy_against_death[2] * immunity_against_hosp_V[age_index, risk_index]

    return res


class ImmunoSEIRSModel(BaseModel):

    def __init__(self, RNG_seed=np.random.SeedSequence()):

        super().__init__(RNG_seed)

        self.add_epi_params_from_json("ImmunoSEIRS_EpiParams.json")
        self.add_simulation_params_from_json("ImmunoSEIRS_SimulationParams.json")
        self.read_vaccine_data("vaccine.csv")

        self.epi_params.immunity_hosp_saturation_constant = np.array(self.epi_params.immunity_hosp_saturation_constant)
        self.epi_params.immunity_inf_saturation_constant = np.array(self.epi_params.immunity_inf_saturation_constant)
        self.epi_params.total_population_val = np.array(self.epi_params.total_population_val)
        self.epi_params.contact_matrix = np.array(self.epi_params.contact_matrix)

        self.add_epi_compartment("S", np.array([[1e6-2e4, 1e6-2e4],
                                                [1e6-2e4, 1e6-2e4]]), ["new_susceptible"], ["new_exposed"])
        self.add_epi_compartment("E", np.array([[1e4, 1e4],
                                                [1e4, 1e4]]), ["new_exposed"], ["new_infected"])
        self.add_epi_compartment("I", np.array([[1e4, 1e4],
                                                [1e4, 1e4]]), ["new_infected"], ["new_recovered_home", "new_hosp"])
        self.add_epi_compartment("H", np.array([[0.0, 0.0],
                                                [0.0, 0.0]]), ["new_hosp"], ["new_recovered_hosp", "new_dead"])
        self.add_epi_compartment("R", np.array([[0.0, 0.0],
                                                [0.0, 0.0]]), ["new_recovered_home", "new_recovered_hosp"], ["new_susceptible"])
        self.add_epi_compartment("D", np.array([[0.0, 0.0],
                                                [0.0, 0.0]]), ["new_dead"], [])

        self.add_state_variable("population_immunity_hosp_H1", np.array([[0.5, 0.5],
                                                                      [0.5, 0.5]]))
        self.add_state_variable("population_immunity_hosp_H3", np.array([[0.5, 0.5],
                                                                         [0.5, 0.5]]))
        self.add_state_variable("population_immunity_hosp_V", np.array([[0.5, 0.5],
                                                                         [0.5, 0.5]]))

        self.add_state_variable("population_immunity_inf_H1", np.array([[0.5, 0.5],
                                                                      [0.5, 0.5]]))
        self.add_state_variable("population_immunity_inf_H3", np.array([[0.6, 0.5],
                                                                        [0.5, 0.5]]))
        self.add_state_variable("population_immunity_inf_V", np.array([[0.5, 0.5],
                                                                        [0.5, 0.5]]))

    def update_change_in_state_variables(self): # for immunity variable

        epi_params = self.epi_params

        self.population_immunity_hosp_H1.change_in_current_val = get_change_in_immunity_H1(
            immune_H1_val=self.population_immunity_hosp_H1.current_val,
            immune_H3_val=self.population_immunity_hosp_H3.current_val,
            immune_V_val=self.population_immunity_hosp_V.current_val,
            R_val=self.R.current_val,
            immunity_increase_factor=epi_params.immunity_hosp_increase_factor,
            total_population_val=epi_params.total_population_val,
            saturation_matrix=epi_params.immunity_hosp_saturation_constant,
            waning_factor=epi_params.waning_factor_hosp
        )

        self.population_immunity_hosp_H3.change_in_current_val = get_change_in_immunity_H3(
            immune_H1_val=self.population_immunity_hosp_H1.current_val,
            immune_H3_val=self.population_immunity_hosp_H3.current_val,
            immune_V_val=self.population_immunity_hosp_V.current_val,
            R_val=self.R.current_val,
            immunity_increase_factor=epi_params.immunity_hosp_increase_factor,
            total_population_val=epi_params.total_population_val,
            saturation_matrix=epi_params.immunity_hosp_saturation_constant,
            waning_factor=epi_params.waning_factor_hosp
        )

        self.population_immunity_hosp_V.change_in_current_val = get_change_in_immunity_V(
            immune_V_val=self.population_immunity_hosp_V.current_val,
            vaccine_doses=self.data_vaccine,
            index_date=self.current_day_counter,
            vaccine_in_effect_delay=epi_params.vaccine_in_effect_delay,
            immunity_increase_factor=epi_params.immunity_vaccine_increase_factor,
            waning_factor=epi_params.waning_factor_vaccine_hosp
            )


        self.population_immunity_inf_H1.change_in_current_val = get_change_in_immunity_H1(
            immune_H1_val=self.population_immunity_inf_H1.current_val,
            immune_H3_val=self.population_immunity_inf_H3.current_val,
            immune_V_val=self.population_immunity_inf_V.current_val,
            R_val=self.R.current_val,
            immunity_increase_factor=epi_params.immunity_inf_increase_factor,
            total_population_val=epi_params.total_population_val,
            saturation_matrix=epi_params.immunity_inf_saturation_constant,
            waning_factor=epi_params.waning_factor_inf
        )

        self.population_immunity_inf_H3.change_in_current_val = get_change_in_immunity_H3(
            immune_H1_val=self.population_immunity_inf_H1.current_val,
            immune_H3_val=self.population_immunity_inf_H3.current_val,
            immune_V_val=self.population_immunity_inf_V.current_val,
            R_val=self.R.current_val,
            immunity_increase_factor=epi_params.immunity_inf_increase_factor,
            total_population_val=epi_params.total_population_val,
            saturation_matrix=epi_params.immunity_inf_saturation_constant,
            waning_factor=epi_params.waning_factor_inf
        )

        self.population_immunity_inf_V.change_in_current_val = get_change_in_immunity_V(
            immune_V_val=self.population_immunity_hosp_V.current_val,
            vaccine_doses=self.data_vaccine,
            index_date=self.current_day_counter,
            vaccine_in_effect_delay=epi_params.vaccine_in_effect_delay,
            immunity_increase_factor=epi_params.immunity_vaccine_increase_factor,
            waning_factor=epi_params.waning_factor_vaccine_inf
        )

    def update_transition_rates(self):

        epi_params = self.epi_params

        self.new_exposed.current_rate = get_new_exposed_rate(
            I_val=self.I.current_val,
            immunity_against_inf_H1=self.population_immunity_inf_H1.current_val,
            immunity_against_inf_H3=self.population_immunity_inf_H1.current_val,
            immunity_against_inf_V=self.population_immunity_inf_V.current_val,
            efficacy_against_inf=epi_params.efficacy_immunity_inf,
            total_population_val=epi_params.total_population_val,
            contact_matrix=epi_params.contact_matrix,
            beta=epi_params.beta)

        self.new_hosp.current_rate = get_new_hosp_rate(
            zeta=epi_params.zeta,
           mu=epi_params.mu,
            immunity_against_hosp_H1=self.population_immunity_hosp_H1.current_val,
            immunity_against_hosp_H3=self.population_immunity_hosp_H3.current_val,
            immunity_against_hosp_V=self.population_immunity_hosp_V.current_val,
            efficacy_against_hosp=epi_params.efficacy_immunity_hosp
        )

        self.new_dead.current_rate = get_new_dead_rate(
            pi=epi_params.pi,
            nu=epi_params.nu,
            immunity_against_hosp_H1=self.population_immunity_hosp_H1.current_val,
            immunity_against_hosp_H3=self.population_immunity_hosp_H3.current_val,
            immunity_against_hosp_V=self.population_immunity_hosp_V.current_val,
            efficacy_against_death=epi_params.efficacy_immunity_death
        )

        self.new_infected.current_rate = np.ones(shape=self.I.current_val.shape) * epi_params.sigma #np.expand_dims(epi_params.sigma, axis=0)

        self.new_recovered_home.current_rate = np.ones(shape=self.I.current_val.shape) * (1 - epi_params.mu) * epi_params.gamma #np.expand_dims((1 - epi_params.mu) * epi_params.gamma, axis=0)

        self.new_recovered_hosp.current_rate = np.ones(shape=self.I.current_val.shape) * (1 - epi_params.nu) * epi_params.gamma_hosp #np.expand_dims((1 - epi_params.nu) * epi_params.gamma_hosp, axis=0)

        self.new_susceptible.current_rate = np.ones(shape=self.I.current_val.shape) * epi_params.eta #np.expand_dims(epi_params.eta, axis=0)

    def read_vaccine_data(self, input_path):
        data_vaccine = pd.read_csv(input_path)
        self.data_vaccine = data_vaccine["Doses"].to_numpy()

start = time.time()

simple_model = ImmunoSEIRSModel()

simple_model.simulate_until_time_period(last_simulation_day=365)

print(time.time() - start)

PlotTools.create_basic_compartment_history_plot(simple_model)
