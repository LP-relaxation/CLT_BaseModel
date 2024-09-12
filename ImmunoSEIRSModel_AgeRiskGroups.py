from BaseModel import BaseModel, SimulationParams
from EpiFunctions import EpiFunctions
import PlotTools
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import pandas as pd
import pdb

from collections import namedtuple

# global values
#VARIANT_DIM = 2
#VACCINE_DIM = 1

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

    def __init__(self, epi_params_json_filename,
                 simulation_params_json_filename,
                 epi_compartments_json_filename,
                 vaccine_filename,
                 RNG_seed=np.random.SeedSequence()):

        super().__init__(RNG_seed)

        self.add_epi_params_from_json(epi_params_json_filename)
        self.add_simulation_params_from_json(simulation_params_json_filename)
        self.read_vaccine_data(vaccine_filename)
        #breakpoint()
        # TODO: Put the following into the add_epi_params_from_json
        self.epi_params.immunity_hosp_saturation_constant = np.array(self.epi_params.immunity_hosp_saturation_constant)
        self.epi_params.immunity_inf_saturation_constant = np.array(self.epi_params.immunity_inf_saturation_constant)
        self.epi_params.total_population_val = np.array(self.epi_params.total_population_val)
        self.epi_params.contact_matrix = np.array(self.epi_params.contact_matrix)

        # initialize the functions
        self.function_base = EpiFunctions()

        self.add_epi_compartments_from_json(epi_compartments_json_filename)
        self.add_state_variables_from_json(epi_compartments_json_filename)
        breakpoint()

    def update_change_in_state_variables(self): # for immunity variable
        for state_variable_name in self.name_to_state_variable_dict:
            self.name_to_state_variable_dict[state_variable_name].change_in_current_val = self.function_base.execute[
            self.name_to_state_variable_dict[state_variable_name].state_update_function](self)


    def update_transition_rates(self):
        for epiCompartment_name in self.name_to_epi_compartment_dict:
            for transition_variable_name in self.name_to_epi_compartment_dict[epiCompartment_name].outflow_transition_variable_names_list:
                self.name_to_transition_variable_dict[transition_variable_name].current_rate = self.function_base.execute[
                    self.name_to_epi_compartment_dict[epiCompartment_name].outgoing_transition_rate_function[transition_variable_name]](self)


    def read_vaccine_data(self, input_path):
        data_vaccine = pd.read_csv(input_path)
        self.data_vaccine = data_vaccine["Doses"].to_numpy()

start = time.time()

simple_model = ImmunoSEIRSModel(epi_params_json_filename = "ImmunoSEIRS_EpiParams.json",
                                simulation_params_json_filename = "ImmunoSEIRS_SimulationParams.json",
                                epi_compartments_json_filename = "ImmunoSEIRS_CompartmentalModel_v2.json",
                                vaccine_filename = "vaccine.csv",
                                RNG_seed=np.random.SeedSequence()
                                )

breakpoint()
simple_model.simulate_until_time_period(last_simulation_day=365)

print(time.time() - start)

PlotTools.create_basic_compartment_history_plot(simple_model)
