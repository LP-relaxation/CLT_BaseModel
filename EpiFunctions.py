import numpy as np
import pandas as pd
##
# This module aims to keep a collection of functions for computing the rates in transitions and make them hashable
#
# absract rate function
def get_change_in_immunity_H1_abstract(arg):
    return get_change_in_immunity_H1(immune_H1_val=arg.population_immunity_hosp_H1.current_val,
            immune_H3_val=arg.population_immunity_hosp_H3.current_val,
            immune_V_val=arg.population_immunity_hosp_V.current_val,
            R_val=arg.R.current_val,
            immunity_increase_factor=arg.epi_params.immunity_hosp_increase_factor,
            total_population_val=arg.epi_params.total_population_val,
            saturation_matrix=arg.epi_params.immunity_hosp_saturation_constant,
            waning_factor=arg.epi_params.waning_factor_hosp)

# implementation of the rate function
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

def get_change_in_immunity_H3_abstract(arg):
    return get_change_in_immunity_H3(immune_H1_val=arg.population_immunity_hosp_H1.current_val,
                                     immune_H3_val=arg.population_immunity_hosp_H3.current_val,
                                     immune_V_val=arg.population_immunity_hosp_V.current_val,
                                     R_val=arg.R.current_val,
                                     immunity_increase_factor=arg.epi_params.immunity_hosp_increase_factor,
                                     total_population_val=arg.epi_params.total_population_val,
                                     saturation_matrix=arg.epi_params.immunity_hosp_saturation_constant,
                                     waning_factor=arg.epi_params.waning_factor_hosp)


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

def get_change_in_immunity_V_abstract(arg):
    return get_change_in_immunity_V(immune_V_val=arg.population_immunity_hosp_V.current_val,
            vaccine_doses=arg.data_vaccine,
            index_date=arg.current_day_counter,
            vaccine_in_effect_delay=arg.epi_params.vaccine_in_effect_delay,
            immunity_increase_factor=arg.epi_params.immunity_vaccine_increase_factor,
            waning_factor=arg.epi_params.waning_factor_vaccine_hosp)


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


def get_new_exposed_rate_abstract(arg):
    return get_new_exposed_rate(I_val=arg.I.current_val,
            immunity_against_inf_H1=arg.population_immunity_inf_H1.current_val,
            immunity_against_inf_H3=arg.population_immunity_inf_H3.current_val,
            immunity_against_inf_V=arg.population_immunity_inf_V.current_val,
            efficacy_against_inf=arg.epi_params.efficacy_immunity_inf,
            total_population_val=arg.epi_params.total_population_val,
            contact_matrix=arg.epi_params.contact_matrix,
            beta=arg.epi_params.beta)

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

def get_new_exposed_rate_no_immune_abstract(arg):
    return get_new_exposed_rate_no_immune(I_val=arg.I.current_val,
            total_population_val=arg.epi_params.total_population_val,
            contact_matrix=arg.epi_params.contact_matrix,
            beta=arg.epi_params.beta)

def get_new_exposed_rate_no_immune(I_val,
                             beta,
                             contact_matrix,
                             total_population_val):
    num_age_group, num_risk_group = I_val.shape
    res = np.zeros(shape=(num_age_group, num_risk_group))
    for age_index in range(num_age_group):
        for risk_index in range(num_risk_group):
            for age_index2 in range(num_age_group):
                for risk_index2 in range(num_risk_group):
                    res[age_index,risk_index] += beta * contact_matrix[age_index,age_index2] * I_val[age_index2,risk_index2] \
                                                 / total_population_val[age_index2,risk_index2]


    return res

def get_new_infected_rate_abstract(arg):
    return get_new_infected_rate(
            E_val=arg.E.current_val,
            sigma=arg.epi_params.sigma,
        )

def get_new_infected_rate(E_val, sigma):
    return np.ones(shape=E_val.shape) * sigma


def get_new_recovered_home_rate_abstract(arg):
    return get_new_recovered_home_rate(I_val=arg.I.current_val,
            mu=arg.epi_params.mu,
            gamma=arg.epi_params.gamma)


def get_new_recovered_home_rate(I_val, mu, gamma):
    return np.ones(shape=I_val.shape) * (1 - mu) * gamma


def get_new_recovered_hosp_rate_abstract(arg):
    return get_new_recovered_hosp_rate(H_val=arg.H.current_val,
            nu=arg.epi_params.nu,
            gamma_hosp=arg.epi_params.gamma_hosp)


def get_new_recovered_hosp_rate(H_val, nu, gamma_hosp):
    return np.ones(shape=H_val.shape) * (1 - nu) * gamma_hosp


def get_new_susceptible_rate_abstract(arg):
    return get_new_susceptible_rate(R_val=arg.R.current_val,
            eta=arg.epi_params.eta)

def get_new_susceptible_rate(R_val, eta):
    return np.ones(shape=R_val.shape) * eta

def get_new_hosp_rate_abstract(arg):
    return get_new_hosp_rate(zeta=arg.epi_params.zeta,
            mu=arg.epi_params.mu,
            immunity_against_hosp_H1=arg.population_immunity_hosp_H1.current_val,
            immunity_against_hosp_H3=arg.population_immunity_hosp_H3.current_val,
            immunity_against_hosp_V=arg.population_immunity_hosp_V.current_val,
            efficacy_against_hosp=arg.epi_params.efficacy_immunity_hosp)


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


def get_new_dead_rate_abstract(arg):
    return get_new_dead_rate(pi=arg.epi_params.pi,
            nu=arg.epi_params.nu,
            immunity_against_hosp_H1=arg.population_immunity_hosp_H1.current_val,
            immunity_against_hosp_H3=arg.population_immunity_hosp_H3.current_val,
            immunity_against_hosp_V=arg.population_immunity_hosp_V.current_val,
            efficacy_against_death=arg.epi_params.efficacy_immunity_death)


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



class EpiFunctions():
    def __init__(self):
        self.execute = {}

        # make functions hashable
        # rates in SEIR compartment
        self.execute["get_new_exposed_rate"] = get_new_exposed_rate_abstract
        self.execute["get_new_exposed_rate_no_immune"] = get_new_exposed_rate_no_immune_abstract
        self.execute["get_new_infected_rate"] = get_new_infected_rate_abstract
        self.execute["get_new_hosp_rate"] = get_new_hosp_rate_abstract
        self.execute["get_new_dead_rate"] = get_new_dead_rate_abstract
        self.execute["get_new_recovered_home_rate"] = get_new_recovered_home_rate_abstract
        self.execute["get_new_recovered_hosp_rate"] = get_new_recovered_hosp_rate_abstract
        self.execute["get_new_susceptible_rate"] = get_new_susceptible_rate_abstract
        # immunity compartment
        self.execute["get_change_in_immunity_H1"] = get_change_in_immunity_H1_abstract
        self.execute["get_change_in_immunity_H3"] = get_change_in_immunity_H3_abstract
        self.execute["get_change_in_immunity_V"] = get_change_in_immunity_V_abstract
