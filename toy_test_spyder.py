#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 13:05:34 2025

@author: rfp437
"""

# Imports

import clt_base as clt
import toy_immunity_model as imm

from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

from pprint import pprint
import copy

from importlib import reload
reload(imm)
del imm
import toy_immunity_model as imm


# %% Functions
def make_graph_set(model: imm.ToyImmunitySubpopModel):

    plt.clf()
    plt.figure(figsize=(8, 12))

    plt.subplot(4, 1, 1)
    plt.plot(np.asarray(model.compartments.S.history_vals_list), label="S")
    plt.plot(np.asarray(model.compartments.I.history_vals_list), label="I")
    plt.plot(np.asarray(model.compartments.R.history_vals_list), label="R")
    plt.title("Simulated compartment populations")
    plt.xlabel("Day")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.subplot(4, 1, 2)
    plt.plot(np.asarray(model.epi_metrics.M.history_vals_list), label="M")
    plt.title("Simulated immunity")
    plt.xlabel("Day")

    plt.subplot(4, 1, 3)
    plt.plot(np.asarray(model.transition_variables.S_to_I.history_vals_list))
    plt.title("Simulated incidence")
    plt.xlabel("Day")

    plt.subplot(4, 1, 4)
    plt.plot(np.asarray(model.transition_variables.R_to_S.history_vals_list))
    plt.title("Simulated R to S")
    plt.xlabel("Day")

    plt.tight_layout()

    plt.show()
    
def make_graph(model: imm.ToyImmunitySubpopModel):

    nb_subplots = 5
    fig, axs = plt.subplots(nb_subplots, 1, figsize = (6, 10))

    axs[0].plot(np.asarray(model.compartments.S.history_vals_list), label="S")
    axs[0].plot(np.asarray(model.compartments.I.history_vals_list), label="I")
    axs[0].plot(np.asarray(model.compartments.R.history_vals_list), label="R")
    axs[0].set_title("Simulated compartment populations")
    axs[0].set_xlabel("Day")
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

    axs[1].plot(np.asarray(model.epi_metrics.M.history_vals_list), label="M")
    axs[1].set_title("Simulated immunity")
    axs[1].set_xlabel("Day")

    axs[2].plot(np.asarray(model.transition_variables.S_to_I.history_vals_list))
    axs[2].set_title("Simulated incidence")
    axs[2].set_xlabel("Day")

    axs[3].plot(np.asarray(model.transition_variables.R_to_S.history_vals_list))
    axs[3].set_title("Simulated R to S")
    axs[3].set_xlabel("Day")

    axs[4].plot(np.asarray(model.schedules.absolute_humidity.history_vals_list))
    axs[4].set_title("Absolute humidity")
    axs[4].set_xlabel("Day")
    axs[4].set_ylim((0.000, None))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)

    plt.show()
    
    
def make_graph_comparison(
        model: imm.ToyImmunitySubpopModel,
        model2: imm.ToyImmunitySubpopModel):

    nb_subplots = 6
    fig, axs = plt.subplots(nb_subplots, 1, figsize = (6, 10))

    axs[0].plot(np.asarray(model.compartments.S.history_vals_list), label="S")
    axs[0].plot(np.asarray(model.compartments.I.history_vals_list), label="I")
    axs[0].plot(np.asarray(model.compartments.R.history_vals_list), label="R")
    axs[0].plot(np.asarray(model2.compartments.S.history_vals_list), label="S2", linestyle='dashed', color='blue')
    axs[0].plot(np.asarray(model2.compartments.I.history_vals_list), label="I2", linestyle='dashed', color='orange')
    axs[0].plot(np.asarray(model2.compartments.R.history_vals_list), label="R2", linestyle='dashed', color='green')
    axs[0].set_title("Simulated compartment populations")
    axs[0].set_xlabel("Day")
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

    axs[1].plot(np.asarray(model.epi_metrics.M.history_vals_list), label="M")
    axs[1].plot(np.asarray(model2.epi_metrics.M.history_vals_list), label="M2", linestyle='dashed', color='blue')
    axs[1].set_title("Simulated immunity")
    axs[1].set_xlabel("Day")

    axs[2].plot(np.asarray(model.transition_variables.S_to_I.history_vals_list))
    axs[2].plot(np.asarray(model2.transition_variables.S_to_I.history_vals_list), linestyle='dashed', color='blue')
    axs[2].set_title("Simulated incidence")
    axs[2].set_xlabel("Day")

    axs[3].plot(np.asarray(model.transition_variables.R_to_S.history_vals_list))
    axs[3].plot(np.asarray(model2.transition_variables.R_to_S.history_vals_list), linestyle='dashed', color='blue')
    axs[3].set_title("Simulated R to S")
    axs[3].set_xlabel("Day")

    axs[4].plot(np.asarray(model.schedules.absolute_humidity.history_vals_list))
    axs[4].plot(np.asarray(model2.schedules.absolute_humidity.history_vals_list), linestyle='dashed')
    axs[4].set_title("Absolute humidity")
    axs[4].set_xlabel("Day")
    axs[4].set_ylim((0.000, None))
    
    axs[5].plot(np.asarray(model.epi_metrics.Rt.history_vals_list))
    axs[5].plot(np.asarray(model2.epi_metrics.Rt.history_vals_list), linestyle='dashed')
    axs[5].set_title("Effective reproduction number")
    axs[5].set_xlabel("Day")
    axs[5].set_ylim((0.000, None))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.65)

    plt.show()

# %% Run

## Load parameters
# Setting up the components of the model

# Obtain path to folder with JSON input files
base_path = Path().resolve() / "toy_immunity_input_files"

# Get filepaths for initial values of state variables, fixed parameters, and config
compartments_epi_metrics_init_vals_filepath = base_path / "compartments_epi_metrics_init_vals.json"
params_filepath = base_path / "params.json"
config_filepath = base_path / "config.json"

compartments_epi_metrics_dict = clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath)
compartments_epi_metrics_dict = clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath)
params_dict = clt.load_json_new_dict(params_filepath)
config_dict = clt.load_json_new_dict(config_filepath)

config_dict["transition_type"] = "binomial_deterministic"
config_dict["timesteps_per_day"] = 1

bit_generator = np.random.MT19937(88888)


## Create model
# beta = 0.3
# model2.params.humidity_impact = 1.3

beta = 0.37
sim_duration = 1095 # 250 1095
M0 = 0.5


model = imm.ToyImmunitySubpopModel(compartments_epi_metrics_dict,
                                   params_dict,
                                   config_dict,
                                   np.random.Generator(bit_generator))

model.params.beta = beta
model.params.immune_gain = 10
model.params.immune_wane = 0.004
model.params.immune_saturation = 6
setattr(getattr(model.epi_metrics, "M"), "init_val", M0)

model.params.humidity_impact = 0.0

model.simulate_until_day(sim_duration)

model2 = imm.ToyImmunitySubpopModel(compartments_epi_metrics_dict,
                                   params_dict,
                                   config_dict,
                                   np.random.Generator(bit_generator))

model2.params.beta = beta
model2.params.immune_gain = 10
model2.params.immune_wane = 0.004
model2.params.immune_saturation = 6
setattr(getattr(model2.epi_metrics, "M"), "init_val", M0)

model2.params.humidity_impact = 0.7

model2.simulate_until_day(sim_duration)



# make_graph_set(model)
# make_graph(model)

make_graph_comparison(model, model2)

model.state
model.params
