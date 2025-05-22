##############################################################################
############## HOW-TO USE THIS FILE (FOR LAB FOLKS) ##########################
###############################################################################

# Try to discover sensible parameter value settings "by hand" --
#   modify the parameters in the section named
#   "TEST OUT DIFFERENT PARAMETER VALUES" and then run this script
#   to generate plots! :)


###########################################################
######################## IMPORTS ##########################
###########################################################

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pathlib import Path
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import copy

# Import city-level transmission base components module
import clt_base as clt

# Import flu model module, which contains customized subclasses
import flu_components as flu

###########################################################
################# READ INPUT FILES ########################
###########################################################

# Texas simplified age groups
# 0-4, 5-49, 50-64, 65+
# Combined 5-17 and 18-49
# https://data.census.gov/profile/Texas?g=040XX00US48

# Obtain path to folder with JSON input files
base_path = Path(__file__).parent / "flu_texas_1subpop"

# Setup time ~0.02 seconds

# Get filepaths for initial values of compartments and epi metrics, fixed parameters,
#   configuration, and travel proportions
compartments_epi_metrics_init_vals_filepath = \
    base_path / "texas_compartments_epi_metrics_init_vals.json"
params_filepath = base_path / "texas_common_params.json"
config_filepath = base_path / "config.json"

# Get filepaths for school-work calendar CSV
calendar_filepath = base_path / "school_work_calendar.csv"

# Read in files as dictionaries and dataframes
# Note that we can also create these dictionaries directly
#   rather than reading from a predefined input data file
compartments_epi_metrics_dict = \
    clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath)
params_dict = clt.load_json_new_dict(params_filepath)
config_dict = clt.load_json_new_dict(config_filepath)

calendar_df = pd.read_csv(calendar_filepath, index_col=0)

###########################################################
################# BUILD SUBPOP MODEL ######################
###########################################################

# Create two independent bit generators
bit_generator = np.random.MT19937(88888)
jumped_bit_generator = bit_generator.jumped(1)

model = flu.FluSubpopModel(compartments_epi_metrics_dict,
                           params_dict,
                           config_dict,
                           calendar_df,
                           np.random.Generator(jumped_bit_generator))

######################################################################
################# ASIDE: DOUBLE-CHECKING RATES #######################
######################################################################

# Recall that we need to adjust the IHR and in-hospital mortality
#   rate to be "rate-adjusted" to recapitulate the correct
#   proportions under the competing clock model
#   (see mathematical formulation of CLT model for details)

non_rate_adjusted_IHR = np.array([0.007, 0.003, 0.006, 0.01, 0.09])

rate_adjusted_IHR = non_rate_adjusted_IHR * model.params.IS_to_R_rate / \
                    (model.params.IS_to_H_rate - non_rate_adjusted_IHR * (
                            model.params.IS_to_H_rate - model.params.IS_to_R_rate))

# print(rate_adjusted_IHR)

non_rate_adjusted_mortality_rate = np.array([0.00009, 0.000027, 0.000165, 0.00063, 0.0073])

rate_adjusted_mortality_rate = non_rate_adjusted_mortality_rate * model.params.H_to_R_rate / \
                               (model.params.H_to_D_rate - non_rate_adjusted_mortality_rate *
                                (model.params.H_to_D_rate - model.params.H_to_R_rate))

# print(rate_adjusted_mortality_rate)

#############################################################################
################# TEST OUT DIFFERENT PARAMETER VALUES #######################
#############################################################################

# Recall that any model parameter or initial value that is
#   adjusted here will override the value specified in
#   `texas_common_params.json` or `texas_compartments_epi_metrics_init_vals.json`
#   -- if a parameter or initial value is not overriden, then the values
#   are taken from the aforementioned `JSON` files

array_of_1s = np.full((5, 1), 1)
array_of_5s = np.full((5, 1), 5)
array_of_10s = np.full((5, 1), 10)
array_of_20s = np.full((5, 1), 20)

# model.params.beta_baseline = 0.0232592 # Warning: beta_baseline is VERY touchy
# model.params.beta_baseline = 2.244e-02

model.params.beta_baseline = 0.021486

model.compartments.E.init_val = copy.copy(array_of_20s)
model.compartments.IS.init_val = copy.copy(array_of_10s)
model.compartments.IP.init_val = copy.copy(array_of_1s)
model.compartments.IA.init_val = copy.copy(array_of_1s)
model.compartments.H.init_val = copy.copy(array_of_1s)

model.params.immune_saturation = 2

model.params.inf_immune_wane = 0.001
model.params.hosp_immune_wane = 0.004
model.epi_metrics.pop_immunity_inf.init_val = 0
model.epi_metrics.pop_immunity_hosp.init_val = 0

model.params.hosp_risk_reduce = 1
model.params.inf_risk_reduce = 1
model.params.death_risk_reduce = 1

# Misc musings to self...
# For week 36, there are 122 total historical hospitalizations
# Adjust for IHR -- could also try weighting by population?
# Could allocate 122 historical hospitalizations to initial model H across
#   age groups using RELATIVE IHR (so, IHR / sum of IHRs)?

######################################################################
################# GET HISTORICAL HOSPITAL ADMITS #####################
######################################################################

total_texas_pop = 30500000

base_path = Path(__file__).parent / "flu_texas_1subpop"
historical_hosp_admits_weekly = pd.read_csv(base_path / "texas_flu_hosp_rate_20232024" / "data.csv")["flu_rate"] * \
                                total_texas_pop / int(1e5)
historical_hosp_admits_weekly = np.asarray(historical_hosp_admits_weekly)

num_historical_weeks = len(historical_hosp_admits_weekly)

##########################################################################
################# PLOTTING AND CALIBRATION CONSTANTS #####################
##########################################################################

weeks_offset = 20
calibration_period_length_weeks = 40

historical_hosp_admits_weekly_subset = historical_hosp_admits_weekly[
                                       weeks_offset:weeks_offset + calibration_period_length_weeks]


###################################################
################# CALIBRATION #####################
###################################################

# NOTE: as mentioned in 4/19 meeting, this does not work --
#   optimization gets stuck on initial point
# TODO: Remy had some good suggestions for different scipy
#   functions to use -- try those

# Quick notes to self
# Compiling did not seem to speed up simulation time significantly
# `scipy.optimize.least_squares` always gets stuck at the initial
#   starting point
# Seems like `curve_fit` also gets stuck at
#   initial point when there are 3 variables to fit:
#   beta_baseline, inf_immune_wane (same as hosp_immune_wane),
#   and init_immunity_level

def compute_hospital_admits(x):
    model.reset_simulation()

    model.params.beta_baseline = x[0]
    model.params.R_to_S_rate = x[1]

    # model.inf_immune_wane = waning_rate
    # model.hosp_immune_wane = waning_rate
    # model.params.pop_immunity_inf = np.full((5, 1), init_immunity_level)
    # model.params.pop_immunity_hosp = np.full((5, 1), init_immunity_level)

    model.simulate_until_day(7 * calibration_period_length_weeks)

    model_hosp_admits = np.asarray(model.transition_variables.IS_to_H.history_vals_list).sum(axis=(1, 2))
    model_hosp_admits_weekly_subset = model_hosp_admits.reshape(-1, 49).sum(axis=1)[:calibration_period_length_weeks]

    epsilon = 1e-8  # small value to avoid log(0)
    loss = np.sum(np.square(
        np.log(historical_hosp_admits_weekly_subset + epsilon) -
        np.log(model_hosp_admits_weekly_subset + epsilon)
    ))

    return loss


# compute_hospital_admits(0.0232592)


run_optimization = False

if run_optimization:
    start_time = time.time()

    x_fit = scipy.optimize.minimize(compute_hospital_admits, method="Nelder-Mead", x0=[0.02161, 0.02])

    beta_baseline_fit = x_fit.x[0]
    R_to_S_rate_fit = x_fit.x[1]

    print(x_fit)
    print(time.time() - start_time)

    breakpoint()

    print("Beta baseline fit is " + str(beta_baseline_fit))

    model.reset_simulation()
    model.params.beta_baseline = beta_baseline_fit
    model.params.R_to_S_rate = R_to_S_rate_fit

    # model.params.pop_immunity_inf = np.full((5, 1), init_immunity_level_fit)
    # model.params.pop_immunity_hosp = np.full((5, 1), init_immunity_level_fit)
    # model.params.hosp_immune_wane = waning_rate_fit
    # model.params.inf_immune_wane = waning_rate_fit

################################################
################# SIMULATION ###################
################################################

model.simulate_until_day(7 * (weeks_offset + calibration_period_length_weeks))

model_hosp_admits = np.asarray(model.transition_variables.IS_to_H.history_vals_list).sum(axis=(1, 2))
model_hosp_admits_weekly = model_hosp_admits.reshape(-1, 49).sum(axis=1)
model_hosp_admits_weekly_subset = model_hosp_admits_weekly[:calibration_period_length_weeks]

model_new_infections = np.asarray(model.transition_variables.E_to_IA.history_vals_list).sum(axis=(1, 2)) + \
                       np.asarray(model.transition_variables.E_to_IP.history_vals_list).sum(axis=(1, 2))

model_new_infections_weekly = model_new_infections.reshape(-1, 49).sum(axis=1)

x_positions = np.arange(calibration_period_length_weeks)

################################################
################# PLOTTING #####################
################################################

print("Total symptomatic " + str(np.sum(model.transition_variables.E_to_IP.history_vals_list)))
print("Total hospital admits " + str(np.sum(np.asarray(model.transition_variables.IS_to_H.history_vals_list))))
print("Total recovered-to-susceptible " + str(np.sum(model.transition_variables.R_to_S.history_vals_list)))

plt.figure(figsize=(10, 12))

plt.subplot(4, 2, 1)  # (rows, columns, index)
plt.plot(x_positions, np.asarray(historical_hosp_admits_weekly_subset),
         marker='o', linestyle='-',
         label="Historical hosp admits")
plt.plot(x_positions, model_hosp_admits_weekly_subset, label="Simulated hospital admits")
plt.title("Hospital admits")
plt.xlabel("Week")
plt.legend()

infected = np.asarray(model.compartments.IS.history_vals_list).sum(axis=(1, 2)) + \
           np.asarray(model.compartments.IP.history_vals_list).sum(axis=(1, 2)) + \
           np.asarray(model.compartments.IA.history_vals_list).sum(axis=(1, 2)) + \
           np.asarray(model.compartments.H.history_vals_list).sum(axis=(1, 2))

plt.subplot(4, 2, 2)
plt.plot(np.asarray(model.compartments.H.history_vals_list).sum(axis=(1, 2)))
plt.xlabel("Day")
plt.title("Simulated hospitalized (census)")

plt.subplot(4, 2, 3)
plt.plot(np.asarray(model.epi_metrics.pop_immunity_inf.history_vals_list)[:, 0, 0], label="0-4 y.o.")
plt.plot(np.asarray(model.epi_metrics.pop_immunity_inf.history_vals_list)[:, 1, 0], label="5-17 y.o.")
plt.plot(np.asarray(model.epi_metrics.pop_immunity_inf.history_vals_list)[:, 2, 0], label="18-49 y.o.")
plt.plot(np.asarray(model.epi_metrics.pop_immunity_inf.history_vals_list)[:, 3, 0], label="50-64 y.o.")
plt.plot(np.asarray(model.epi_metrics.pop_immunity_inf.history_vals_list)[:, 4, 0], label="65+ y.o.")
plt.title("Simulated immunity by age group")
plt.xlabel("Day")
plt.legend()

plt.subplot(4, 2, 4)
plt.plot(np.asarray(model.compartments.S.history_vals_list).sum(axis=(1, 2)), label="S")
plt.plot(np.asarray(model.compartments.R.history_vals_list).sum(axis=(1, 2)), label="R")
plt.xlabel("Day")
plt.title("Simulated susceptible and recovered")
plt.legend()

plt.subplot(4, 2, 5)
plt.plot(np.asarray(model.compartments.IS.history_vals_list).sum(axis=(1, 2)), label="IS")
plt.plot(np.asarray(model.compartments.IP.history_vals_list).sum(axis=(1, 2)), label="IP")
plt.plot(np.asarray(model.compartments.IA.history_vals_list).sum(axis=(1, 2)), label="IA")
plt.title("Simulated infected (excluding hospitalized)")
plt.xlabel("Day")
plt.legend()

plt.subplot(4, 2, 6)
plt.plot(np.asarray(model.compartments.D.history_vals_list).sum(axis=(1, 2)), label="D")
plt.title("Simulated cumulative deaths")
plt.xlabel("Day")
plt.legend()

plt.tight_layout()
plt.show()

# breakpoint()
