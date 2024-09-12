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
                                epi_compartments_json_filename = "ImmunoSEIRS_CompartmentalModel.json",
                                vaccine_filename = "vaccine.csv",
                                RNG_seed=np.random.SeedSequence()
                                )

breakpoint()
simple_model.simulate_until_time_period(last_simulation_day=365)

print(time.time() - start)

PlotTools.create_basic_compartment_history_plot(simple_model)
