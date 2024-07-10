import numpy as np
import matplotlib.pyplot as plt
import time

from collections import namedtuple

class EpiCompartment:

    def __init__(self, name, initial_val):
        self.name = name
        self.initial_value = initial_val

        self.current_day_val = initial_val  # current day's values
        self.current_timestep_val = initial_val
        self.previous_day_val = []  # previous day's values

        self.history_vals_list = []  # historical values

        self.__class__.varnames_dict[name] = self

    varnames_dict = {}


class EpiParams:

    def __init__(self, starting_simulation_day=0):
        self.starting_simulation_day = starting_simulation_day


class SimulationParams:

    def __init__(self, timesteps_per_day):
        self.timesteps_per_day = timesteps_per_day


class ImmunityTracker:

    def __init__(self, current_variant_prevalence):
        self.current_variant_prevalence = current_variant_prevalence


def create_basic_compartment_history_plot(sim_model):
    plt.figure(1)
    for name, compartment in sim_model.name_to_compartment_dict.items():
        plt.plot(compartment.history_vals_list, label=name)
    plt.legend()
    plt.xlabel("Days")
    plt.xlim([0, 400])
    plt.ylabel("Number of individuals")
    plt.show()


def create_name_to_object_dict(set_of_objects):
    '''
    :param set_of_objects (set of objects): each object must have the attribute "name":
    :return: (dict): with each key equal to "name" attribute of object
        and each value returning the corresponding object itself
    '''

    name_to_object_dict = {}

    for object in set_of_objects:
        name_to_object_dict[object.name] = object

    return name_to_object_dict


def create_name_to_val_dict(name_to_object_dict, val_name):
    '''
    :param name_to_object_dict (dict): each object must have attribute val_name
    :param val_name (str): attribute name corresponding to value that is saved in dictionary
    :return: (dict): with each key equal to "name" attribute of object
        and each value returning the object's attribute val_name
    '''

    name_to_val_dict = {}

    for name in name_to_object_dict.keys():
        name_to_val_dict[name] = name_to_object_dict[val_name]

    return name_to_object_dict


class BaseModel:

    def __init__(self, epi_params, simulation_params):

        self.epi_params = epi_params
        self.simulation_params = simulation_params

        self.current_simulation_day = epi_params.starting_simulation_day
        self.name_to_compartment_dict = EpiCompartment.varnames_dict

        for name, compartment in self.name_to_compartment_dict.items():
            setattr(self, name, compartment)

    def set_previous_day_vals_to_current_day_vals(self):
        for compartment in self.name_to_compartment_dict.values():
            compartment.previous_day_val = compartment.current_day_val

    def update_current_day_vals(self):
        for compartment in self.name_to_compartment_dict.values():
            compartment.current_day_val = compartment.current_timestep_val

    def update_history_vals_list(self):
        for compartment in self.name_to_compartment_dict.values():
            compartment.history_vals_list.append(compartment.current_day_val)

    def simulate_until_time_period(self, last_simulation_day):
        # last_simulation_day is inclusive endpoint
        for day in range(last_simulation_day + 1):
            self.simulate_next_day()

    def simulate_next_day(self):

        start_time = time.time()

        self.set_previous_day_vals_to_current_day_vals()

        self.simulate_next_timesteps(num_timesteps=self.simulation_params.timesteps_per_day)

        self.update_current_day_vals()
        self.update_history_vals_list()

        print(time.time() - start_time)

    def simulate_next_timesteps(self, num_timesteps):

        # Create short-hand for instance attribute access
        epi_params = self.epi_params

        beta = epi_params.beta
        phi = epi_params.phi
        gamma = epi_params.gamma
        kappa = epi_params.kappa
        eta = epi_params.eta
        total_population_val = epi_params.total_population_val

        S = self.S.current_day_val
        E = self.E.current_day_val
        I = self.I.current_day_val
        R = self.R.current_day_val

        for timestep in range(num_timesteps):
            # Generate (possibly random) transition variables
            newly_exposed_val = (beta * phi * S * I / total_population_val) / num_timesteps
            newly_infected_val = kappa * E / num_timesteps
            newly_recovered_val = gamma * I / num_timesteps
            newly_susceptible_val = eta * R / num_timesteps

            # Update counts in each compartment using discretized timestep
            S = S - newly_exposed_val + newly_susceptible_val
            E = E + newly_exposed_val - newly_infected_val
            I = I + newly_infected_val - newly_recovered_val
            R = R + newly_recovered_val - newly_susceptible_val

            self.S.current_timestep_val = S
            self.E.current_timestep_val = E
            self.I.current_timestep_val = I
            self.R.current_timestep_val = R


if __name__ == "__main__":
    start_time = time.time()

    EpiCompartment("S", np.array([8500000]))
    EpiCompartment("E", np.array([0]))
    EpiCompartment("I", np.array([20]))
    EpiCompartment("R", np.array([0]))

    # EpiParams and SimulationParams will eventually be read in from a file
    epi_params = EpiParams()
    epi_params.beta = 0.65
    epi_params.phi = 1
    epi_params.gamma = 0.2
    epi_params.kappa = 0.331
    epi_params.eta = 0.05
    epi_params.total_population_val = np.array([8500000])

    simulation_params = SimulationParams(timesteps_per_day=7)

    simple_model = BaseModel(epi_params,
                             simulation_params)

    print(time.time() - start_time)

    start_time = time.time()

    simple_model.simulate_until_time_period(last_simulation_day=365)

    print(time.time() - start_time)

    create_basic_compartment_history_plot(simple_model)


