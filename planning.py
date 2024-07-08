import numpy as np
import matplotlib.pyplot as plt
import time


class EpiCompartment:

    def __init__(self, name, initial_count):
        self.name = name
        self.initial_value = initial_count

        self.current_count = initial_count  # current day's values
        self.previous_count = []  # previous day's values

        self.history_counts_list = []  # historical values

        self.__class__.varnames_dict["name"] = self

    varnames_dict = {}


class EpiParams:

    def __init__(self, starting_simulation_day=0):
        self.starting_simulation_day = starting_simulation_day


class SimulationParams:

    def __init__(self, timesteps_per_day):
        self.timesteps_per_day = timesteps_per_day


class ImmunityTracker:

    def __init__(self, starting_variant_prevance):
        self.current_variant_prevalence = starting_variant_prevance


# class InfluenzaModel:
#     pass


def create_basic_compartment_history_plot(sim_model):
    plt.figure(1)

    for name, epi_compartment in sim_model.name_to_epi_compartment_dict.items():
        plt.plot(epi_compartment.history_counts_list, label=name)
    plt.legend()
    plt.xlabel("Days")
    plt.xlim([0, 400])
    plt.ylabel("Number of individuals")
    plt.show()


def create_name_to_object_dict(set_of_objects):
    '''
    :param set_of_objects (set of objects -- each object must have the attribute "name"):
    :return: dictionary with each key equal to "name" attribute of object
        and each value returning the corresponding object itself
    '''

    name_to_object_dict = {}

    for object in set_of_objects:
        name_to_object_dict[object.name] = object

    return name_to_object_dict


class SEIRModel:

    def __init__(self, epi_params, simulation_params):

        self.epi_params = epi_params
        self.simulation_params = simulation_params

        self.current_simulation_day = epi_params.starting_simulation_day

        name_to_initial_count_dict = epi_params.name_to_initial_count_dict

        S = EpiCompartment("S", name_to_initial_count_dict["S"])
        E = EpiCompartment("E", name_to_initial_count_dict["E"])
        I = EpiCompartment("I", name_to_initial_count_dict["I"])
        R = EpiCompartment("R", name_to_initial_count_dict["R"])

        self.name_to_epi_compartment_dict = create_name_to_object_dict((S, E, I, R))

    def simulate_until_time_period(self, last_simulation_day):

        timesteps_per_day = self.simulation_params.timesteps_per_day
        name_to_epi_compartment_dict = self.name_to_epi_compartment_dict

        # last_simulation_day is inclusive endpoint
        for day in range(last_simulation_day + 1):
            self.simulate_next_day(timesteps_per_day)

            # breakpoint()

            for epi_compartment in name_to_epi_compartment_dict.values():
                epi_compartment.history_counts_list.append(epi_compartment.current_count)

    def simulate_next_day(self, timesteps_per_day):

        # Create short-hand for instance attribute access
        name_to_epi_compartment_dict = self.name_to_epi_compartment_dict
        epi_params = self.epi_params

        beta = epi_params.beta
        phi = epi_params.phi
        gamma = epi_params.gamma
        kappa = epi_params.kappa
        total_count = epi_params.total_count

        for epi_compartment in name_to_epi_compartment_dict.values():
            epi_compartment.previous_count = epi_compartment.current_count

        S = name_to_epi_compartment_dict["S"].current_count
        E = name_to_epi_compartment_dict["E"].current_count
        I = name_to_epi_compartment_dict["I"].current_count
        R = name_to_epi_compartment_dict["R"].current_count

        for timestep in range(timesteps_per_day):
            # Generate (possibly random) transition variables
            newly_exposed_count = (beta * phi * S * I / total_count) / timesteps_per_day
            newly_infected_count = kappa * E / timesteps_per_day
            newly_recovered_count = gamma * I / timesteps_per_day

            # Update counts in each compartment using discretized timestep
            S = S - newly_exposed_count
            E = E + newly_exposed_count - newly_infected_count
            I = I + newly_infected_count - newly_recovered_count
            R = R + newly_recovered_count

        name_to_epi_compartment_dict["S"].current_count = S
        name_to_epi_compartment_dict["E"].current_count = E
        name_to_epi_compartment_dict["I"].current_count = I
        name_to_epi_compartment_dict["R"].current_count = R

if __name__ == "__main__":
    start_time = time.time()
    # EpiParams and SimulationParams will eventually be read in from a file
    epi_params = EpiParams()
    epi_params.beta = 0.65
    epi_params.phi = 1
    epi_params.gamma = 0.2
    epi_params.kappa = 0.331
    epi_params.eta = 0
    epi_params.total_count = np.array([8500000])
    epi_params.name_to_initial_count_dict = {"S": np.array([8500000]),
                                             "E": np.array([0]),
                                             "I": np.array([20]),
                                             "R": np.array([0])}

    simulation_params = SimulationParams(timesteps_per_day=7)

    simple_model = SEIRModel(epi_params, simulation_params)

    print(time.time() - start_time)

    start_time = time.time()

    simple_model.simulate_until_time_period(last_simulation_day=365)

    print(time.time() - start_time)

    create_basic_compartment_history_plot(simple_model)


