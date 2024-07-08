import numpy as np


class EpiCompartment:

    def __init__(self, name, initial_population):
        self.name = name
        self.initial_value = initial_population

        self.current_population = initial_population  # current day's values
        self.previous_population = []  # previous day's values

        self.history_populations_list = []  # historical values

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


class InfluenzaModel:
    pass


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

        S = EpiCompartment("S", np.array([8500000]))
        E = EpiCompartment("E", np.array([0]))
        I = EpiCompartment("I", np.array([20]))
        R = EpiCompartment("R", np.array([0]))

        self.name_to_epi_compartment_dict = create_name_to_object_dict((S, E, I, R))

    def simulate_until_time_period(self, last_simulation_day):

        timesteps_per_day = self.simulation_params.timesteps_per_day

    def simulate_next_day(self, timesteps_per_day):

        # Create short-hand for instance attribute access
        name_to_epi_compartment_dict = self.name_to_epi_compartment_dict
        epi_params = self.epi_params

        beta = epi_params.beta
        phi = epi_params.phi


        for epi_compartment in name_to_epi_compartment_dict.keys():
            epi_compartment.previous_population = epi_compartment.current_population

        for timestep in timesteps_per_day:

            S = name_to_epi_compartment_dict["S"].current_population
            E = name_to_epi_compartment_dict["E"].current_population
            I = name_to_epi_compartment_dict["I"].current_population
            R = name_to_epi_compartment_dict["R"].current_population

            # Generate (possibly random) transition variables
            newly_exposed_count =

            # Update populations in each compartment using discretized timestep



if __name__ == "__main__":
    # EpiParams and SimulationParams will eventually be read in from a file
    epi_params = EpiParams
    epi_params.phi = 1
    epi_params.gamma = 0.2
    epi_params.eta = 0
    epi_params.total_population = np.array([8500000])
    epi_params.name_to_initial_population_dict = {"S": np.array([8500000]),
                                                  "E": np.array([0]),
                                                  "I": np.array([20]),
                                                  "R": np.array([0])}

    simulation_params = SimulationParams(timesteps_per_day=7)

    simple_model = SEIRModel(epi_params, simulation_params)

    simple_model.simulate_until_time_period(last_simulation_day=365)
