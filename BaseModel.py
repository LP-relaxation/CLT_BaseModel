import numpy as np
import json
import copy
from abc import ABC, abstractmethod


class TransitionVariableGroup:
    """
    Container for TransitionVariable objects to handle joint sampling,
    when there are multiple outflows from a single compartment

    For example, if all outflows of compartment H are: R and D,
    i.e. from the Hospital compartment, people either go to Recovered
    or Dead, a TransitionVariableGroup that holds both R and D handles
    the correct correlation structure between R and D
    """

    def __init__(self,
                 name,
                 base_count_epi_compartment,
                 timesteps_per_day,
                 RNG,
                 transition_type,
                 transition_variables_list,
                 num_age_groups,
                 num_risk_groups):
        """
        :parameter name: string,
            name of transition variable group
        :parameter base_count_epi_compartment: EpiCompartment object,
            corresponds to common EpiCompartment that transitions
            variables in transition_variable_list outflow from
        :parameter timesteps_per_day: positive int,
            number of discretized timesteps per day
        :parameter RNG: numpy Generator object,
            used to generate random variables
        :parameter transition_type: TransitionType Enum instance,
            Determines which method to use to compute realization,
            corresponding to get_binomial_realization,
            get_binomial_taylor_approx_realization, etc...
            See methods for rigorous explanation.
        :parameter transition_variables_list: array-like of TransitionVariable objects,
            holds all TransitionVariable objects that outflow from
            base_count_epi_compartment
        :parameter num_age_groups: positive int,
            number of age groups (rows)
        :parameter num_risk_groups: positive int,
            number of risk groups (columns)
        """

        self.name = name
        self.base_count_epi_compartment = base_count_epi_compartment
        self.timesteps_per_day = timesteps_per_day
        self.RNG = RNG
        self.transition_variables_list = transition_variables_list

        self.num_age_groups = num_age_groups
        self.num_risk_groups = num_risk_groups

        self.current_realizations_list = []

        self.assign_transition_type(transition_type)

    @property
    def transition_type(self):
        return self._transition_type

    def assign_transition_type(self, transition_type):
        """
        Updates transition_type and updates get_joint_realization
        method according to transition_type.

        :parameter transition_type: str,
            Supported values are in {"multinomial", "multinomial_taylor_approx", "poisson",
            "multinomial_deterministic", "multinomial_taylor_approx_deterministic",
            "poisson_deterministic"}. Determines which method to use to compute
            realization, corresponding to get_binomial_realization,
            get_binomial_taylor_approx_realization, etc... See methods for
            rigorous explanation.
        """

        # If marginal transition type is binomial, then
        #   joint transition type is multinomial
        transition_type = transition_type.replace("binomial", "multinomial")

        self._transition_type = transition_type
        self.assign_get_joint_realization_function(transition_type)

    def assign_get_joint_realization_function(self, transition_type):
        """
        Updates get_joint_realization method according to transition_type.

        Overrides get_joint_realization so that joint transitions
        are computed according to the desired transition_type.

        :parameter transition_type: str,
            Supported values are in {"multinomial", "multinomial_taylor_approx", "poisson",
            "multinomial_deterministic", "multinomial_taylor_approx_deterministic",
            "poisson_deterministic"}. Determines which method to use to compute
            realization, corresponding to get_binomial_realization,
            get_binomial_taylor_approx_realization, etc... See methods for
            rigorous explanation.
        """

        self.get_joint_realization = getattr(self, "get_" + transition_type + "_realization")

    def get_total_rate(self):
        """
        Return the age-risk-specific total transition rate,
        which is the sum of the current rate of each transition variable
        in this transition variable group

        Used to properly scale multinomial probabilities vector so
        that elements sum to 1

        :return: array-like of positive numbers,
            size equal to number of age groups x number of risk groups,
            sum of current rates of transition variables in
            transition variable group
        """

        # axis 0: corresponds to outgoing transition variable
        # axis 1: corresponds to age groups
        # axis 2: corresponds to risk groups
        # --> summing over axis 0 gives us the total rate for each age-risk group
        return np.sum(self.get_current_rates_array(), axis=0)

    def get_probabilities_array(self):
        """
        Returns an array of probabilities used for joint binomial
        (multinomial) transitions (get_multinomial_realization method)

        :return: array-like of positive numbers,
            size equal to (length of outgoing transition variables list + 1)
            x number of age groups x number of risk groups --
            note the "+1" corresponds to the multinomial outcome of staying
            in the same epi compartment (not transitioning to any outgoing
            epi compartment)
        """

        total_rate = self.get_total_rate()

        total_outgoing_probability = approx_binomial_probability_from_rate(total_rate, 1/self.timesteps_per_day)

        # Create probabilities_list, where element i corresponds to the
        #   transition variable i's current rate divided by the total rate,
        #   multiplied by the total outgoing probability
        # This generates the probabilities array that parameterizes the
        #   multinomial distribution
        probabilities_list = []

        for transition_variable in self.transition_variables_list:
            probabilities_list.append((transition_variable.current_rate / total_rate) * total_outgoing_probability)

        # Append the probability that a person stays in the compartment
        probabilities_list.append(1 - total_outgoing_probability)

        return np.asarray(probabilities_list)

    def get_current_rates_array(self):
        """
        Returns an array of current rates of transition variables in
        self.transition_variables_list -- ith element in array
        corresponds to current rate of ith transition variable

        :return: array-like of positive numbers,
            size equal to length of outgoing transition variables list
            x number of age groups x number of risk groups
        """

        current_rates_list = [tvar.current_rate for tvar in self.transition_variables_list]

        return np.asarray(current_rates_list)

    def get_joint_realization(self):
        pass

    def get_multinomial_realization(self):
        """
        Returns an array of transition realizations (number transitioning
        to outgoing compartments) sampled from multinomial distribution

        :return: array-like of positive numbers,
            size equal to (length of outgoing transition variables list + 1)
            x number of age groups x number of risk groups --
            note the "+1" corresponds to the multinomial outcome of staying
            in the same epi compartment (not transitioning to any outgoing
            epi compartment)
        """

        probabilities_array = self.get_probabilities_array()

        num_outflows = len(self.transition_variables_list)

        # We use num_outflows + 1 because for the multinomial distribution we explicitly model
        #   the number who stay/remain in the compartment
        realizations_array = np.zeros((num_outflows + 1, self.num_age_groups, self.num_risk_groups))

        for age_group in range(self.num_age_groups):
            for risk_group in range(self.num_risk_groups):
                realizations_array[:, age_group, risk_group] = self.RNG.multinomial(
                    np.asarray(self.base_count_epi_compartment.current_val[age_group,risk_group], dtype=int),
                    probabilities_array[:,age_group,risk_group])

        return realizations_array

    def get_multinomial_taylor_approx_realization(self):
        """
        Returns an array of transition realizations (number transitioning
        to outgoing compartments) sampled from multinomial distribution
        using Taylor Series approximation for probability parameter

        :return: array-like of positive numbers,
            size equal to (length of outgoing transition variables list + 1)
            x number of age groups x number of risk groups --
            note the "+1" corresponds to the multinomial outcome of staying
            in the same epi compartment (not transitioning to any outgoing
            epi compartment)
        """

        num_outflows = len(self.transition_variables_list)

        timesteps_per_day = self.timesteps_per_day

        current_rates_array = self.get_current_rates_array()

        total_rate = self.get_total_rate()

        # Multiply current rates array by length of time interval (1 / timesteps_per_day)
        # Also append additional value corresponding to probability of
        #   remaining in current epi compartment (not transitioning at all)
        # Note: vstack function here works better than append function because append
        #   automatically flattens the resulting array, resulting in dimension issues
        current_scaled_rates_array = np.vstack((current_rates_array / timesteps_per_day,
                                               np.expand_dims(1 - total_rate / timesteps_per_day, axis=0)))

        # We use num_outflows + 1 because for the multinomial distribution we explicitly model
        #   the number who stay/remain in the compartment
        realizations_array = np.zeros((num_outflows + 1, self.num_age_groups, self.num_risk_groups))

        for age_group in range(self.num_age_groups):
            for risk_group in range(self.num_risk_groups):
                realizations_array[:, age_group, risk_group] = self.RNG.multinomial(
                    np.asarray(self.base_count_epi_compartment.current_val[age_group,risk_group], dtype=int),
                    current_scaled_rates_array[:,age_group,risk_group])

        return realizations_array

    def get_poisson_realization(self):
        """
        Returns an array of transition realizations (number transitioning
        to outgoing compartments) sampled from Poisson distribution

        :return: array-like of positive numbers,
            size equal to length of outgoing transition variables list
            x number of age groups x number of risk groups
        """

        timesteps_per_day = self.timesteps_per_day

        num_outflows = len(self.transition_variables_list)

        realizations_array = np.zeros((num_outflows, self.num_age_groups, self.num_risk_groups))

        transition_variables_list = self.transition_variables_list

        for age_group in range(self.num_age_groups):
            for risk_group in range(self.num_risk_groups):
                for outflow_ix in range(num_outflows):
                    realizations_array[outflow_ix, age_group, risk_group] = self.RNG.poisson(
                        self.base_count_epi_compartment.current_val[age_group,risk_group] *
                        transition_variables_list[outflow_ix].current_rate[age_group, risk_group] * 1/timesteps_per_day)

        return realizations_array

    def get_multinomial_deterministic_realization(self):
        """
        Deterministic counterpart to get_multinomial_realization --
        uses mean (n x p, i.e. total counts x probability array) as realization
        rather than randomly sampling

        :return: array-like of positive numbers,
            size equal to (length of outgoing transition variables list + 1)
            x number of age groups x number of risk groups --
            note the "+1" corresponds to the multinomial outcome of staying
            in the same epi compartment (not transitioning to any outgoing
            epi compartment)
        """

        probabilities_array = self.get_probabilities_array()
        return self.base_count_epi_compartment.current_val * probabilities_array

    def get_multinomial_taylor_approx_deterministic_realization(self):
        """
        Deterministic counterpart to get_multinomial_taylor_approx_realization --
        uses mean (n x p, i.e. total counts x probability array) as realization
        rather than randomly sampling

        :return: array-like of positive numbers,
            size equal to (length of outgoing transition variables list + 1)
            x number of age groups x number of risk groups --
            note the "+1" corresponds to the multinomial outcome of staying
            in the same epi compartment (not transitioning to any outgoing
            epi compartment)
        """

        current_rates_array = self.get_current_rates_array()
        return self.base_count_epi_compartment.current_val * current_rates_array / self.timesteps_per_day

    def get_poisson_deterministic_realization(self):
        """
        Deterministic counterpart to get_poisson_realization --
        uses mean (rate array) as realization rather than randomly sampling

        :return: array-like of positive numbers,
            size equal to length of outgoing transition variables list
            x number of age groups x number of risk groups
        """

        return self.get_current_rates_array() / self.timesteps_per_day

    def reset(self):
        self.current_realizations_list = []

    def update_transition_variable_realizations(self):
        """
        Updates current_realization attribute on all
        TransitionVariable instances contained in this
        transition variable group
        """

        # Since the ith element in probabilities_array corresponds to the ith transition variable
        #   in transition_variables_list, the ith element in multinomial_realizations_list
        #   also corresponds to the ith transition variable in transition_variables_list
        # Update the current realization of the transition variables contained in this group
        for ix in range(len(self.transition_variables_list)):
            self.transition_variables_list[ix].current_realization = self.current_realizations_list[ix,:,:]


class TransitionVariable:

    def __init__(self,
                 name,
                 base_count_epi_compartment,
                 timesteps_per_day,
                 RNG,
                 transition_type,
                 is_jointly_distributed,
                 num_age_groups,
                 num_risk_groups):
        """
        :parameter name: string,
            name of TransitionVariable
        :parameter base_count_epi_compartment: EpiCompartment object,
            compartment from which TransitionVariable flows
        :parameter timesteps_per_day: positive int,
            number of discretized timesteps per day
        :parameter RNG: numpy Generator object,
            used to generate random variables
        :parameter transition_type: str,
            Supported values are in {"binomial", "binomial_taylor_approx", "poisson",
            "binomial_deterministic", "binomial_taylor_approx_deterministic",
            "poisson_deterministic"}. Determines which method to use to compute
            realization, corresponding to get_binomial_realization,
            get_binomial_taylor_approx_realization, etc... See methods for
            rigorous explanation.
        :parameter is_jointly_distributed: Boolean,
            True if there are multiple TransitionVariable outflows
            from the same base_count_epi_compartment and
            random sampling must be handled jointly (then this
            TransitionVariable instance is also contained in a
            TransitionVariableGroup). False otherwise, in which case
            random sampling is marginal.
        :parameter num_age_groups: positive int,
            number of age groups (rows)
        :parameter num_risk_groups: positive int,
            number of risk groups (columns)
        """

        self.name = name
        self.base_count_epi_compartment = base_count_epi_compartment
        self.timesteps_per_day = timesteps_per_day
        self.RNG = RNG
        self.is_jointly_distributed = is_jointly_distributed

        self.num_age_groups = num_age_groups
        self.num_risk_groups = num_risk_groups

        self.current_realization = None
        self.current_rate = None

        self.assign_transition_type(transition_type)

    @property
    def transition_type(self):
        return self._transition_type

    def assign_get_realization_function(self, transition_type):
        if self.is_jointly_distributed:
            pass
        else:
            self.get_realization = getattr(self, "get_" + transition_type + "_realization")

    def assign_transition_type(self, transition_type):
        self._transition_type = transition_type
        self.assign_get_realization_function(transition_type)

    def get_binomial_realization(self):
        return self.RNG.binomial(n=np.asarray(self.base_count, dtype=int),
                            p=approx_binomial_probability_from_rate(self.current_rate, 1 / self.timesteps_per_day))

    def get_binomial_taylor_approx_realization(self):
        return self.RNG.binomial(n=np.asarray(self.base_count, dtype=int),
                                 p= self.current_rate * (1/self.timesteps_per_day))

    def get_poisson_realization(self):
        return self.RNG.poisson(self.base_count * self.current_rate * (1/self.timesteps_per_day))

    def get_binomial_deterministic_realization(self):
        return np.asarray(self.base_count *
                        approx_binomial_probability_from_rate(self.current_rate, 1/self.timesteps_per_day),
                        dtype=int)

    def get_binomial_taylor_approx_deterministic_realization(self):
        return np.asarray(self.base_count * self.current_rate * (1/self.timesteps_per_day), dtype=int)

    def get_poisson_deterministic_realization(self):
        return np.asarray(self.base_count * self.current_rate * (1/self.timesteps_per_day), dtype=int)

    @property
    def base_count(self):
        return self.base_count_epi_compartment.current_val

    def reset(self):
        self.current_realization = None
        self.current_rate = None


class EpiParams:
    """
    Container for epidemiological parameters.
    """

    def __init__(self):
        pass


class EpiCompartment:
    """
    Population compartment in a compartmental epidemiological model.

    For example, in an S-E-I-R model, we would have an EpiCompartment
    instance for each compartment.
    """

    __slots__ = ("name",
                 "init_val",
                 "current_val",
                 "inflow_transition_variable_names_list",
                 "outflow_transition_variable_names_list",
                 "history_vals_list",
                 "outgoing_transition_rate_function")

    def __init__(self,
                 name,
                 init_val,
                 inflow_transition_variable_names_list,
                 outflow_transition_variable_names_list):
        """
        :parameter name: string,
            name of EpiCompartment
        :parameter init_val: array-like of positive ints,
            starting population for EpiCompartment
        :parameter incoming_transition_variable_names_list: array-like of TransitionVariable objects,
            list of TransitionVariable objects that inflow to epi compartment
        :parameter outgoing_transition_variable_names_list: array-like of TransitionVariable objects,
            list of TransitionVariable objects that outflow from epi compartment
        """

        self.name = name
        self.init_val = init_val
        self.current_val = copy.deepcopy(init_val)

        self.inflow_transition_variable_names_list = inflow_transition_variable_names_list
        self.outflow_transition_variable_names_list = outflow_transition_variable_names_list

        self.history_vals_list = []  # historical values

        self.outgoing_transition_rate_function = None


class StateVariable:
    """
    Class for variables that are deterministic functions of
    the simulation state (meaning epi compartment values, other parameters,
    and time.)

    For example, population-level immunity against hospitalization and
    infection can both be modeled as a StateVariable instance.
    """

    def __init__(self,
                 name,
                 init_val):
        self.name = name
        self.init_val = init_val
        self.current_val = copy.deepcopy(init_val)
        self.change_in_current_val = None
        self.state_update_function = None


class SimulationParams:

    def __init__(self,
                 timesteps_per_day=7,
                 starting_simulation_day=0,
                 transition_type="binomial"):
        self.timesteps_per_day = timesteps_per_day
        self.starting_simulation_day = starting_simulation_day
        self.transition_type = transition_type

class BaseModel(ABC):

    def __init__(self,
                 RNG_seed=np.random.SeedSequence()):

        self.name_to_epi_compartment_dict = {}
        self.name_to_transition_variable_dict = {}
        self.name_to_transition_variable_group_dict = {}
        self.name_to_state_variable_dict = {}

        self._bit_generator = np.random.MT19937(seed=RNG_seed)
        self._RNG = np.random.Generator(self._bit_generator)

        self.current_day_counter = 0

        self.epi_params = None
        self.simulation_params = None

    def modify_model_RNG_seed(self, RNG_seed):

        self._bit_generator = np.random.MT19937(seed=RNG_seed)
        self._RNG = np.random.Generator(self._bit_generator)

        for tvar_group in self.name_to_transition_variable_group_dict.values():
            tvar_group.RNG = self._RNG

        for tvar in self.name_to_transition_variable_dict.values():
            tvar.RNG = self._RNG

    def modify_model_transition_type(self, transition_type):

        for tvar_group in self.name_to_transition_variable_group_dict.values():
            tvar_group.assign_transition_type(transition_type)

        for tvar in self.name_to_transition_variable_dict.values():
            tvar.assign_transition_type(transition_type)

    def add_epi_compartments_from_json(self, json_filename):
        """
        Load json_filename and add Epi Compartment instances
            for each entry

        :parameter json_filename: string ending in ".json"
        """

        with open(json_filename) as f:
            d = json.load(f)

        # Must convert list to numpy array to support numpy operations
        for epi_compartment_name, data_dict in d["epi_compartments"].items():
            self.add_epi_compartment(epi_compartment_name,
                                     np.asarray(data_dict["init_val"]),
                                     data_dict["incoming_transition_variable_names_list"],
                                     data_dict["outgoing_transition_variable_names_list"],
                                     data_dict["outgoing_transition_rate_function"])

    def add_epi_compartment(self,
                            name,
                            init_val,
                            incoming_transition_variable_names_list,
                            outgoing_transition_variable_names_list,
                            outgoing_transition_rate_function):
        """
        Constructor method for adding EpiCompartment instance to model

        Automatically constructs TransitionVariable instances and
        TransitionVariableGroup instances based on
        incoming_transition_variable_names_list and
        outgoing_transition_variable_names_list

        Creates model attribute named "name" that is assigned to
        new EpiCompartment instance

        Updates name_to_epi_compartment_dict attribute by adding
        (name, EpiCompartment) as (key, value) pair

        For each variable name in incoming_transition_variable_names_list
        and outgoing_transition_variable_names_list, creates a
        corresponding TransitionVariable if one does not already exist
        and assigns it as a model attribute

        If outgoing_Transition_variable_names_list is longer than 1,
        then multiple outflows exist and a TransitionVariableGroup is
        created and also assigned as a model attribute with
        the automatic name "name" [name of epi compartment] + "_out"

        For example, if all outgoing transition variables from H are:
        "H_to_D" and "H_to_R", then these transition variables are
        automatically created, and a transition variable group
        named "H_out" is automatically created

        See EpiCompartment class (__init__ function) for parameters.
        """

        epi_compartment = EpiCompartment(name,
                                         init_val,
                                         incoming_transition_variable_names_list,
                                         outgoing_transition_variable_names_list)
        # store rate update function
        epi_compartment.outgoing_transition_rate_function = outgoing_transition_rate_function


        self.name_to_epi_compartment_dict[name] = epi_compartment
        setattr(self, name, epi_compartment)

        timesteps_per_day = self.simulation_params.timesteps_per_day
        transition_type = self.simulation_params.transition_type
        RNG = self._RNG

        # If there are multiple outflows from this epi compartment,
        #   create corresponding TransitionVariables where is_jointly_distributed = True
        #   and create a TransitionVariableGroup instance to hold these transition variables
        if len(outgoing_transition_variable_names_list) > 1:
            for transition_variable_name in outgoing_transition_variable_names_list:
                self.add_transition_variable(transition_variable_name,
                                             epi_compartment,
                                             timesteps_per_day,
                                             RNG,
                                             transition_type,
                                             True,
                                             self.epi_params.num_age_groups,
                                             self.epi_params.num_risk_groups)

            # Create a list of transition variable objects to create a TransitionVariableGroup
            transition_variables_list = [
                self.name_to_transition_variable_dict[vname] for vname in outgoing_transition_variable_names_list]

            # Create a new TransitionVariableGroup with the name "name" [name of epi compartment] + "_out"
            self.add_transition_variable_group(name + "_out",
                                               epi_compartment,
                                               timesteps_per_day,
                                               RNG,
                                               transition_type,
                                               transition_variables_list,
                                               self.epi_params.num_age_groups,
                                               self.epi_params.num_risk_groups)
        else:
            for transition_variable_name in outgoing_transition_variable_names_list:
                self.add_transition_variable(transition_variable_name,
                                             epi_compartment,
                                             timesteps_per_day,
                                             RNG,
                                             transition_type,
                                             False,
                                             self.epi_params.num_age_groups,
                                             self.epi_params.num_risk_groups)


    def add_transition_variable(self,
                                name,
                                base_count_epi_compartment,
                                timesteps_per_day,
                                RNG,
                                transition_type,
                                is_jointly_distributed,
                                num_age_groups,
                                num_risk_groups):
        """
        Constructor method for adding TransitionVariable instance to model

        Creates model attribute named "name" that is assigned to
        new TransitionVariable instance

        Updates name_to_transition_variable_dict attribute by adding
        (name, TransitionVariable) as (key, value) pair

        For example, if all outflows of compartment H are: R and D,
        i.e. from the Hospital compartment, people either go to Recovered
        or Dead, then TransitionVariable instances R and D have
        attribute is_jointly_distributed = True

        See TransitionVariable class (__init__ function) for parameters.
        """

        tvar = TransitionVariable(name,
                                  base_count_epi_compartment,
                                  timesteps_per_day,
                                  RNG,
                                  transition_type,
                                  is_jointly_distributed,
                                  num_age_groups,
                                  num_risk_groups)

        self.name_to_transition_variable_dict[name] = tvar
        setattr(self, name, tvar)

    def add_transition_variable_group(self,
                                      name,
                                      base_count_epi_compartment,
                                      timesteps_per_day,
                                      RNG,
                                      transition_type,
                                      transition_variables_list,
                                      num_age_groups,
                                      num_risk_groups):
        """
        Constructor method for adding TransitionVariableGroup instance to model

        Creates model attribute named "name" that is assigned to
        new TransitionVariableGroup instance

        Updates name_to_transition_variable_group_dict attribute by adding
        (name, TransitionVariableGroup instance) as new (key, value) pair

        For example, if all outflows of compartment H are: R and D,
        i.e. from the Hospital compartment, people either go to Recovered
        or Dead, then base_count_epi_compartment is H and
        transition_variable_list is [R, D], where H, R, D are all
        EpiCompartment objects

        See TransitionVariableGroup class (__init__ function) for parameters.
        """

        transition_type = transition_type.replace("binomial", "multinomial")

        tvgroup = TransitionVariableGroup(name,
                                          base_count_epi_compartment,
                                          timesteps_per_day,
                                          RNG,
                                          transition_type,
                                          transition_variables_list,
                                          num_age_groups,
                                          num_risk_groups)

        self.name_to_transition_variable_group_dict[name] = tvgroup
        setattr(self, name, tvgroup)

    def add_state_variables_from_json(self, json_filename):
        """
        Load json_filename and add Epi Compartment instances
        for each entry
        """

        with open(json_filename) as f:
            d = json.load(f)

        # Must convert list to numpy array to support numpy operations
        for state_variable_name, data_dict in d["immune_compartments"].items():
            self.add_state_variable(state_variable_name,
                                    np.asarray(data_dict["init_val"]),
                                    data_dict["state_update_function"])

    def add_state_variable(self,
                           name,
                           init_val,
                           state_update_function):
        """
        Constructor method for adding StateVariable instance to model

        Creates model attribute named "name" that is assigned to
        new StateVariable instance

        Updates name_to_state_variable_dict attribute by adding
        (name, StateVariable instance) as new (key, value) pair

        :parameter name: string,
            name of state variable
        :parameter init_val: array-like,
            initial value of state variable at simulation start
        """

        state_variable = StateVariable(name, init_val)
        # store state update function
        state_variable.state_update_function = state_update_function

        self.name_to_state_variable_dict[name] = state_variable
        setattr(self, name, state_variable)


    def add_epi_params_from_json(self, json_filename):
        """
        Assign epi_params attribute to EpiParams instance
        Load json_filename and assign values to epi_params attribute
        Converts lists to numpy arrays
        """

        self.epi_params = EpiParams()

        with open(json_filename) as f:
            d = json.load(f)

        # json does not support arrays -- so must convert lists
        #   to numpy arrays to support numpy operations
        for key, value in d.items():
            if type(value) is list:
                value = np.asarray(value)
            setattr(self.epi_params, key, value)

    def add_epi_params(self, epi_params):
        self.epi_params = epi_params

    def add_simulation_params_from_json(self, json_filename):
        """
        Assign simulation_params attribute to SimulationParams instance
        Load json_filename and assign values to simulation_params attribute
        """

        self.simulation_params = SimulationParams()

        with open(json_filename) as f:
            d = json.load(f)

        for key, value in d.items():
            setattr(self.simulation_params, key, value)

    def add_simulation_params(self, simulation_params):
        self.simulation_params = simulation_params

    def reset_simulation(self):
        """
        Reset each compartment, state variable, and transition variable
        to default/starting values, reset current_day_counter to 0

        For each EpiCompartment instance, set current_val to init_val
        and reset history_vals_list to []

        For each StateVariable instance, set current_val to init_val
        and change_in_current_val to None

        For each TransitionVariable instance, set current_rate to None
        and current_realization to None

        "Rewind" simulation time to 0 by setting current_day_counter to 0

        Note that TransitionVariableGroup do not store/possess their own
        current values or rates -- they are simply containers for
        TransitionVariables, and so TransitionVariableGroup instances
        do not need to be reset
        """

        for compartment in self.name_to_epi_compartment_dict.values():
            compartment.current_val = copy.deepcopy(compartment.init_val)
            compartment.history_vals_list = []

        for svar in self.name_to_state_variable_dict.values():
            svar.current_val = copy.deepcopy(svar.init_val)
            svar.change_in_current_val = None

        for tvar in self.name_to_transition_variable_dict.values():
            tvar.current_rate = None
            tvar.current_realization = None

        self.current_day_counter = 0

    def update_history_vals_list(self):
        """
        For each EpiCompartment instance attached to current model,
        append copy of compartment's current_val to compartment's
        history_vals_list
        """

        for compartment in self.name_to_epi_compartment_dict.values():
            compartment.history_vals_list.append(copy.deepcopy(compartment.current_val))

    def simulate_until_time_period(self, last_simulation_day):
        """
        Advance simulation model time until last_simulation_day

        Advance time by iterating through simulation days,
        which are simulated by iterating through discretized
        timesteps

        Save daily simulation data as history

        :parameter last_simulation_day: positive int,
            stop simulation at last_simulation_day (i.e. exclusive,
            simulate up to but not including last_simulation_day)
        """

        # last_simulation_day is exclusive endpoint
        while self.current_day_counter < last_simulation_day:
            self.simulate_discretized_timesteps()
            self.update_history_vals_list()

    def simulate_discretized_timesteps(self):
        """
        Subroutine for simulate_until_time_period

        Iterates through discretized timesteps to simulate next
        simulation day. Granularity of discretization is given by
        attribute simulation_params.timesteps_per_day

        Properly scales transition variable realizations and changes
        in state variables by specified timesteps per day
        """

        # Attribute lookup shortcut
        timesteps_per_day = self.simulation_params.timesteps_per_day

        for timestep in range(timesteps_per_day):

            # Users will inherit from base class BaseModel and override
            #   these functions to customize their model
            self.update_change_in_state_variables()
            self.update_transition_rates()

            # Obtain transition variable realizations for jointly distributed transition variables
            #   (i.e. when there are multiple transition variable otuflows from an epi compartment)
            for tvargroup in self.name_to_transition_variable_group_dict.values():
                tvargroup.current_realizations_list = tvargroup.get_joint_realization()
                tvargroup.update_transition_variable_realizations()

            # Obtain transition variable realizations for marginally distributed transition variables
            #   (i.e. when there is only one transition variable outflow from an epi compartment)
            # If transition variable is jointly distributed, then its realization has already
            #   been computed by its transition variable group container previously, so skip it
            for tvar in self.name_to_transition_variable_dict.values():
                if tvar.is_jointly_distributed:
                    continue
                else:
                    tvar.current_realization = tvar.get_realization()

            # In-place updates -- advance the simulation
            self.update_epi_compartments()
            self.update_state_variables()

            # Reset transition variable current_realization and current_rate to None
            for tvar in self.name_to_transition_variable_dict.values():
                tvar.reset()

        # Move to next day in simulation
        self.current_day_counter += 1

    def update_epi_compartment_current_val(self, compartment):
        """
        Update current_val in-place for given compartment

        Computes total inflow based on all inflow transition variables
        and total outflow based on all outflow transition variables
        to get net outflow

        Updated value is previous current_val plus net outflow

        Assumes that transition variables have been already sampled.

        :parameter compartment: EpiCompartment object,
            whose attribute current_val gets updated in place
        """

        total_inflow = np.sum([self.name_to_transition_variable_dict[var_name].current_realization
                               for var_name in compartment.inflow_transition_variable_names_list], axis=0)
        total_outflow = np.sum([self.name_to_transition_variable_dict[var_name].current_realization
                               for var_name in compartment.outflow_transition_variable_names_list], axis=0)
        compartment.current_val += (total_inflow - total_outflow)

    def update_epi_compartments(self):
        """
        Update current_val in-place for each epi_compartment in model
        """
        for compartment in self.name_to_epi_compartment_dict.values():
            self.update_epi_compartment_current_val(compartment)

    def update_state_variables(self):
        """
        Update state variables with proper scale according to timesteps per day
        """

        timesteps_per_day = self.simulation_params.timesteps_per_day

        for svar in self.name_to_state_variable_dict.values():
            svar.current_val += svar.change_in_current_val / timesteps_per_day

    @abstractmethod
    def update_change_in_state_variables(self):
        pass

    @abstractmethod
    def update_transition_rates(self):
        pass


def approx_binomial_probability_from_rate(rate, interval_length):
    """
    Converts a rate (events per time) to the probability of any event
    occurring in the next time interval of length interval_length,
    assuming the number of events occurring in time interval
    follows a Poisson distribution with given rate parameter

    The probability of 0 events in time_interval_length is
    e^(-rate * time_interval_length), so the probability of any event
    in time_interval_length is 1 - e^(-rate * time_interval_length)

    :parameter rate: array-like or positive scalars,
        dimension A x L (number of age groups
        x number of risk groups), rate parameters
        in a Poisson distribution
    :parameter interval_length: positive scalar,
        length of time interval in simulation days
    :return: array-like of positive scalars,
        dimension A x L
    """

    return 1 - np.exp(-rate * interval_length)