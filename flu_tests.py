# WARNING:
#   Currently excluding Poisson transition types from automatic
#   test run. Poisson transition types are not always well-behaved
#   -- this is because the Poisson distribution is unbounded.
#   For example, values of beta_baseline that are too high
#   can result in negative compartment populations.
#   Similarly, if num_timesteps is small, this can result
#   in negative compartment populations. Sometimes tests fail because
#   these choices of parameter values and parameter initial values
#   are unsuitable for well-behaved Poisson random variables.

# WARNING:
#   If there are too few timesteps_per_day (in each SubpopModel's
#   Config), then it is possible for Binomial Taylor Approximation
#   (both stochastic and deterministic) to fail. This is because
#   the input into the "probability parameter" for the numpy
#   random binomial draw may be not be in [0,1]. Thus, the
#   Binomial Taylor Approximation transition type may not reliably
#   pass all tests with arbitrary Config values.

import flu_model as flu
import clt_base as clt

import numpy as np
import pandas as pd
import copy
import pytest

from pathlib import Path

base_path = Path(__file__).parent / "flu_demo_input_files"

config_filepath = base_path / "config.json"
params_filepath = base_path / "common_params.json"
compartments_epi_metrics_init_vals_filepath = base_path / "compartments_epi_metrics_init_vals.json"
calendar_filepath = base_path / "school_work_calendar.csv"
travel_proportions_filepath = base_path / "travel_proportions.json"

compartments_epi_metrics_dict = clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath)
params_dict = clt.load_json_new_dict(params_filepath)
config_dict = clt.load_json_new_dict(config_filepath)
calendar_df = pd.read_csv(calendar_filepath, index_col=0)
travel_proportions = clt.load_json_new_dict(travel_proportions_filepath)

bit_generator = np.random.MT19937(88888)
jumped_bit_generator = bit_generator.jumped(1)

subpopA_model = flu.FluSubpopModel(compartments_epi_metrics_dict,
                                   params_dict,
                                   config_dict,
                                   calendar_df,
                                   np.random.Generator(bit_generator))

starting_random_seed = 123456789123456789


def check_state_variables_same_history(subpop_model_A: clt.SubpopModel,
                                       subpop_model_B: clt.SubpopModel):
    for name in subpop_model_A.all_state_variables.keys():
        assert np.array_equal(np.array(subpop_model_A.all_state_variables[name].history_vals_list),
                              np.array(subpop_model_B.all_state_variables[name].history_vals_list))


def create_subpop_models_all_transition_types(RNG: np.random.Generator) -> list:
    models_list = []

    for transition_type in clt.TransitionTypes:

        if "poisson" not in transition_type:
            #  Need deep copy -- otherwise changing "transition_type" on
            #   model_constructor.config changes config attribute for all
            #   models in models_list
            new_config_dict = copy.deepcopy(config_dict)
            new_config_dict["transition_type"] = transition_type

            models_list.append(flu.FluSubpopModel(compartments_epi_metrics_dict,
                                                  params_dict,
                                                  new_config_dict,
                                                  calendar_df,
                                                  RNG))

    return models_list


subpop_models_transition_variations_list = \
    create_subpop_models_all_transition_types(np.random.Generator(bit_generator))


def test_num_timesteps():
    """
    If "timesteps_per_day" in Config increases (number of timesteps per day
        increases), then step sizes are smaller.

    Using binomial deterministic transitions, realizations will be smaller
        for more timesteps per day.
    """

    new_config = copy.deepcopy(subpopA_model.config)
    new_config.timesteps_per_day = 2
    new_config.transition_type = "binomial_deterministic"

    few_timesteps_model = flu.FluSubpopModel(compartments_epi_metrics_dict,
                                             params_dict,
                                             config_dict,
                                             calendar_df,
                                             np.random.default_rng(starting_random_seed))

    few_timesteps_model.prepare_daily_state()
    few_timesteps_model.simulate_timesteps(1)

    new_config = copy.deepcopy(subpopA_model.config)
    new_config.timesteps_per_day = 20
    new_config.transition_type = "binomial_deterministic"

    many_timesteps_model = flu.FluSubpopModel(compartments_epi_metrics_dict,
                                              params_dict,
                                              config_dict,
                                              calendar_df,
                                              np.random.default_rng(starting_random_seed))

    many_timesteps_model.prepare_daily_state()
    many_timesteps_model.simulate_timesteps(1)

    for name in few_timesteps_model.transition_variables.keys():
        assert (few_timesteps_model.transition_variables[name].current_val >=
                many_timesteps_model.transition_variables[name].current_val).all()


def test_subpop_correct_object_count():
    """
    For each SubpopModel, there should be 8 epi compartments,
        10 transition variables, 2 transition variable groups,
        and 3 epi metrics
    """

    assert len(subpopA_model.compartments) == 8
    assert len(subpopA_model.transition_variables) == 10
    assert len(subpopA_model.transition_variable_groups) == 3

    if subpopA_model.wastewater_enabled:
        assert len(subpopA_model.epi_metrics) == 3
    else:
        assert len(subpopA_model.epi_metrics) == 2

    assert len(subpopA_model.dynamic_vals) == 1


def test_subpop_constructor_no_unintended_sharing():
    """
    Regression test: there was a previous bug where the same
        SubpopState object was being shared across multiple models created
        by subsequent creation calls on the model constructor.
        This is remedied using deep copies -- we make sure that
        the model constructor always creates a transmission model
        with its own distinct/independent SubpopState OBJECT,
        even if the actual initial SubpopState VALUES are the same
        across models.

    This test makes sure that SubpopState objects across models
        created by the same constructor are indeed distinct/independent.
    """

    initial_compartments_epi_metrics_dict = copy.deepcopy(compartments_epi_metrics_dict)

    first_model = flu.FluSubpopModel(compartments_epi_metrics_dict,
                                     params_dict,
                                     config_dict,
                                     calendar_df,
                                     np.random.default_rng(1))

    second_model = flu.FluSubpopModel(compartments_epi_metrics_dict,
                                      params_dict,
                                      config_dict,
                                      calendar_df,
                                      np.random.default_rng(1))

    first_model.simulate_until_day(100)

    # The initial state of the second model should still be the same
    #   initial state -- it should not have been affected by simulating
    #   the first model

    for key, value in initial_compartments_epi_metrics_dict.items():
        if isinstance(value, (np.ndarray, list)):
            try:
                assert (getattr(second_model.state, key) ==
                        initial_compartments_epi_metrics_dict[key])
            # if it's an array, have to check equality of each element --
            #   Python will complain that the truth value of an array is ambiguous
            except ValueError:
                assert (getattr(second_model.state, key) ==
                        initial_compartments_epi_metrics_dict[key]).all()


def test_subpop_constructor_reproducible_results():
    """
    If the flu model constructor creates two identical models
        with the same starting random number seed, they should give
        the same results. Specifically, if the first model is simulated
        before the second model is created, the results should still
        be the same.

    Also a way of ensuring there is no unintended object sharing
        or unintended mutability issues with model constructors.
        Specifically, simulating a model created from a constructor
        should not modify objects on that constructor.
    """

    first_model = flu.FluSubpopModel(compartments_epi_metrics_dict,
                                     params_dict,
                                     config_dict,
                                     calendar_df,
                                     np.random.default_rng(1))

    first_model.simulate_until_day(100)

    second_model = flu.FluSubpopModel(compartments_epi_metrics_dict,
                                      params_dict,
                                      config_dict,
                                      calendar_df,
                                      np.random.default_rng(1))
    second_model.simulate_until_day(100)

    check_state_variables_same_history(first_model, second_model)


def test_subpop_no_transmission_when_beta_zero():
    """
    If the transmission rate beta_baseline = 0, then S should not decrease over time
    """

    subpop_model = flu.FluSubpopModel(compartments_epi_metrics_dict,
                                      params_dict,
                                      config_dict,
                                      calendar_df,
                                      np.random.default_rng(1))

    subpop_model.reset_simulation()
    subpop_model.params.beta_baseline = 0
    subpop_model.simulate_until_day(300)

    S_history = subpop_model.compartments["S"].history_vals_list

    assert np.sum((np.diff(np.sum(S_history, axis=(1, 2))) >= 0)) == len(S_history) - 1


@pytest.mark.parametrize("subpop_model", subpop_models_transition_variations_list)
def test_subpop_dead_compartment_monotonic(subpop_model):
    """
    People do not rise from the dead; the dead compartment
        should not decrease over time
    """

    subpop_model.reset_simulation()
    subpop_model.params.beta = 2
    subpop_model.simulate_until_day(300)

    D_history = subpop_model.compartments["D"].history_vals_list

    assert np.sum(np.diff(np.sum(D_history, axis=(1, 2))) >= 0) == len(D_history) - 1


@pytest.mark.parametrize("subpop_model", subpop_models_transition_variations_list)
def test_subpop_population_is_constant(subpop_model):
    """
    The total population (summed over all compartments and age-risk groups)
        should be constant over time, equal to the initial total population.
    """

    subpop_model.reset_simulation()
    subpop_model.params.beta = 0.25

    for day in range(300):
        subpop_model.simulate_until_day(day)

        current_sum_all_compartments = 0
        for compartment in subpop_model.compartments.values():
            current_sum_all_compartments += np.sum(compartment.current_val)

        assert np.abs(current_sum_all_compartments -
                      np.sum(subpop_model.params.total_pop_age_risk)) < 1e-6


@pytest.mark.parametrize("subpop_model", subpop_models_transition_variations_list)
def test_subpop_reset_reproducible_results(subpop_model: flu.FluSubpopModel):
    """
    Resetting the random number generator and simulating should
        give the same results as the initial run.
    """

    subpop_model.reset_simulation()
    subpop_model.modify_random_seed(starting_random_seed)
    subpop_model.simulate_until_day(100)

    original_model_history_dict = {}

    for name, compartment in subpop_model.compartments.items():
        original_model_history_dict[name] = \
            copy.deepcopy(compartment.history_vals_list)

    reset_model_history_dict = {}

    subpop_model.reset_simulation()
    subpop_model.modify_random_seed(starting_random_seed)
    subpop_model.simulate_until_day(100)

    for name, compartment in subpop_model.compartments.items():
        reset_model_history_dict[name] = \
            copy.deepcopy(compartment.history_vals_list)

    for name in subpop_model.compartments.keys():
        assert np.array_equal(np.array(original_model_history_dict[name]),
                              np.array(reset_model_history_dict[name]))


@pytest.mark.parametrize("subpop_model", subpop_models_transition_variations_list)
def test_compartments_integer_population(subpop_model: flu.FluSubpopModel):
    """
    Compartment populations should be integer-valued.
    """

    subpop_model.reset_simulation()
    subpop_model.modify_random_seed(starting_random_seed)

    for day in [1, 10, 100]:
        subpop_model.simulate_until_day(day)

        for compartment in subpop_model.compartments.values():
            assert (compartment.current_val ==
                    np.asarray(compartment.current_val, dtype=int)).all()


@pytest.mark.parametrize("subpop_model", subpop_models_transition_variations_list)
def test_transition_format(subpop_model: flu.FluSubpopModel):
    """
    Transition variables' transition rates and
        current value should be A x L, where
        A is the number of risk groups and L is the
        number of age groups.

    Transition rates should also be floats, even though the
        transition variable realization is integer
        (so that population counts in compartments
        always stay integer). Transition rates should be
        floats to prevent premature rounding. Binomial
        and Poisson random variables are always integer,
        but their deterministic equivalents may not be
        under our implementation -- so we round them
        after the fact.
    """

    A = subpop_model.params.num_age_groups
    L = subpop_model.params.num_risk_groups

    subpop_model.reset_simulation()
    subpop_model.modify_random_seed(starting_random_seed)

    for day in [1, 10, 100]:
        subpop_model.simulate_until_day(day)

        for tvar in subpop_model.transition_variables.values():
            assert np.shape(tvar.current_rate) == (A, L)
            assert np.shape(tvar.current_val) == (A, L)

            for element in tvar.current_rate.flatten():
                assert isinstance(element, float)


@pytest.mark.parametrize("subpop_model", subpop_models_transition_variations_list)
def test_metapop_no_travel(subpop_model: flu.FluSubpopModel):
    """
    If two subpopulations comprise a MetapopModel (travel model), then
    if there is no travel between the two subpopulations, the
    MetapopModel should behave exactly like two INDEPENDENTLY RUN
    versions of the SubpopModel instances.

    We can "turn travel off" in multiple ways:
    - Setting pairwise travel proportions to 0 (so that 0% of
        subpopulation i travels to subpopulation j, for each
        distinct i,j subpopulation pair, i != j)
    - Or setting the prop_time_away_by_age to 0 for each
        subpopulation
    We test both of these options, one at a time

    Note -- this test will only pass when timesteps_per_day on
    each Config is 1. This is because, for the sake of efficiency,
    for MetapopModel instances, each InteractionTerm is updated
    only ONCE PER DAY rather than after every single discretized timestep.
    In contrast, independent SubpopModel instances (not linked by any
    metapopulation/travel model) do not have any interaction terms.
    The S_to_E transition variable rate does not depend on any
    interaction terms, and depends on state variables that get updated
    at every discretized timestep.
    """

    config_dict_1_timestep = copy.deepcopy(config_dict)
    config_dict_1_timestep["timesteps_per_day"] = 1

    subpopA = flu.FluSubpopModel(compartments_epi_metrics_dict,
                                 params_dict,
                                 config_dict_1_timestep,
                                 calendar_df,
                                 np.random.default_rng(starting_random_seed),
                                 name="subpopA")

    subpopB = flu.FluSubpopModel(compartments_epi_metrics_dict,
                                 params_dict,
                                 config_dict_1_timestep,
                                 calendar_df,
                                 np.random.default_rng(starting_random_seed ** 2),
                                 name="subpopB")

    AB_inter_subpop_repo = flu.FluInterSubpopRepo({"subpopA": subpopA, "subpopB": subpopB},
                                                  {"subpopA": 0, "subpopB": 1},
                                                  travel_proportions["travel_proportions_array"])

    metapopAB_model = flu.FluMetapopModel(AB_inter_subpop_repo)

    metapopAB_model.simulate_until_day(100)

    subpopA_independent = flu.FluSubpopModel(compartments_epi_metrics_dict,
                                             params_dict,
                                             config_dict_1_timestep,
                                             calendar_df,
                                             np.random.default_rng(starting_random_seed),
                                             name="subpopA")

    subpopB_independent = flu.FluSubpopModel(compartments_epi_metrics_dict,
                                             params_dict,
                                             config_dict_1_timestep,
                                             calendar_df,
                                             np.random.default_rng(starting_random_seed ** 2),
                                             name="subpopB")

    subpopA_independent.simulate_until_day(100)
    subpopB_independent.simulate_until_day(100)

    check_state_variables_same_history(subpopA, subpopA_independent)
    check_state_variables_same_history(subpopB, subpopB_independent)

    metapopAB_model.reset_simulation()

    subpopA_independent.reset_simulation()
    subpopB_independent.reset_simulation()

    num_age_groups = params_dict["num_age_groups"]
    num_risk_groups = params_dict["num_risk_groups"]

    metapopAB_model.subpop_models.subpopA.params.prop_time_away_by_age = np.zeros((num_age_groups, num_risk_groups))

    subpopA_independent.params.prop_time_away_by_age = np.zeros((num_age_groups, num_risk_groups))
    subpopB_independent.params.prop_time_away_by_age = np.zeros((num_age_groups, num_risk_groups))

    metapopAB_model.simulate_until_day(100)

    subpopA_independent.simulate_until_day(100)
    subpopB_independent.simulate_until_day(100)

    check_state_variables_same_history(subpopA, subpopA_independent)
    check_state_variables_same_history(subpopB, subpopB_independent)


# TODO: NEED TO ADD MORE WASTEWATER TESTS
# wastewater test
@pytest.mark.parametrize("model", subpop_models_transition_variations_list)
def test_wastewater_when_beta_zero(model):
    """
    If the transmission rate beta_baseline = 0, then viral load should be zero
    """
    if model.wastewater_enabled:
        model.reset_simulation()
        model.params.beta_baseline = 0
        model.simulate_until_day(300)

        ww_history = model.epi_metrics["wastewater"].history_vals_list
        tol = 1e-6
        assert np.sum(np.abs(ww_history) < tol) == len(ww_history)
