# Note: pytest fixtures are AMAZING! For those unfamiliar
#   with fixtures, please read more (for example, here:
#   https://docs.pytest.org/en/6.2.x/fixture.html)
#   -- fixtures are recreated for each test, which means that
#   each test gets a fresh instant and we do not have to
#   "reset" the simulation (or state) in between tests.
#   There are many other benefits as well...

# WARNING:
#   If there are too few timesteps_per_day (in each SubpopModel's
#   SimulationSettings), then it is possible for Binomial Taylor Approximation
#   (both stochastic and deterministic) to fail. This is because
#   the input into the "probability parameter" for the numpy
#   random binomial draw may be not be in [0,1]. Thus, the
#   Binomial Taylor Approximation transition type may not reliably
#   pass all tests with arbitrary SimulationSettings values.

import flu_core as flu
import clt_toolkit as clt

import numpy as np
import pandas as pd
import copy
import pytest

from flu_fixtures import subpop_inputs, make_subpop_model
from clt_toolkit import updated_dataclass

base_path = clt.utils.PROJECT_ROOT / "tests" / "test_input_files"

import sys

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
binomial_transition_types_list = [clt.TransitionTypes.BINOMIAL,
                                  clt.TransitionTypes.BINOMIAL_DETERMINISTIC,
                                  clt.TransitionTypes.BINOMIAL_TAYLOR_APPROX,
                                  clt.TransitionTypes.BINOMIAL_TAYLOR_APPROX_DETERMINISTIC]

binomial_random_transition_types_list = [clt.TransitionTypes.BINOMIAL,
                                         clt.TransitionTypes.BINOMIAL_TAYLOR_APPROX]

inputs_id_list = ["caseA", "caseB_subpop1"]


def check_state_variables_same_history(subpop_model_A: clt.SubpopModel,
                                       subpop_model_B: clt.SubpopModel):
    for name in subpop_model_A.all_state_variables.keys():
        assert np.array_equal(np.array(subpop_model_A.all_state_variables[name].history_vals_list),
                              np.array(subpop_model_B.all_state_variables[name].history_vals_list))


def test_num_timesteps(make_subpop_model):
    """
    If "timesteps_per_day" in SimulationSettings increases (number of timesteps per day
        increases), then step sizes are smaller.

    Using binomial deterministic transitions, realizations will be smaller
        for more timesteps per day.
    """

    few_timesteps_model = make_subpop_model("few_timesteps",
                                            clt.TransitionTypes.BINOMIAL_DETERMINISTIC,
                                            timesteps_per_day = 2)

    few_timesteps_model.prepare_daily_state()
    few_timesteps_model._simulate_timesteps(1)

    many_timesteps_model = make_subpop_model("many_timesteps",
                                             clt.TransitionTypes.BINOMIAL_DETERMINISTIC,
                                             timesteps_per_day = 20)

    many_timesteps_model.prepare_daily_state()
    many_timesteps_model._simulate_timesteps(1)

    for name in few_timesteps_model.transition_variables.keys():
        assert (few_timesteps_model.transition_variables[name].current_val >=
                many_timesteps_model.transition_variables[name].current_val).all()


def test_subpop_correct_object_count(make_subpop_model):
    """
    For each SubpopModel, there should be 8 epi compartments,
        10 transition variables, 2 transition variable groups,
        and 3 epi metrics
    """

    model = make_subpop_model("model")

    assert len(model.compartments) == 8
    assert len(model.transition_variables) == 10
    assert len(model.transition_variable_groups) == 3

    assert len(model.epi_metrics) == 2

    assert len(model.dynamic_vals) == 1


@pytest.mark.parametrize("transition_type", binomial_transition_types_list)
def test_subpop_constructor_no_unintended_sharing(make_subpop_model, transition_type):
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

    first_model = make_subpop_model("first", transition_type)
    second_model = make_subpop_model("second", transition_type)

    init_vals = copy.deepcopy(second_model.state)

    first_model.simulate_until_day(100)

    # The initial state of the second model should still be the same
    #   initial state -- it should not have been affected by simulating
    #   the first model

    for key, value in vars(init_vals).items():
        if isinstance(value, (np.ndarray, list)):
            try:
                assert (getattr(second_model.state, key) ==
                        getattr(init_vals, key))
            # if it's an array, have to check equality of each element --
            #   Python will complain that the truth value of an array is ambiguous
            except ValueError:
                assert (getattr(second_model.state, key) ==
                        getattr(init_vals, key)).all()


@pytest.mark.parametrize("transition_type", binomial_random_transition_types_list)
def test_subpop_constructor_reproducible_results(make_subpop_model, transition_type):
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

    first_model = make_subpop_model("first", transition_type)
    second_model = make_subpop_model("second", transition_type)

    first_model.simulate_until_day(100)
    second_model.simulate_until_day(100)

    check_state_variables_same_history(first_model, second_model)


@pytest.mark.parametrize("transition_type", binomial_random_transition_types_list)
def test_subpop_no_transmission_when_beta_zero(make_subpop_model, transition_type):
    """
    If the transmission rate beta_baseline = 0, then S should not decrease over time
    """

    subpop_model = make_subpop_model("subpop_model", transition_type)
    subpop_model.reset_simulation()
    subpop_model.modify_params({"beta_baseline": 0})
    subpop_model.simulate_until_day(300)

    S_history = subpop_model.compartments["S"].history_vals_list

    assert np.sum((np.diff(np.sum(S_history, axis=(1, 2))) >= 0)) == len(S_history) - 1


@pytest.mark.parametrize("transition_type", binomial_transition_types_list)
def test_subpop_dead_compartment_monotonic(make_subpop_model, transition_type):
    """
    People do not rise from the dead; the dead compartment
        should not decrease over time
    """

    subpop_model = make_subpop_model("subpop_model", transition_type)

    subpop_model.reset_simulation()
    subpop_model.modify_params({"beta_baseline": 1.1})
    subpop_model.simulate_until_day(300)

    D_history = subpop_model.compartments["D"].history_vals_list

    assert np.sum(np.diff(np.sum(D_history, axis=(1, 2))) >= 0) == len(D_history) - 1


@pytest.mark.parametrize("transition_type", binomial_transition_types_list)
@pytest.mark.parametrize("inputs_id", inputs_id_list)
def test_subpop_population_is_constant(make_subpop_model, transition_type, inputs_id):
    """
    The total population (summed over all compartments and age-risk groups)
        should be constant over time, equal to the initial total population.
    """

    subpop_model = make_subpop_model("subpop_model", transition_type, case_id_str = inputs_id)

    for day in range(300):
        subpop_model.simulate_until_day(day)

        current_sum_all_compartments = 0
        for compartment in subpop_model.compartments.values():
            current_sum_all_compartments += np.sum(compartment.current_val)

        assert np.abs(current_sum_all_compartments -
                      np.sum(subpop_model.params.total_pop_age_risk)) < 1e-6


@pytest.mark.parametrize("transition_type", binomial_random_transition_types_list)
@pytest.mark.parametrize("inputs_id", inputs_id_list)
def test_subpop_reset_reproducible_results(make_subpop_model, transition_type, inputs_id):
    """
    Resetting the random number generator and simulating should
        give the same results as the initial run.
    """

    subpop_model = make_subpop_model("subpop_model", transition_type, case_id_str = inputs_id)

    subpop_model.modify_random_seed(123456789123456789)
    subpop_model.simulate_until_day(100)

    original_model_history_dict = {}

    for name, compartment in subpop_model.compartments.items():
        original_model_history_dict[name] = \
            copy.deepcopy(compartment.history_vals_list)

    reset_model_history_dict = {}

    subpop_model.reset_simulation()
    subpop_model.modify_random_seed(123456789123456789)
    subpop_model.simulate_until_day(100)

    for name, compartment in subpop_model.compartments.items():
        reset_model_history_dict[name] = \
            copy.deepcopy(compartment.history_vals_list)

    for name in subpop_model.compartments.keys():
        assert np.array_equal(np.array(original_model_history_dict[name]),
                              np.array(reset_model_history_dict[name]))


@pytest.mark.parametrize("transition_type", binomial_transition_types_list)
def test_compartments_integer_population(make_subpop_model, transition_type):
    """
    Compartment populations should be integer-valued.
    """

    subpop_model = make_subpop_model("subpop_model", transition_type)

    for day in [1, 10, 100]:
        subpop_model.simulate_until_day(day)

        for compartment in subpop_model.compartments.values():
            assert (compartment.current_val ==
                    np.asarray(compartment.current_val, dtype=int)).all()


@pytest.mark.parametrize("transition_type", binomial_transition_types_list)
def test_transition_format(make_subpop_model, transition_type):
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

    subpop_model = make_subpop_model("subpop_model", transition_type)

    A = subpop_model.params.num_age_groups
    L = subpop_model.params.num_risk_groups

    for day in [1, 10, 100]:
        subpop_model.simulate_until_day(day)

        for tvar in subpop_model.transition_variables.values():
            assert np.shape(tvar.current_rate) == (A, L)
            assert np.shape(tvar.current_val) == (A, L)

            for element in tvar.current_rate.flatten():
                assert isinstance(element, float)


# See note at beginning of document on why taylor approximation is excluded --
#   it's badly behaved for only 1 timestep per day (p < 0 or p > 1 for the
#   binomial probability -- so the test could "fail" for reasons unrelated to the actual test
binomial_no_taylor_transition_types_list = [clt.TransitionTypes.BINOMIAL,
                                            clt.TransitionTypes.BINOMIAL_DETERMINISTIC]


@pytest.mark.parametrize("transition_type", binomial_no_taylor_transition_types_list)
@pytest.mark.parametrize("inputs_id", inputs_id_list)
def test_metapop_no_travel(make_subpop_model, transition_type, inputs_id):
    """
    If two subpopulations comprise a MetapopModel (travel model), then
    if there is no travel between the two subpopulations, the
    MetapopModel should behave exactly like two INDEPENDENTLY RUN
    versions of the SubpopModel instances.

    We can "turn travel off" in multiple ways:
    - Setting pairwise travel proportions to 0 (so that 0% of
        subpopulation i travels to subpopulation j, for each
        distinct i,j subpopulation pair, i != j)
    - Or setting the mobility_modifier to 0 for each
        subpopulation
    We test both of these options, one at a time

    Note -- this test will only pass when timesteps_per_day on
    each SimulationSettings is 1. This is because, for the sake of efficiency,
    for MetapopModel instances, each InteractionTerm is updated
    only ONCE PER DAY rather than after every single discretized timestep.
    In contrast, independent SubpopModel instances (not linked by any
    metapopulation/travel model) do not have any interaction terms.
    The S_to_E transition variable rate does not depend on any
    interaction terms, and depends on state variables that get updated
    at every discretized timestep.
    """

    subpopA = make_subpop_model("A", transition_type, timesteps_per_day = 1, case_id_str = inputs_id)
    subpopB = make_subpop_model("B", transition_type, num_jumps = 1, timesteps_per_day = 1, case_id_str = inputs_id)

    metapopAB_model = flu.FluMetapopModel([subpopA, subpopB],
                                          flu.FluMixingParams(travel_proportions=np.zeros((2, 2)),
                                                              num_locations=2))

    metapopAB_model.simulate_until_day(1)

    subpopA_independent = make_subpop_model("A_independent", transition_type, timesteps_per_day = 1, case_id_str = inputs_id)
    subpopB_independent = make_subpop_model("B_independent", transition_type, num_jumps = 1, timesteps_per_day = 1, case_id_str = inputs_id)

    subpopA_independent.simulate_until_day(1)
    subpopB_independent.simulate_until_day(1)

    check_state_variables_same_history(subpopA, subpopA_independent)
    check_state_variables_same_history(subpopB, subpopB_independent)
