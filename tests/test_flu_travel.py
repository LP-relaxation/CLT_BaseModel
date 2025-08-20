import torch
import numpy as np
import clt_toolkit as clt
import flu_core as flu
import pytest

from helpers import binom_transition_types_list, binom_random_transition_types_list, \
    binom_no_taylor_transition_types_list, inputs_id_list, check_state_variables_same_history
from conftest import subpop_inputs, make_subpop_model


@pytest.mark.parametrize("transition_type", binom_no_taylor_transition_types_list)
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


def test_size_travel_computations(make_subpop_model):

    subpopA = make_subpop_model("A", clt.TransitionTypes.BINOM_DETERMINISTIC, timesteps_per_day = 1)
    subpopB = make_subpop_model("B", clt.TransitionTypes.BINOM_DETERMINISTIC, num_jumps = 1, timesteps_per_day = 1)

    metapopAB_model = flu.FluMetapopModel([subpopA, subpopB],
                                          flu.FluMixingParams(num_locations=2,
                                                              travel_proportions=np.zeros((2,2))))

    for i in [1, 10, 100]:

        metapopAB_model.simulate_until_day(i)

        params = metapopAB_model.travel_params_tensors
        state = metapopAB_model.travel_state_tensors
        precomputed = metapopAB_model.precomputed

        L, A, R = params.num_locations, params.num_age_groups, params.num_risk_groups

        assert flu.compute_wtd_infectious_LA(state, params).size() == torch.Size([L, A])

        assert flu.compute_active_pop_LAR(state, params, precomputed).size() == torch.Size([L, A, R])

        assert flu.compute_effective_pop_LA(state, params, precomputed).size() == torch.Size([L, A])

        wtd_infectious_ratio_LLA = flu.compute_wtd_infectious_ratio_LLA(state, params, precomputed)

        assert wtd_infectious_ratio_LLA.size() == torch.Size([L, L, A])

        for i in range(L):
            assert flu.compute_local_to_local_exposure(state.flu_contact_matrix,
                                                       params.mobility_modifier,
                                                       precomputed.sum_residents_nonlocal_travel_prop,
                                                       wtd_infectious_ratio_LLA,
                                                       i).size() == torch.Size([A])

            for j in range(L):
                assert flu.compute_outside_visitors_exposure(state.flu_contact_matrix,
                                                             params.mobility_modifier,
                                                             params.travel_proportions,
                                                             wtd_infectious_ratio_LLA,
                                                             i,
                                                             j).size() == torch.Size([A])

                assert flu.compute_residents_traveling_exposure(state.flu_contact_matrix,
                                                                params.mobility_modifier,
                                                                params.travel_proportions,
                                                                wtd_infectious_ratio_LLA,
                                                                i,
                                                                j).size() == torch.Size([A])

            assert flu.compute_total_mixing_exposure(state, params, precomputed).size() == torch.Size([L, A, R])
