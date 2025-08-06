import torch
import numpy as np
import clt_base as clt
import flu_core as flu
from flu_fixtures import subpop_inputs, make_subpop_model


def test_size_travel_computations(make_subpop_model):

    subpopA = make_subpop_model("A", clt.TransitionTypes.BINOMIAL_DETERMINISTIC, timesteps_per_day = 1)
    subpopB = make_subpop_model("B", clt.TransitionTypes.BINOMIAL_DETERMINISTIC, num_jumps = 1, timesteps_per_day = 1)

    metapopAB_model = flu.FluMetapopModel([subpopA, subpopB],
                                          {"travel_proportions": np.zeros((2, 2)),
                                           "num_locations": 2})

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
                                                       params.prop_time_away,
                                                       precomputed.sum_residents_nonlocal_travel_prop,
                                                       wtd_infectious_ratio_LLA,
                                                       i).size() == torch.Size([A])

            for j in range(L):
                assert flu.compute_outside_visitors_exposure(state.flu_contact_matrix,
                                                             params.prop_time_away,
                                                             params.travel_proportions,
                                                             wtd_infectious_ratio_LLA,
                                                             i,
                                                             j).size() == torch.Size([A])

                assert flu.compute_residents_traveling_exposure(state.flu_contact_matrix,
                                                                params.prop_time_away,
                                                                params.travel_proportions,
                                                                wtd_infectious_ratio_LLA,
                                                                i,
                                                                j).size() == torch.Size([A])

            assert flu.compute_total_mixing_exposure(state, params, precomputed).size() == torch.Size([L, A, R])
