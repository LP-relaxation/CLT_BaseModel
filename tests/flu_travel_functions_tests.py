import torch
import numpy as np

from flu_data_structures import FluMetapopStateTensors, \
    FluMetapopParamsTensors, FluPrecomputedTensors
from flu_travel_functions import compute_total_mixing_exposure, \
    compute_wtd_infectious_LA, compute_active_pop_LAR, compute_effective_pop_LA, \
    compute_effective_pop_LA, compute_wtd_infectious_ratio_LLA, compute_local_to_local_exposure, \
    compute_outside_visitors_exposure, compute_residents_traveling_exposure
from flu_components import compute_beta_adjusted


def test_size_travel_computations(state: FluMetapopStateTensors,
                                  params: FluMetapopParamsTensors,
                                  precomputed: FluPrecomputedTensors):

    L, A, R = params.num_locations, params.num_age_groups, params.num_risk_groups

    assert compute_wtd_infectious_LA(state, params).size() == torch.Size([L, A])

    assert compute_active_pop_LAR(state, params).size() == torch.Size([L, A, R])

    assert compute_effective_pop_LA(state, params, precomputed).size() == torch.Size([L, A])

    assert compute_wtd_infectious_ratio_LLA(state, params, precomputed).size() == torch.Size([L, L, A])

    for i in range(L):
        assert compute_local_to_local_exposure(params.prop_time_away,
                                              params.total_contact_matrix,
                                              precomputed.sum_residents_nonlocal_travel_prop,
                                              precomputed.wtd_infectious_ratio_LLA,
                                              i).size() == torch.Size([A])

        for j in range(L):
            assert compute_outside_visitors_exposure(params.prop_time_away,
                                                    params.total_contact_matrix,
                                                    params.travel_proportions,
                                                    precomputed.wtd_infectious_ratio_LLA,
                                                    i,
                                                    j).size() == torch.Size([A])

            assert compute_residents_traveling_exposure(params.prop_time_away,
                                                       params.total_contact_matrix,
                                                       params.travel_proportions,
                                                       precomputed.wtd_infectious_ratio_LLA,
                                                       i,
                                                       j).size() == torch.Size([A])

        # Picked arbitrary timestep counter 10 here
        beta_adjusted = compute_beta_adjusted(state, params, 10)
        assert compute_total_mixing_exposure(state, params, precomputed, beta_adjusted).size() == torch.Size([L, A, R])