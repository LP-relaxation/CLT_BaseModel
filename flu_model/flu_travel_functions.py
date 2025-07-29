import torch
import numpy as np

from flu_data_structures import FluMetapopStateTensors, \
    FluMetapopParamsTensors, FluPrecomputedTensors


def compute_wtd_infected(state: FluMetapopStateTensors, params: FluMetapopParamsTensors) -> torch.Tensor:
    return params.IA_relative_inf * state.IA + params.IP_relative_inf + state.IP + state.IS


def compute_wtd_infectious_LA(state: FluMetapopStateTensors,
                              params: FluMetapopParamsTensors) -> torch.Tensor:
    """
    Returns L x A array -- summed over risk groups
    """

    # Einstein notation here means sum over risk groups
    IS = torch.einsum("lar->la", state.IS)
    wtd_IP = \
        params.IP_relative_inf * torch.einsum("lar->la", state.IP)
    wtd_IA = \
        params.IA_relative_inf * torch.einsum("lar->la", state.IA)

    return IS + wtd_IP + wtd_IA


def compute_active_pop_LAR(state: FluMetapopStateTensors,
                           _params: FluMetapopParamsTensors,
                           precomputed: FluPrecomputedTensors) -> torch.Tensor:
    """
    Active population refers to those who are not
        symptomatic infectious or hospitalized (i.e. to
        represent people who are healthy enough to be
        out and about)
    """

    # _params is not used now -- but this is included for
    #   function signature consistency with other
    #   similar computation functions

    return precomputed.total_pop_LAR - state.IS - state.H


def compute_effective_pop_LA(state: FluMetapopStateTensors,
                             params: FluMetapopParamsTensors,
                             precomputed: FluPrecomputedTensors) -> torch.Tensor:

    active_pop_LAR = compute_active_pop_LAR(state, params, precomputed)

    # Nonlocal travel proportions is L x L
    # Active population LAR is L x A x R
    outside_visitors_LAR = torch.einsum("kl,kar->lar",
                                        precomputed.nonlocal_travel_prop,
                                        active_pop_LAR)

    # This is correct -- Dave checked in meeting -- we don't need Einstein
    #   notation here!
    # In computation, broadcast sum_residents_nonlocal_travel_prop to be L x 1 x 1
    traveling_residents_LAR = precomputed.sum_residents_nonlocal_travel_prop[:, None, None] * \
                              active_pop_LAR

    effective_pop_LA = precomputed.total_pop_LA + \
                       params.prop_time_away[0, :, 0] * \
                       torch.sum(outside_visitors_LAR + traveling_residents_LAR, dim=2)

    return effective_pop_LA


def compute_wtd_infectious_ratio_LLA(state: FluMetapopStateTensors,
                                     params: FluMetapopParamsTensors,
                                     precomputed: FluPrecomputedTensors) -> torch.Tensor:
    """
    Returns L x L x A array -- element i,j,a corresponds to
        ratio of weighted infectious people in location i, age group a
        (summed over risk groups) to the effective population in location j
        (summed over risk groups) -- where, again, "effective" population
        means that the population accounts for travelers and symptomatic
        and hospitalized individuals
    """

    wtd_infectious_LA = compute_wtd_infectious_LA(state, params)

    effective_pop_LA = compute_effective_pop_LA(state, params, precomputed)

    prop_wtd_infectious = torch.einsum("ka,la->kla",
                                       wtd_infectious_LA,
                                       1 / effective_pop_LA)

    return prop_wtd_infectious


def compute_raw_local_to_local_foi(prop_time_away: torch.Tensor,
                                   total_contact_matrix: torch.Tensor,
                                   sum_residents_nonlocal_travel_prop: torch.Tensor,
                                   wtd_infectious_ratio_LLA: torch.Tensor,
                                   location_ix: int) -> torch.Tensor:
    """
    Raw means that this is unnormalized by `relative_suscept`

    Excludes beta -- that is factored in later
    """

    result = (1 - prop_time_away.squeeze()[location_ix, :] * sum_residents_nonlocal_travel_prop[location_ix]) * \
             torch.matmul(total_contact_matrix[location_ix, :, :],
                          wtd_infectious_ratio_LLA[location_ix, location_ix, :])

    return result


def compute_raw_outside_visitors_foi(prop_time_away: torch.Tensor,
                                     total_contact_matrix: torch.Tensor,
                                     travel_proportions_array: torch.Tensor,
                                     wtd_infectious_ratio_LLA: torch.Tensor,
                                     local_ix: int,
                                     visitors_ix: int) -> torch.Tensor:
    """
    Computes raw (unnormalized by `relative_suscept`) force
        of infection to local_ix, due to outside visitors from
        visitors_ix

    Excludes beta -- that is factored in later

    Output should be size A
    """

    # In location dest_ix, we are looking at the visitors from
    #   origin_ix who come to dest_ix (and infect folks in dest_ix)

    result = travel_proportions_array[visitors_ix, local_ix] * \
             torch.matmul(prop_time_away.squeeze()[visitors_ix, :] * total_contact_matrix[local_ix, :, :],
                          wtd_infectious_ratio_LLA[visitors_ix, local_ix, :])

    return result


def compute_raw_residents_traveling_foi(prop_time_away: torch.Tensor,
                                        total_contact_matrix: torch.Tensor,
                                        travel_proportions_array: torch.Tensor,
                                        wtd_infectious_ratio_LLA: torch.Tensor,
                                        local_ix: int,
                                        dest_ix: int) -> torch.Tensor:
    """
    Computes raw (unnormalized by `relative_suscept`) force
        of infection to local_ix, due to residents of local_ix
        traveling to dest_ix and getting infected in dest_ix

    Excludes beta -- that is factored in later

    Output should be size A
    """

    result = prop_time_away.squeeze()[local_ix, :] * travel_proportions_array[local_ix, dest_ix] * \
             torch.matmul(total_contact_matrix[local_ix, :, :],
                          wtd_infectious_ratio_LLA[dest_ix, dest_ix, :])

    return result


def compute_total_foi(state: FluMetapopStateTensors,
                      params: FluMetapopParamsTensors,
                      precomputed: FluPrecomputedTensors,
                      beta_adjusted: torch.tensor) -> torch.Tensor:
    """
    Compute total force of infection! Includes beta
    """

    L, A, R = precomputed.L, precomputed.A, precomputed.R

    prop_time_away = params.prop_time_away
    total_contact_matrix = params.total_contact_matrix
    travel_proportions_array = params.travel_proportions_array
    sum_residents_nonlocal_travel_prop = precomputed.sum_residents_nonlocal_travel_prop
    wtd_infectious_ratio_LLA = compute_wtd_infectious_ratio_LLA(state, params, precomputed)

    relative_suscept_by_age = params.relative_suscept[0, :, 0]

    foi = torch.tensor(np.zeros((L, A, R)))

    for l in np.arange(L):

        raw_foi = torch.tensor(np.zeros(A))

        # local-to-local force of infection
        raw_foi = raw_foi + compute_raw_local_to_local_foi(prop_time_away,
                                                           total_contact_matrix,
                                                           sum_residents_nonlocal_travel_prop,
                                                           wtd_infectious_ratio_LLA,
                                                           l)

        for k in np.arange(L):
            raw_foi = raw_foi + compute_raw_outside_visitors_foi(prop_time_away,
                                                                 total_contact_matrix,
                                                                 travel_proportions_array,
                                                                 wtd_infectious_ratio_LLA,
                                                                 l,
                                                                 k)

            raw_foi = raw_foi + compute_raw_residents_traveling_foi(prop_time_away,
                                                                    total_contact_matrix,
                                                                    travel_proportions_array,
                                                                    wtd_infectious_ratio_LLA,
                                                                    l,
                                                                    k)

        normalized_foi = relative_suscept_by_age * raw_foi

        foi[l, :, :] = normalized_foi.view(A, 1).expand((A, R))

    return beta_adjusted * foi