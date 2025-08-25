import torch
import numpy as np

from .flu_data_structures import FluTravelStateTensors, \
    FluTravelParamsTensors, FluPrecomputedTensors


# Dimensions
#   L (int):
#       number of locations/subpopulations
#   A (int):
#       number of age groups
#   R (int):
#       number of risk groups

# Suffixes with some combination of the letters "L", "A", "R"
#   can be found after some function and variable names --
#   this is to make the dimensions/indices explicit to help with
#   the tensor computations


def compute_wtd_infectious_LA(state: FluTravelStateTensors,
                              params: FluTravelParamsTensors) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A)):
            Weighted infectious, summed over risk groups:
            includes presymptomatic, asymptomatic, and symptomatic,
            weighted by relative infectiousness
    """

    # Einstein notation here means sum over risk groups
    IS = torch.einsum("lar->la", state.IS)
    wtd_IP = \
        params.IP_relative_inf * torch.einsum("lar->la", state.IP)
    wtd_IA = \
        params.IA_relative_inf * torch.einsum("lar->la", state.IA)

    return IS + wtd_IP + wtd_IA


def compute_active_pop_LAR(state: FluTravelStateTensors,
                           _params: FluTravelParamsTensors,
                           precomputed: FluPrecomputedTensors) -> torch.Tensor:
    """
    Compute the active population for location-age-risk
    (l, a, r) as a tensor. Used to compute the
    effective population in the travel model, which is
    the population size adjusted for incoming visitors,
    residents traveling, and assuming hospitalized
    individuals are not mobile enough to infect others.

    Returns:
        (torch.Tensor of size (L, A, R)):
            Active population: those who are not
            hospitalized (i.e. those who are not too sick
            to move and travel regularly)
    """

    # _params is not used now -- but this is included for
    #   function signature consistency with other
    #   similar computation functions

    return precomputed.total_pop_LAR_tensor - state.H


def compute_effective_pop_LA(state: FluTravelStateTensors,
                             params: FluTravelParamsTensors,
                             precomputed: FluPrecomputedTensors) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A)):
            Effective population, summed over risk groups.
            See `compute_active_pop_LAR` docstring for more
            information.
    """

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

    mobility_modifier = params.mobility_modifier[0, :, 0]

    effective_pop_LA = precomputed.total_pop_LA + mobility_modifier * \
                       torch.sum(outside_visitors_LAR + traveling_residents_LAR, dim=2)

    return effective_pop_LA


def compute_wtd_infectious_ratio_LLA(state: FluTravelStateTensors,
                                     params: FluTravelParamsTensors,
                                     precomputed: FluPrecomputedTensors) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, L, A):
            Element i,j,a corresponds to ratio of weighted infectious people
            in location i, age group a (summed over risk groups) to the effective
            population in location j (summed over risk groups)
    """

    wtd_infectious_LA = compute_wtd_infectious_LA(state, params)

    effective_pop_LA = compute_effective_pop_LA(state, params, precomputed)

    prop_wtd_infectious = torch.einsum("ka,la->kla",
                                       wtd_infectious_LA,
                                       1 / effective_pop_LA)

    return prop_wtd_infectious


def compute_local_to_local_exposure(flu_contact_matrix: torch.Tensor,
                                    mobility_modifier: torch.Tensor,
                                    sum_residents_nonlocal_travel_prop: torch.Tensor,
                                    wtd_infectious_ratio_LLA: torch.Tensor,
                                    location_ix: int) -> torch.Tensor:
    """
    Raw means that this is unnormalized by `relative_suscept`.
    Excludes beta and population-level immunity adjustments --
    those are factored in later.

    Returns:
        (torch.Tensor of size (A)):
            For a given location (specified by `location_ix`), compute
            local transmission caused by residents traveling within their
            home location, summed over risk groups.
    """

    # WARNING: we assume `mobility_modifier` is input as (A, 1) --
    # if this changes, we have to change the implementation.
    # The risk dimension does not have unique values, so we just
    # grab the first element of the risk dimension.
    mobility_modifier = mobility_modifier[location_ix, :, 0]

    result = (1 - mobility_modifier * sum_residents_nonlocal_travel_prop[location_ix]) * \
             torch.matmul(flu_contact_matrix[location_ix, :, :],
                          wtd_infectious_ratio_LLA[location_ix, location_ix, :])

    return result


def compute_outside_visitors_exposure(flu_contact_matrix: torch.Tensor,
                                      mobility_modifier: torch.Tensor,
                                      travel_proportions: torch.Tensor,
                                      wtd_infectious_ratio_LLA: torch.Tensor,
                                      local_ix: int,
                                      visitors_ix: int) -> torch.Tensor:
    """
    Computes raw (unnormalized by `relative_suscept`) transmission
    to `local_ix` due to outside visitors from `visitors_ix`.
    Excludes beta and population-level immunity adjustments --
    those are factored in later.

    Returns:
        (torch.Tensor of size (A))
    """

    # In location `local_ix`, we are looking at the visitors from
    #   `visitors_ix` who come to `local_ix` (and infect folks in `local_ix`)

    # See WARNING in `compute_local_to_local_exposure()`
    mobility_modifier = mobility_modifier[visitors_ix, :, 0]

    result = travel_proportions[visitors_ix, local_ix] * \
             torch.matmul(mobility_modifier * flu_contact_matrix[local_ix, :, :],
                          wtd_infectious_ratio_LLA[visitors_ix, local_ix, :])

    return result


def compute_residents_traveling_exposure(flu_contact_matrix: torch.Tensor,
                                         mobility_modifier: torch.Tensor,
                                         travel_proportions: torch.Tensor,
                                         wtd_infectious_ratio_LLA: torch.Tensor,
                                         local_ix: int,
                                         dest_ix: int) -> torch.Tensor:
    """
    Computes raw (unnormalized by `relative_suscept`) transmission
    to `local_ix`, due to residents of `local_ix` traveling to `dest_ix`
    and getting infected in `dest_ix`. Excludes beta and population-level
    immunity adjustments -- those are factored in later.

    Returns:
        (torch.Tensor of size (A))
    """

    # See WARNING in `compute_local_to_local_exposure()`
    mobility_modifier = mobility_modifier[local_ix, :, 0]

    result = mobility_modifier * travel_proportions[local_ix, dest_ix] * \
             torch.matmul(flu_contact_matrix[local_ix, :, :],
                          wtd_infectious_ratio_LLA[dest_ix, dest_ix, :])

    return result


def compute_total_mixing_exposure(state: FluTravelStateTensors,
                                  params: FluTravelParamsTensors,
                                  precomputed: FluPrecomputedTensors) -> torch.Tensor:
    """
    Computes "total mixing exposure" for location-age-risk
    (l, a, r) -- the rate of exposure to infectious individuals,
    accounting for both local transmission, incoming visitors, and
    residents traveling. **Normalized by `relative_suscept`!**

    Combines subroutines `compute_local_to_local_exposure()`,
    `compute_outside_visitors_exposure()`, and `compute_residents_traveling_exposure()`.
    Note that these subroutines do not include relative susceptibility --
    but this function includes relative susceptibility -- this is to avoid
    unnecessary repeated multiplication by relative susceptible in each subroutine.

    Returns:
        (torch.Tensor of size (L, A, R))
    """

    L, A, R = precomputed.L, precomputed.A, precomputed.R

    mobility_modifier = params.mobility_modifier
    flu_contact_matrix = state.flu_contact_matrix
    travel_proportions = params.travel_proportions
    sum_residents_nonlocal_travel_prop = precomputed.sum_residents_nonlocal_travel_prop
    wtd_infectious_ratio_LLA = compute_wtd_infectious_ratio_LLA(state, params, precomputed)

    relative_suscept = params.relative_suscept[0, :, 0]

    total_mixing_exposure = torch.tensor(np.zeros((L, A, R)))

    # Couldn't figure out how to do this without two for-loops ;)
    # Welcoming any efficiency improvements!
    for l in np.arange(L):

        raw_total_mixing_exposure = torch.tensor(np.zeros(A))

        raw_total_mixing_exposure = raw_total_mixing_exposure + \
                                    compute_local_to_local_exposure(flu_contact_matrix,
                                                                    mobility_modifier,
                                                                    sum_residents_nonlocal_travel_prop,
                                                                    wtd_infectious_ratio_LLA,
                                                                    l)

        for k in np.arange(L):
            raw_total_mixing_exposure = raw_total_mixing_exposure + \
                                        compute_outside_visitors_exposure(
                                            flu_contact_matrix,
                                            mobility_modifier,
                                            travel_proportions,
                                            wtd_infectious_ratio_LLA,
                                            l,
                                            k)

            raw_total_mixing_exposure = raw_total_mixing_exposure + \
                                        compute_residents_traveling_exposure(
                                            flu_contact_matrix,
                                            mobility_modifier,
                                            travel_proportions,
                                            wtd_infectious_ratio_LLA,
                                            l,
                                            k)

        normalized_total_mixing_exposure = relative_suscept * raw_total_mixing_exposure

        total_mixing_exposure[l, :, :] = normalized_total_mixing_exposure.view(A, 1).expand((A, R))

    return total_mixing_exposure
