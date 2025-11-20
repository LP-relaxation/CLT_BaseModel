# Metapopulation tests
# For both object-oriented version and pytorch tensor version

import flu_core as flu
import clt_toolkit as clt

import torch
import numpy as np
import pandas as pd
import copy
import pytest

from dataclasses import fields

from helpers import binom_transition_types_list, binom_random_transition_types_list, \
    binom_no_taylor_transition_types_list, inputs_id_list, check_state_variables_same_history
from clt_toolkit import updated_dataclass


def test_metapop_params_tensors_subpop_indexing(make_flu_metapop_model):

    """
    Confirm that lth element indeed refers to lth subpopulation
        in _subpop_models_ordered
    """

    oop_model = make_flu_metapop_model("binom_deterministic_no_round")
    d = oop_model.get_flu_torch_inputs()

    params_tensors = d["params_tensors"]

    # Note: may want to expand this test to check more than one parameter
    assert np.all(np.asarray(params_tensors.total_pop_age_risk[0]) ==
                  oop_model._subpop_models_ordered[0].params.total_pop_age_risk)
    assert np.all(np.asarray(params_tensors.total_pop_age_risk[1]) ==
                  oop_model._subpop_models_ordered[1].params.total_pop_age_risk)


def test_oop_and_torch_agree(make_flu_metapop_model):

    """
    Make sure that the object-oriented version of the flu model (in `flu_components.py`)
    agrees with the functional, torch (tensor-ized) version of the flu model
    (in `flu_torch_det_components.py`). They should be different implementations of the
    same model and give the same result.
    """

    oop_model = make_flu_metapop_model("binom_deterministic_no_round")
    d = oop_model.get_flu_torch_inputs()

    state = d["state_tensors"]
    params = d["params_tensors"]
    schedules = d["schedule_tensors"]
    precomputed = d["precomputed"]

    days_list = [1, 5, 10, 25, 50]
    num_days = max(days_list)

    torch_state_history, torch_tvar_history = flu.torch_simulate_full_history(state,
                                                                              params,
                                                                              precomputed,
                                                                              schedules,
                                                                              num_days,
                                                                              1)
    
    for max_day in days_list:
        print('max_day:', max_day)
        for day in range(max_day):
            oop_model = make_flu_metapop_model("binom_deterministic_no_round")
            oop_model.simulate_until_day(day + 1)
            L = oop_model.precomputed.L
            for subpop_ix in range(L):
                subpop_model = oop_model._subpop_models_ordered[subpop_ix]
                for name, compartment in subpop_model.compartments.items():
                    oop_val = torch.tensor(compartment.history_vals_list[day])
                    assert torch.allclose(oop_val,
                                        torch_state_history[name][day][subpop_ix], rtol=1e-2)
                    
                    # Check vaccine induced immunity epi metrics are the same too
                    oop_MV = torch.tensor(subpop_model.state.MV)
                    torch_MV = torch_state_history['MV'][day][subpop_ix]
                    assert torch.allclose(oop_MV, torch_MV)


@pytest.mark.parametrize("transition_type", binom_no_taylor_transition_types_list)
@pytest.mark.parametrize("inputs_id", inputs_id_list)
def test_metapop_no_travel(make_flu_subpop_model, transition_type, inputs_id):
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

    subpopA = make_flu_subpop_model("A", transition_type, timesteps_per_day = 1, case_id_str = inputs_id)
    subpopB = make_flu_subpop_model("B", transition_type, num_jumps = 1, timesteps_per_day = 1, case_id_str = inputs_id)

    metapopAB_model = flu.FluMetapopModel([subpopA, subpopB],
                                          flu.FluMixingParams(travel_proportions=np.zeros((2, 2)),
                                                              num_locations=2))

    metapopAB_model.simulate_until_day(1)

    subpopA_independent = make_flu_subpop_model("A_independent", transition_type, timesteps_per_day = 1, case_id_str = inputs_id)
    subpopB_independent = make_flu_subpop_model("B_independent", transition_type, num_jumps = 1, timesteps_per_day = 1, case_id_str = inputs_id)

    subpopA_independent.simulate_until_day(1)
    subpopB_independent.simulate_until_day(1)

    check_state_variables_same_history(subpopA, subpopA_independent)
    check_state_variables_same_history(subpopB, subpopB_independent)