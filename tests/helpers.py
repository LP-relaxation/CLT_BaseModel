import clt_toolkit as clt
import numpy as np

binom_transition_types_list = [clt.TransitionTypes.BINOM,
                                  clt.TransitionTypes.BINOM_DETERMINISTIC,
                                  clt.TransitionTypes.BINOM_TAYLOR_APPROX,
                                  clt.TransitionTypes.BINOM_TAYLOR_APPROX_DETERMINISTIC]

binom_random_transition_types_list = [clt.TransitionTypes.BINOM,
                                      clt.TransitionTypes.BINOM_TAYLOR_APPROX]

# See tests readme on why taylor approximation is excluded --
#   it's badly behaved for only 1 timestep per day (p < 0 or p > 1 for the
#   binomial probability -- so the test could "fail" for reasons unrelated to the actual test
binom_no_taylor_transition_types_list = [clt.TransitionTypes.BINOM,
                                         clt.TransitionTypes.BINOM_DETERMINISTIC]


inputs_id_list = ["caseA", "caseB_subpop1"]


def check_state_variables_same_history(subpop_model_A: clt.SubpopModel,
                                       subpop_model_B: clt.SubpopModel):
    for name in subpop_model_A.all_state_variables.keys():
        assert np.array_equal(np.array(subpop_model_A.all_state_variables[name].history_vals_list),
                              np.array(subpop_model_B.all_state_variables[name].history_vals_list))