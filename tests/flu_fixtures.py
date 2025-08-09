import clt_base as clt
import flu_core as flu
import numpy as np
import pandas as pd
import pytest

base_path = clt.utils.PROJECT_ROOT / "tests" / "test_input_files"


def subpop_inputs(id: str):

    if id == "caseA":
        init_vals_filepath = base_path / "caseA_init_vals.json"
        params_filepath = base_path / "caseA_common_subpop_params.json"
        mixing_params_filepath = base_path / "caseA_mixing_params.json"
    elif id == "caseB_subpop1":
        init_vals_filepath = base_path / "caseB_subpop1_init_vals.json"
        params_filepath = base_path / "caseB_common_subpop_params.json"
        mixing_params_filepath = base_path / "caseB_mixing_params.json"
    elif id == "caseB_subpop2":
        init_vals_filepath = base_path / "caseB_subpop2_init_vals.json"
        params_filepath = base_path / "caseB_common_subpop_params.json"
        mixing_params_filepath = base_path / "caseB_mixing_params.json"

    config_filepath = base_path / "config.json"
    calendar_filepath = base_path / "school_work_calendar.csv"

    init_vals_dict = clt.load_json_new_dict(init_vals_filepath)
    params_dict = clt.load_json_new_dict(params_filepath)
    mixing_params_dict = clt.load_json_new_dict(mixing_params_filepath)
    config_dict = clt.load_json_new_dict(config_filepath)

    calendar_df = pd.read_csv(calendar_filepath, index_col=0)
    humidity_df = pd.read_csv(base_path / "humidity_austin_2023_2024.csv", index_col=0)
    vaccines_df = pd.read_csv(base_path / "daily_vaccines_constant.csv", index_col = 0)

    schedules_info = {}
    schedules_info["flu_contact_matrix"] = calendar_df
    schedules_info["daily_vaccines"] = vaccines_df
    schedules_info["absolute_humidity"] = humidity_df

    return init_vals_dict, params_dict, mixing_params_dict, \
           config_dict, schedules_info


# Factory function
# Need factory because pytest only runs a fixture once
#   per test! So, to be able to use this function
#   twice (or more) in a test -- for example, to
#   create two models, this actually needs to be
#   a function that returns a function.
# pytest documentation:
#   https://docs.pytest.org/en/6.2.x/fixture.html
# See section named 'Factories as fixtures'
# Also pytest doesn’t allow passing arguments to fixtures
#   like regular functions -- that's why arguments are
#   in the inner function, not the outer function
@pytest.fixture
def make_subpop_model():
    def _make_subpop_model(name: str,
                           transition_type: clt.TransitionTypes = clt.TransitionTypes.BINOMIAL,
                           num_jumps: int = 0,
                           timesteps_per_day: int = 7,
                           case_id_str: str = "caseA"):

        init_vals_dict, params_dict, mixing_params_dict, \
        config_dict, schedules_info = subpop_inputs(case_id_str)

        config_dict["timesteps_per_day"] = timesteps_per_day

        # Modify transition type
        config_dict["transition_type"] = transition_type

        starting_random_seed = 123456789123456789
        bit_generator = np.random.MT19937(starting_random_seed)

        model = flu.FluSubpopModel(init_vals_dict,
                                   params_dict,
                                   config_dict,
                                   np.random.Generator(bit_generator.jumped(num_jumps)),
                                   schedules_info,
                                   name)

        return model

    return _make_subpop_model
