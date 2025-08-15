import clt_toolkit as clt
import flu_core as flu
import numpy as np
import pandas as pd
import pytest

base_path = clt.utils.PROJECT_ROOT / "tests" / "test_input_files"


def subpop_inputs(id: str):

    if id == "caseA":
        init_vals_filepath = base_path / "caseA_init_vals.json"
        params_filepath = base_path / "caseA_common_subpop_params.json"
        mixing_params_filepath = base_path / "caseAB_mixing_params.json"
    elif id == "caseB_subpop1":
        init_vals_filepath = base_path / "caseB_subpop1_init_vals.json"
        params_filepath = base_path / "caseB_common_subpop_params.json"
        mixing_params_filepath = base_path / "caseAB_mixing_params.json"
    elif id == "caseB_subpop2":
        init_vals_filepath = base_path / "caseB_subpop2_init_vals.json"
        params_filepath = base_path / "caseB_common_subpop_params.json"
        mixing_params_filepath = base_path / "caseAB_mixing_params.json"

    simulation_settings_filepath = base_path / "simulation_settings.json"
    calendar_filepath = base_path / "school_work_calendar.csv"

    state = clt.make_dataclass_from_json(init_vals_filepath,
                                         flu.FluSubpopState)
    params = clt.make_dataclass_from_json(params_filepath, flu.FluSubpopParams)
    mixing_params = clt.make_dataclass_from_json(mixing_params_filepath, flu.FluMixingParams)
    settings = clt.make_dataclass_from_json(simulation_settings_filepath, flu.SimulationSettings)

    calendar_df = pd.read_csv(calendar_filepath, index_col=0)
    humidity_df = pd.read_csv(base_path / "absolute_humidity_austin_2023_2024.csv", index_col=0)
    vaccines_df = pd.read_csv(base_path / "daily_vaccines_constant.csv", index_col = 0)

    schedules_info = flu.FluSubpopSchedules(absolute_humidity=humidity_df,
                                            flu_contact_matrix=calendar_df,
                                            daily_vaccines=vaccines_df)

    return state, params, mixing_params, \
           settings, schedules_info


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

        init_vals, params, mixing_params, simulation_settings, schedules_info = \
            subpop_inputs(case_id_str)

        simulation_settings = clt.updated_dataclass(simulation_settings, {"timesteps_per_day": timesteps_per_day,
                                                                          "transition_type": transition_type})

        starting_random_seed = 123456789123456789
        bit_generator = np.random.MT19937(starting_random_seed)

        model = flu.FluSubpopModel(init_vals,
                                   params,
                                   simulation_settings,
                                   np.random.Generator(bit_generator.jumped(num_jumps)),
                                   schedules_info,
                                   name)

        return model

    return _make_subpop_model
