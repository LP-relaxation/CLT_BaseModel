# Test suite for experiments.py in clt_base
# Models are simulated deterministically (without microstochastics,
#   i.e. random transitions between compartments)

from flu_model import flu_components as flu
import clt_base as clt

import numpy as np
import pandas as pd
import pytest

base_path = clt.utils.PROJECT_ROOT / " / "test_input_files"

#################################################
#################### SETUP ######################
#################################################

params_filepath = base_path / "common_params.json"
compartments_epi_metrics_init_vals_filepath = base_path / "init_vals.json"
calendar_filepath = base_path / "school_work_calendar.csv"
humidity_filepath = base_path / "absolute_humidity_austin_2023_2024.csv"

simulation_settings_dict = {}
simulation_settings_dict["timesteps_per_day"] = 2
simulation_settings_dict["transition_type"] = "binomial_deterministic"
simulation_settings_dict["start_real_date"] = "2022-08-08"
simulation_settings_dict["save_daily_history"] = False

compartments_epi_metrics_dict = clt.load_json_new_dict(compartments_epi_metrics_init_vals_filepath)
params_dict = clt.load_json_new_dict(params_filepath)
calendar_df = pd.read_csv(calendar_filepath, index_col=0)

bit_generator = np.random.MT19937(88888)
jumped_bit_generator = bit_generator.jumped(1)
twice_jumped_bit_generator = jumped_bit_generator.jumped(1)

subpopA = flu.FluSubpopModel(compartments_epi_metrics_dict,
                             params_dict,
                             simulation_settings_dict,
                             calendar_df,
                             np.random.Generator(bit_generator),
                             humidity_filepath,
                             name="subpopA")

subpopB = flu.FluSubpopModel(compartments_epi_metrics_dict,
                             params_dict,
                             simulation_settings_dict,
                             calendar_df,
                             np.random.Generator(jumped_bit_generator),
                             humidity_filepath,
                             name="subpopB")

metapopAB = flu.FluMetapopModel([subpopA, subpopB])

# Test on both SubpopModel and on MetapopModel
experiment_models_list = [subpopA, metapopAB]


#################################################
#################### TESTS ######################
#################################################

# Note: for clean up, any files that are created in the
#   experiment test cases are deleted (e.g. database files
#   and csv files for results and inputs).


def test_subpop_sequences_init_vals():
    """
    Tests run_sequences_of_inputs method on `Experiment` class.
    Also confirms that `Experiment` instances correctly modify
    initial values of state variables, i.e. that calling
    `run_sequences_of_inputs` is the same as creating a new
    model for each set of inputs' values and simulating.
    """
    experiment = clt.Experiment(subpopA,
                                ["S", "R"],
                                "results.db")

    # Reset subpopulation model's RNG
    subpopA.RNG = np.random.Generator(np.random.MT19937(88888))

    inputs_dict = {"subpopA": {"S": [int(2e6)]}}

    experiment.run_sequences_of_inputs(num_reps=1,
                                       simulation_end_day=20,
                                       sequences_of_inputs=inputs_dict,
                                       inputs_filename_suffix="inputs.csv",
                                       results_filename="results.csv")

    results_df = pd.read_csv("results.csv")

    # Reset subpopulation model's RNG
    subpopA.reset_simulation()

    subpopA.RNG = np.random.Generator(np.random.MT19937(88888))

    subpopA.simulation_settings.save_daily_history = True

    subpopA.compartments.S.current_val = np.array([[int(2e6)],[int(2e6)]])

    subpopA.simulate_until_day(20)

    assert (np.asarray(results_df[results_df["state_var_name"] == "S"].groupby("timepoint").sum(numeric_only=True)["value"]) ==
           np.asarray(subpopA.compartments.S.history_vals_list).sum(axis=1).flatten()).all()

    Path("results.db").unlink()
    Path("subpopA_inputs.csv").unlink()
    Path("results.csv").unlink()


def test_subpop_random_sampling_reproducibility():
    """
    Confirm that random sampling is reproducible.
    """

    experiment1 = clt.Experiment(subpopA,
                                 ["IS", "IP", "IA"],
                                 "results1.db")

    spec = {"subpopA": {"beta_baseline": [0.4, 0.6]}}

    experiment1.run_random_inputs(num_reps=1,
                                  simulation_end_day=20,
                                  random_inputs_RNG=np.random.Generator(np.random.MT19937(10)),
                                  random_inputs_spec=spec,
                                  inputs_filename_suffix="inputs1.csv",
                                  results_filename="results1.csv")

    experiment2 = clt.Experiment(subpopA,
                                 ["IS", "IP", "IA"],
                                 "results2.db")

    experiment2.run_random_inputs(num_reps=1,
                                  simulation_end_day=20,
                                  random_inputs_RNG=np.random.Generator(np.random.MT19937(10)),
                                  random_inputs_spec=spec,
                                  inputs_filename_suffix="inputs2.csv",
                                  results_filename="results2.csv")

    inputs1 = pd.read_csv("subpopA_inputs1.csv")
    inputs2 = pd.read_csv("subpopA_inputs2.csv")

    assert (inputs1 == inputs2).all().all()

    results1 = pd.read_csv("results1.csv")
    results2 = pd.read_csv("results2.csv")

    assert (results1 == results2).all().all()

    Path("subpopA_inputs1.csv").unlink()
    Path("subpopA_inputs2.csv").unlink()
    Path("results1.csv").unlink()
    Path("results2.csv").unlink()
    Path("results1.db").unlink()
    Path("results2.db").unlink()


def test_subpop_random_sampling_applies_scalar_to_full_array():
    """
    Confirms that random sampling only draws one scalar
    for each input (parameter or initial value). If the input
    is a multi-dimensional array, its random realization is
    a single sampled scalar value applied uniformly to all elements.
    See `sample_random_inputs()` method in `Experiment` class
    for more information.
    """
    experiment = clt.Experiment(subpopA,
                                ["IS", "IP", "IA"],
                                "results.db")

    spec = {"subpopA": {"E_to_IA_prop": [0.1, 0.1]}}

    experiment.run_random_inputs(num_reps=1,
                                 simulation_end_day=20,
                                 random_inputs_RNG=np.random.Generator(twice_jumped_bit_generator),
                                 random_inputs_spec=spec,
                                 inputs_filename_suffix="inputs.csv")

    assert (subpopA.params.E_to_IA_prop == np.array([[0.1], [0.1]])).all()

    Path("subpopA_inputs.csv").unlink()
    Path("results.db").unlink()


def test_subpop_random_inputs_csv_format():
    """
    Make sure that the saved random samples for inputs
        is correctly formatted in the CSV file.
        Here we do an example with a single `SubpopModel`
        and trivial random sampling where the lower
        and upper bounds are the same, so that the realized
        values are known and constant across replications.
    """
    experiment = clt.Experiment(subpopA,
                                ["IS", "IP", "IA"],
                                "results.db")

    spec = {"subpopA": {"beta_baseline": [10, 10],
                        "inf_immune_wane": [0.0015, 0.0015]}}

    experiment.run_random_inputs(num_reps=10,
                                 simulation_end_day=20,
                                 random_inputs_RNG=np.random.Generator(twice_jumped_bit_generator),
                                 random_inputs_spec=spec,
                                 inputs_filename_suffix="inputs.csv")

    inputs_csv_filename = "subpopA_inputs.csv"

    assert Path(inputs_csv_filename).is_file(), \
        f"Inputs CSV file {inputs_csv_filename} was not created."

    inputs_df = pd.read_csv(inputs_csv_filename)

    # Confirm columns are correct
    assert (inputs_df.columns == ["rep", "beta_baseline", "inf_immune_wane"]).all()

    # Confirm sampled values of beta_baseline and inf_immune_wane
    assert (inputs_df["beta_baseline"] == 10).all()
    assert (inputs_df["inf_immune_wane"] == 0.0015).all()

    Path("results.db").unlink()
    Path(inputs_csv_filename).unlink()


def test_metapop_random_inputs_csv_format():
    """
    Analogous to test_subpop_random_inputs_csv_format()
        except for `MetapopModel` instances.

    Make sure that the saved random samples for inputs
        is correctly formatted in the CSV file.
        Here we do an example with a two `SubpopModel`
        instances and trivial random sampling where the lower
        and upper bounds are the same, so that the realized
        values are known and constant across replications.
    """

    experiment = clt.Experiment(metapopAB,
                                ["IS", "IP", "IA"],
                                "results.db")

    spec = {"subpopA": {"beta_baseline": [10, 10],
                        "inf_immune_wane": [0.0015, 0.0015]},
            "subpopB": {"H_to_D_rate": [0.03, 0.03]}}

    experiment.run_random_inputs(num_reps=10,
                                 simulation_end_day=20,
                                 random_inputs_RNG=np.random.Generator(twice_jumped_bit_generator),
                                 random_inputs_spec=spec,
                                 inputs_filename_suffix="inputs.csv")

    for inputs_csv_filename in ["subpopA_inputs.csv", "subpopB_inputs.csv"]:
        assert Path(inputs_csv_filename).is_file(), \
            f"Inputs CSV file {inputs_csv_filename} was not created."

    subpopA_inputs_df = pd.read_csv("subpopA_inputs.csv")

    # Confirm columns are correct
    assert (subpopA_inputs_df.columns == ["rep", "beta_baseline", "inf_immune_wane"]).all()

    # Confirm sampled values of beta_baseline and inf_immune_wane
    assert (subpopA_inputs_df["beta_baseline"] == 10).all()
    assert (subpopA_inputs_df["inf_immune_wane"] == 0.0015).all()

    subpopB_inputs_df = pd.read_csv("subpopB_inputs.csv")

    # Confirm columns are correct
    assert (subpopB_inputs_df.columns == ["rep", "H_to_D_rate"]).all()

    # Confirm sampled values of beta_baseline and inf_immune_wane
    assert (subpopB_inputs_df["H_to_D_rate"] == .03).all()

    Path("subpopA_inputs.csv").unlink()
    Path("subpopB_inputs.csv").unlink()
    Path("results.db").unlink()


@pytest.mark.parametrize("experiment_model", experiment_models_list)
def test_dataframe_query_aggregation(experiment_model):
    """
    Make sure that results database and results CSV files are created.
    """

    experiment = clt.Experiment(experiment_model,
                                ["E", "pop_immunity_inf"],
                                "results.db")
    # Experiment still runs if days_between_save_history > simulation_end_day
    #   because results are still recorded on last day
    experiment.run_static_inputs(50, 5, 10)

    total_exposed = experiment.get_state_var_df("E")
    age0_exposed = experiment.get_state_var_df("E", age_group=0)
    age1_exposed = experiment.get_state_var_df("E", age_group=1)
    assert (age0_exposed + age1_exposed == total_exposed).all().all()

    # There is only 1 risk group, so risk_group=0 filter returns
    #   empty DataFrame
    risk0_exposed = experiment.get_state_var_df("E", risk_group=0)
    risk1_exposed = experiment.get_state_var_df("E", risk_group=1)

    assert risk1_exposed.empty
    assert (risk0_exposed == total_exposed).all().all()

    Path("results.db").unlink()


@pytest.mark.parametrize("experiment_model", experiment_models_list)
def test_csv_db_creation(experiment_model):
    """
    Make sure that results database and results CSV files
    (from `run_static_inputs`) are created.
    """

    experiment = clt.Experiment(experiment_model,
                                ["H", "D"],
                                "results.db")
    experiment.run_static_inputs(10, 100, 7, "results.csv")

    database_filepath = Path("results.db")
    assert database_filepath.is_file(), \
        f"Results database file {database_filepath} was not created."
    database_filepath.unlink()

    csv_filepath = Path("results.csv")
    assert csv_filepath.is_file(), \
        f"Results CSV file {csv_filepath} was not created."
    csv_filepath.unlink()
