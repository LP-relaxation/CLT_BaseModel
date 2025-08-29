# Test suite for experiments.py in clt_base
# Models are simulated deterministically (without microstochastics,
#   i.e. random transitions between compartments)

from flu_core import flu_components as flu
import clt_toolkit as clt

import numpy as np
import pandas as pd
import pytest

from pathlib import Path


#################################################
#################### TESTS ######################
#################################################

# Note: for clean up, any files that are created in the
#   experiment test cases are deleted (e.g. database files
#   and csv files for results and inputs).


def test_dataframe_query_aggregation(make_flu_metapop_model):
    """
    Make sure that results database and results CSV files are created.
    """

    metapop_model = make_flu_metapop_model("binom")

    experiment = clt.Experiment(metapop_model,
                                ["E", "M"],
                                "results.db")
    # Experiment still runs if days_between_save_history > simulation_end_day
    #   because results are still recorded on last day
    experiment.run_static_inputs(50, 5, 10)

    total_exposed = experiment.get_state_var_df("E")
    age0_exposed = experiment.get_state_var_df("E", age_group=0)
    age1_exposed = experiment.get_state_var_df("E", age_group=1)
    age2_exposed = experiment.get_state_var_df("E", age_group=2)
    age3_exposed = experiment.get_state_var_df("E", age_group=3)
    age4_exposed = experiment.get_state_var_df("E", age_group=4)
    assert (age0_exposed + age1_exposed + age2_exposed + age3_exposed + age4_exposed
            == total_exposed).all().all()

    # There is only 1 risk group, so risk_group=0 filter returns
    #   empty DataFrame
    risk0_exposed = experiment.get_state_var_df("E", risk_group=0)
    risk1_exposed = experiment.get_state_var_df("E", risk_group=1)

    assert risk1_exposed.empty
    assert (risk0_exposed == total_exposed).all().all()

    Path("results.db").unlink()


def test_csv_db_creation(make_flu_metapop_model):
    """
    Make sure that results database and results CSV files
    (from `run_static_inputs`) are created.
    """

    metapop_model = make_flu_metapop_model("binom")

    experiment = clt.Experiment(metapop_model,
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
