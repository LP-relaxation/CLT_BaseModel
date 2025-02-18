from .utils import np, sc, Optional, List, sqlite3, functools, os, pd
from .base_components import SubpopModel, MetapopModel


# Connect to the SQLite database (creates database file if it doesn't exist)
# Create a cursor object to execute SQL commands
# Create a table (if it doesn't exist already)
# Commit changes and close the connection


class ExperimentError(Exception):
    """Custom exceptions for experiment errors."""
    pass


class Results:

    def __init__(self,
                 name: str):
        self.name = name
        self.column_names = ("subpop_name", "state_var_name",
                             "age_group", "risk_group",
                             "rep", "timepoint")
        self.df = pd.DataFrame()

        if os.path.exists(name + ".db"):
            raise ExperimentError("Database already exists! To avoid accidental "
                                  "overwriting, Results instances create new databases. "
                                  "Delete existing .db file or rename Experiment/Results "
                                  "instance to unique name.")

        conn = sqlite3.connect(name + ".db")
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            subpop_name TEXT,
            state_var_name TEXT,
            age_group INT,
            risk_group INT,
            rep INT,
            timepoint INT,
            value FLOAT,
            PRIMARY KEY (subpop_name, state_var_name, age_group, risk_group, rep, timepoint)
        )
        """)
        conn.commit()
        conn.close()


class Experiment:

    def __init__(self,
                 name: str,
                 metapop_model: MetapopModel,
                 state_variables_to_record: list):

        """
        Params:
            name (str):
            metapop_model (MetapopModel):
            state_variables_to_record (List[str]):
        """

        self.name = name
        self.metapop_model = metapop_model
        self.state_variables_to_record = state_variables_to_record

        # Results will be stored as objdict of odicts
        self.results = Results(name)

        for subpop_name, subpop_model in metapop_model.subpop_models.items():
            for svar_name in state_variables_to_record:
                if svar_name not in subpop_model.all_state_variables.keys():
                    raise (f"\"{svar_name}\" in \"state_variables_to_record\" list "
                           "is not a state variable on SubpopModel \"{subpop_name}\" -- "
                           "modify \"state_variables_to_record\" and re-initialize "
                           "experiment.")

    def run(self,
            num_reps: int,
            last_simulation_day: int,
            days_between_save_history: int = 1,
            output_csv_filename: str = None):

        """
        Params:
            num_reps (int):
            last_simulation_day (int):
            days_between_save_history (int):
            output_csv_filename (str):
        """

        metapop_model = self.metapop_model
        subpop_models = metapop_model.subpop_models
        state_variables_to_record = self.state_variables_to_record

        results = self.results

        # Override each subpop config's save_daily_history attribute --
        #   set it to False -- because we will manually save history
        #   to results database according to user-defined
        #   days_between_save_history for all subpops
        for subpop_model in subpop_models.values():
            subpop_model.config.save_daily_history = False

        # Connect to SQL database
        conn = sqlite3.connect(self.name + ".db")
        cursor = conn.cursor()

        for rep in range(num_reps):

            metapop_model.reset_simulation()

            current_simulation_day = metapop_model.current_simulation_day

            while current_simulation_day < last_simulation_day:

                metapop_model.simulate_until_day(min(current_simulation_day + days_between_save_history,
                                                     last_simulation_day))

                current_simulation_day = metapop_model.current_simulation_day

                for subpop_name, subpop_model in subpop_models.items():

                    A = subpop_model.params.num_age_groups
                    R = subpop_model.params.num_risk_groups

                    for svar_name in state_variables_to_record:
                        current_val = subpop_model.all_state_variables[svar_name].current_val
                        current_val = current_val.reshape(-1, 1)

                        # (subpop_name, state_var_name, age_group, risk_group, rep, timepoint)
                        data = np.column_stack((np.full((A * R, 1), subpop_name),
                                                np.full((A * R, 1), svar_name),
                                                np.tile(np.arange(A), (R, 1)).reshape(A * R, 1),
                                                np.tile(np.arange(A), R).reshape(-1, 1),
                                                np.full((A * R, 1), rep),
                                                np.full((A * R, 1), current_simulation_day),
                                                current_val)).tolist()

                        cursor.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?)", data)

        # To avoid memory issues, read in 10k rows at a time into pandas DataFrame
        chunk_size = int(1e4)
        chunks = []

        for chunk in pd.read_sql_query("SELECT * FROM results", conn, chunksize=chunk_size):
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)

        results.df = df

        if output_csv_filename:
            df.to_csv(output_csv_filename)

        conn.commit()
        conn.close()

    def get_state_var_df(self,
                         state_var_name: str,
                         subpop_name: str = None,
                         age_group: int = None,
                         risk_group: int = None,
                         output_csv_filename: str = None):

        conn = sqlite3.connect(self.name + ".db")

        df = pd.read_sql_query(
            """
            SELECT *
            FROM results 
            WHERE state_var_name = ?
            """,
            conn,
            params=(state_var_name))

        conn.close()

        # Define filter conditions
        filters = {
            "subpop_name": subpop_name,
            "age_group": age_group,
            "risk_group": risk_group
        }

        # Build a list of conditions, ignoring None or empty values
        conditions = [(df[col] == value) for col, value in filters.items() if value]

        # If no conditions exist, return the original DataFrame
        df_filtered = df if not conditions else df[np.logical_and.reduce(conditions)]

        df_final = \
            df_filtered.groupby(["rep",
                                 "timepoint"]).sum(numeric_only=True)["value"].reset_index().pivot(index="rep",
                                                                                                   columns="timepoint",
                                                                                                   values="value")

        if output_csv_filename:
            df_final.to_csv(output_csv_filename)

        return df_final

    def run_random_inputs(self,
                          num_reps: int,
                          last_simulation_day: int,
                          state_variables_to_record: List[str],
                          random_inputs_RNG: np.random.Generator,
                          days_between_save_history: int = 1):
        """

        Params:
            num_reps (int):
            last_simulation_day (int):
            state_variables_to_record (List[str]):
            random_inputs_RNG (np.random.Generator):
            days_between_save_history (int):
            simulation_RNG (np.random.Generator):
        """
        pass

    def run_sequence_inputs(self,
                            last_simulation_day: int,
                            num_reps):
        pass

    def sample_random_inputs(self):
        pass
