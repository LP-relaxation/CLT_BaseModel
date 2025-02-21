from .utils import np, sc, Optional, List, sqlite3, functools, os, pd, fields
from .base_components import SubpopModel, MetapopModel


class ExperimentError(Exception):
    """Custom exceptions for experiment errors."""
    pass


def check_is_subset_list(listA: list,
                         listB: list) -> bool:
    """
    Returns True if listA is a subset of listB,
    and False otherwise.

    Params:
        listA (list):
            list-like of elements to check if
            subset of listB.
        listB (list):
            list-like of elements.
    """
    return all(item in listB for item in listA)


def get_sql_table_as_df(conn: sqlite3.Connection,
                        sql_query: str,
                        sql_query_params: tuple[str] = None,
                        chunk_size: int = int(1e4)) -> pd.DataFrame:
    """
    Returns pandas DataFrame with information from
    SQL table with table_name from database given
    by Connection. Reads in SQL rows in batches of size
    chunk_size to avoid memory issues for very large
    tables.

    Params:
        conn (sqlite3.Connection):
            connection to SQL database.
        sql_query (str):
            SQL query/statement to execute on database.
        sql_query_params (tuple[str]):
            tuple of strings to pass as parameters to
            SQL query -- used to avoid SQL injections.
        chunk_size (positive int):
            number of rows to read in at a time.
    :return:
    """

    chunks = []
    for chunk in pd.read_sql_query(sql_query,
                                   conn,
                                   chunksize=chunk_size,
                                   params=sql_query_params):
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    return df


class Results:
    """
    Container for Experiment results -- each Experiment
    instance has a Results instance as an attribute.
    Each Results instance initializes a new SQL database
    to hold results and a pandas DataFrame to hold results
    (this DataFrame is stored on the Results instance).

    Attributes:
        database_filename (str):
            name uniquely identifying Results instance.
        df (pd.DataFrame):
            DataFrame holding simulation results from each
            simulation replication
    """

    def __init__(self,
                 database_filename: str):
        self.database_filename = database_filename
        self.df = None


class Experiment:
    """
    Class to manage running multiple simulation replications
    on a MetapopModel instance and query its results.

    Also allows running a batch of simulation replications on a
    deterministic sequence of values for a given input
    (for example, to see how output changes as a function of
    a given input).

    Also handles random sampling of inputs from a uniform
    distribution.

    NOTE:
        If an input is an |A| x |R| array (for age-risk),
        the current functionality does not support sampling individual
        age-risk elements separately. Instead, a single scalar value
        is sampled at a time for the entire input.
        See sample_random_inputs method for more details.

    Params:
        results (Results):
            container for Experiment results -- gets populated
            after an experiment run.
        random_inputs_realizations (dict):
            dictionary of dictionaries that stores realizations
            of random input sampling -- keys are SubpopModel names,
            values are dictionaries. These second-layer dictionaries'
            keys are strings corresponding to input names (they must match
            names in each SubpopModel's SubpopParams to be valid)
            and values are list-like, where the ith element
            corresponds to the ith random sample for that input.

    See __init__ docstring for other attributes.
    """

    def __init__(self,
                 name: str,
                 metapop_model: MetapopModel,
                 state_variables_to_record: list,
                 database_filename: str):

        """
        Params:
            name (str):
                name uniquely identifying Experiments instance.
            metapop_model (MetapopModel):
                MetapopModel instance on which to run multiple
                replications.
            state_variables_to_record (list[str]):
                list or list-like of strings corresponding to
                state variables to record -- each string must match
                a state variable name on each SubpopModel in
                the MetapopModel.
            database_filename (str):
                must be valid filename with suffix ".db" --
                experiment results are saved to this SQL database
        """

        self.name = name
        self.metapop_model = metapop_model
        self.state_variables_to_record = state_variables_to_record

        # Results will be stored as objdict of odicts
        self.results = Results(database_filename)

        # Randomly sampled inputs' realizations are stored in this
        #   dictionary
        self.random_inputs_realizations = {}

        for subpop_name in metapop_model.subpop_models.keys():
            self.random_inputs_realizations[subpop_name] = {}

        # Make sure the state variables to record are valid -- the names
        #   of the state variables to record must match actual state variables
        #   on each SubpopModel
        for subpop_name, subpop_model in metapop_model.subpop_models.items():
            if not check_is_subset_list(state_variables_to_record,
                                        subpop_model.all_state_variables.keys()):
                raise (f"\"state_variables_to_record\" list is not a subset "
                       "of the state variables on SubpopModel \"{subpop_name}\" -- "
                       "modify \"state_variables_to_record\" and re-initialize experiment.")

    def run(self,
            num_reps: int,
            simulation_end_day: int,
            days_between_save_history: int = 1,
            output_csv_filename: str = None):
        """
        Runs the associated MetapopModel for a given number of
        independent replications until simulation_end_day.
        User can specify how often to save the history and
        a CSV file in which to store this history.

        Params:
            num_reps (positive int):
                number of independent simulation replications
                to run in an experiment.
            simulation_end_day (positive int):
                stop simulation at simulation_end_day (i.e. exclusive,
                simulate up to but not including simulation_end_day).
            days_between_save_history (positive int):
                indicates how often to save simulation results.
            output_csv_filename (str):
                if specified, must be valid filename with suffix ".csv" --
                experiment results are saved to this CSV file.
        """

        metapop_model = self.metapop_model
        subpop_models = metapop_model.subpop_models
        state_variables_to_record = self.state_variables_to_record

        database_filename = self.results.database_filename

        # Override each subpop config's save_daily_history attribute --
        #   set it to False -- because we will manually save history
        #   to results database according to user-defined
        #   days_between_save_history for all subpops
        for subpop_model in subpop_models.values():
            subpop_model.config.save_daily_history = False

        # Connect to the SQLite database and create database
        # Create a cursor object to execute SQL commands
        # Initialize a table with columns given by column_names
        # Commit changes and close the connection
        if os.path.exists(database_filename):
            raise ExperimentError("Database already exists! To avoid accidental "
                                  "overwriting, Results instances create new databases. "
                                  "Delete existing .db file or change database_filename "
                                  "attribute.")

        conn = sqlite3.connect(database_filename)
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

        # Loop through replications
        for rep in range(num_reps):

            # Reset MetapopModel between replications
            metapop_model.reset_simulation()

            # Reset current_simulation_day counter
            # Loop through days until simulation_end_day
            current_simulation_day = metapop_model.current_simulation_day

            while current_simulation_day < simulation_end_day:

                metapop_model.simulate_until_day(min(current_simulation_day + days_between_save_history,
                                                     simulation_end_day))

                current_simulation_day = metapop_model.current_simulation_day

                # Loop through each SubpopModel instance in MetapopModel
                for subpop_name, subpop_model in subpop_models.items():

                    A = subpop_model.params.num_age_groups
                    R = subpop_model.params.num_risk_groups

                    # Each state variable's current_val is an A x R numpy array
                    # We need to "unpack" this into an (AxR, 1) numpy array
                    #   and similarly convert all other information (subpop_name,
                    #   state_var_name, etc...) to (AxR, 1) numpy arrays, respectively
                    # Then we add all this to the SQL table as a batch of AxR ROWS
                    for state_var_name in state_variables_to_record:
                        current_val = subpop_model.all_state_variables[state_var_name].current_val

                        # numpy's default is row-major / C-style order
                        # This means the elements are unpacked ROW BY ROW
                        current_val_reshaped = current_val.reshape(-1, 1)

                        # (AxR, 1) column vector of row indices, indicating the original row in current_val
                        #   before reshaping
                        # Each integer in np.arange(A) repeated R times
                        age_group_indices = np.repeat(np.arange(A), R).reshape(-1, 1)

                        # (AxR, 1) column vector of column indices, indicating the original column
                        #   each element belonged to in current_val before reshaping
                        # Repeat np.arange(R) A times
                        risk_group_indices = np.tile(np.arange(R), A).reshape(-1, 1)

                        # (subpop_name, state_var_name, age_group, risk_group, rep, timepoint)
                        data = np.column_stack(
                            (np.full((A * R, 1), subpop_name),
                             np.full((A * R, 1), state_var_name),
                             age_group_indices,
                             risk_group_indices,
                             np.full((A * R, 1), rep),
                             np.full((A * R, 1), current_simulation_day),
                             current_val_reshaped)).tolist()

                        cursor.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?)", data)

        df = get_sql_table_as_df(conn, "SELECT * FROM results", int(1e4))

        self.results.df = df

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

        conn = sqlite3.connect(self.results.database_filename)

        df = get_sql_table_as_df(conn,
                                 "SELECT * FROM results WHERE state_var_name = ?",
                                 int(1e4),
                                 sql_query_params=(state_var_name,))

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
                          simulation_end_day: int,
                          random_inputs_RNG: np.random.Generator,
                          random_inputs_spec: dict,
                          days_between_save_history: int = 1,
                          output_csv_filename: str = None):
        """


        Params:
            num_reps (positive int):
                number of independent simulation replications
                to run in an experiment.
            simulation_end_day (positive int):
                stop simulation at simulation_end_day (i.e. exclusive,
                simulate up to but not including simulation_end_day).
            random_inputs_RNG (np.random.Generator):
                random number generator used to sample random
                inputs -- for reproducibility, it is recommended to
                use a distinct RNG for sampling random inputs different
                from the RNG for simulating the experiment/model.
            random_inputs_spec (dict):
                random inputs' specification -- stores details
                for random input sampling -- keys are strings
                corresponding to input names (they must match
                names in each SubpopModel's SubpopParams to be valid)
                and values are 2-tuples of nonnegative floats corresponding
                to lower and upper bounds for sampling that input from a
                uniform distribution.
            days_between_save_history (positive int):
                indicates how often to save simulation results.
            output_csv_filename (str):
                if specified, must be valid filename with suffix ".csv" --
                experiment results are saved to this CSV file
        """
        pass

    def run_sequence_inputs(self,
                            simulation_end_day: int,
                            num_reps):

        pass

    def sample_random_inputs(self,
                             num_reps: int,
                             random_inputs_RNG: np.random.Generator,
                             random_inputs_spec: dict):
        """
        Randomly and independently samples inputs specified by keys of
        random_inputs_spec according to uniform distribution with lower
        and upper bounds specified by values of random_inputs_spec.
        Stores random realizations in random_inputs_realizations attribute.
        Uses random_inputs_RNG to sample inputs.

        Params:
            num_reps (positive int):
                number of independent simulation replications
                to run in an experiment -- corresponds to number of
                Uniform random variables to draw for each
                state variable.
            random_inputs_RNG (np.random.Generator):
                random number generator used to sample random
                inputs -- for reproducibility, it is recommended to
                use a distinct RNG for sampling random inputs different
                from the RNG for simulating the experiment/model.
            random_inputs_spec (dict):
                random inputs' specification -- stores details
                for random input sampling -- keys are strings
                corresponding to input names (they must match
                names in each SubpopModel's SubpopParams to be valid)
                and values are 2-tuples of nonnegative floats corresponding
                to lower and upper bounds for sampling that input from a
                uniform distribution.

        NOTE:
            If an input is an |A| x |R| array (for age-risk),
            the current functionality does not support sampling individual
            age-risk elements separately. Instead, a single scalar value
            is sampled at a time for the entire input. Consequently,
            if an |A| x |R| input is chosen to be randomly sampled,
            all its elements will have the same sampled value.

            If a user wants to sample some age-risk elements separately,
            they should create new inputs for these elements. "Inputs"
            refers to both parameters (in SubpopParams) and initial values
            of Compartment and EpiMetric instances. For example, if the model
            has a parameter "H_to_R_rate" that is 2x1 (2 age groups, 1 risk group)
            and the user wants to sample each element separately, they should create
            two parameters: "H_to_R_rate_age_group_1" and "H_to_R_rate_age_group_2."
            These should be added to the relevant SubpopParams instance and
            input dictionary/file used to create the SubpopParams instance.
            The user can then specify both parameters to be randomly sampled
            and specify the lower and upper bounds accordingly.

            TODO: allow sampling individual age-risk elements separately
            without creating new parameters for each element.

            (Developer note: the difficulty is not with randomly sampling
            arrays, but rather storing arrays in SQL -- SQL tables only
            support atomic values.)

        """

        random_inputs_realizations = self.random_inputs_realizations

        for subpop_name, subpop_model in self.metapop_model.subpop_models.items():
            state_variables_names_list = list(subpop_model.all_state_variables.keys())
            params_names_list = [field.name for field in fields(subpop_model.params)]

            if not check_is_subset_list(random_inputs_spec[subpop_name],
                                        state_variables_names_list + params_names_list):
                raise (f"\"random_inputs_spec[\"{subpop_name}\"]\" keys are not a subset "
                       "of the state variables and parameters on SubpopModel \"{subpop_name}\" -- "
                       "modify \"random_inputs_spec\" and re-initialize experiment.")

        for subpop_name in self.metapop_model.subpop_models.keys():
            for state_var_name, bounds in random_inputs_spec[subpop_name].items():
                lower_bd = bounds[0]
                upper_bd = bounds[1]
                random_inputs_realizations[subpop_name][state_var_name] = \
                    random_inputs_RNG.uniform(low=lower_bd,
                                              high=upper_bd,
                                              size=num_reps)

        breakpoint()