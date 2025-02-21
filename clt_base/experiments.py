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


def format_current_val_for_sql(subpop_model: SubpopModel,
                               state_var_name: str,
                               rep: int) -> list:
    """
    Processes current_val of given subpop_model's StateVariable with state_var_name.
    Current_val is an |A| x |R| numpy array (for age-risk) and
    "unpacks" it into an (|A| x |R|, 1) numpy array (a column vector).
    Converts metadata (subpop_name, state_var_name, rep, and current_simulation_day)
    into list of |A| x |R| rows, where each row has 7 elements, for consistent
    row formatting for batch SQL insertion.

    Params:
        subpop_model (SubpopModel):
            SubpopModel to record.
        state_var_name (str):
            StateVariable name to record.
        rep (int):
            replication counter to record.

    Returns:
        data (list):
            list of |A| x |R| rows, where each row is a list of 7 elements
            corresponding to subpop_name, state_var_name, age_group, risk_group,
            rep, current_simulation_day, and the scalar element of current_val
            corresponding to that age-risk group.
    """

    current_val = subpop_model.all_state_variables[state_var_name].current_val

    A, R = np.shape(current_val)

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
        (np.full((A * R, 1), subpop_model.name),
         np.full((A * R, 1), state_var_name),
         age_group_indices,
         risk_group_indices,
         np.full((A * R, 1), rep),
         np.full((A * R, 1), subpop_model.current_simulation_day),
         current_val_reshaped)).tolist()

    return data


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


class Experiment:
    """
    Class to manage running multiple simulation replications
    on a SubpopModel or MetapopModel instance and query its results.

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
        experiment_subpop_models (tuple):
            tuple of SubpopModel instances associated with the Experiment.
            If the Experiment is for a MetapopModel, then
            this tuple contains all the associated SubpopModel instances
            that comprise that MetapopModel. If the Experiment is for
            a SubpopModel only, then this tuple contains only that
            particular SubpopModel.
        inputs_realizations (dict):
            dictionary of dictionaries that stores user-specified deterministic
            sequences for inputs or realizations of random input sampling --
            keys are SubpopModel names, values are dictionaries.
            These second-layer dictionaries' keys are strings corresponding
            to input names (they must match names in each SubpopModel's
            SubpopParams to be valid) and values are list-like, where the ith
            element corresponds to the ith random sample for that input.
        results_df (pd.DataFrame):
            DataFrame holding simulation results from each
            simulation replication

    See __init__ docstring for other attributes.
    """

    def __init__(self,
                 name: str,
                 model: SubpopModel | MetapopModel,
                 state_variables_to_record: list,
                 database_filename: str):

        """
        Params:
            name (str):
                name uniquely identifying Experiments instance.
            model (SubpopModel | MetapopModel):
                SubpopModel or MetapopModel instance on which to
                run multiple replications.
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
        self.model = model
        self.state_variables_to_record = state_variables_to_record
        self.database_filename = database_filename

        if isinstance(model, MetapopModel):
            experiment_subpop_models = tuple(model.subpop_models.values())
        elif isinstance(model, SubpopModel):
            experiment_subpop_models = (model,)
        else:
            raise ExperimentError("\"model\" argument must be an instance of SubpopModel "
                                  "or MetapopModel class.")
        self.experiment_subpop_models = experiment_subpop_models

        # Initialize results_df attribute -- this will store
        #   results of experiment run
        self.results_df = None

        # User-specified deterministic sequences for inputs or randomly
        #   sampled inputs' realizations are stored in this dictionary
        self.inputs_realizations = {}

        error_message_state_variables_to_record = \
            f"\"state_variables_to_record\" list is not a subset " \
            "of the state variables on SubpopModel \"{subpop_name}\" -- " \
            "modify \"state_variables_to_record\" and re-initialize experiment."

        for subpop_model in self.experiment_subpop_models:
            self.inputs_realizations[subpop_model.name] = {}

            # Make sure the state variables to record are valid -- the names
            #   of the state variables to record must match actual state variables
            #   on each SubpopModel
            if not check_is_subset_list(state_variables_to_record,
                                        subpop_model.all_state_variables.keys()):
                raise ExperimentError(error_message_state_variables_to_record)

    def run_with_static_inputs(self,
                               num_reps: int,
                               simulation_end_day: int,
                               days_between_save_history: int = 1,
                               output_csv_filename: str = None):
        """
        Runs the associated SubpopModel or MetapopModel for a
        given number of independent replications until simulation_end_day.
        User can specify how often to save the history and a CSV file
        in which to store this history.

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

        self.create_results_sql_table()

        self.simulate_and_log_reps(num_reps,
                                   simulation_end_day,
                                   days_between_save_history,
                                   True,
                                   output_csv_filename)

    def get_state_var_df(self,
                         state_var_name: str,
                         subpop_name: str = None,
                         age_group: int = None,
                         risk_group: int = None,
                         output_csv_filename: str = None):
        """
        Get pandas DataFrame of recorded values of StateVariable given by state_var_name,
        in the SubpopModel given by subpop_name, for the age-risk group given by
        age_group and risk_group. If subpop_name is not specified, then values
        are summed across all associated subpopulations. Similarly, if age_group
        (or risk_group) is not specified, then values are summed across all age groups
        (or risk groups).

        Args:
            state_var_name (str):
                Name of the StateVariable to retrieve.
            subpop_name (Optional[str]):
                The name of the SubpopModel for filtering. If None, values are
                summed across all SubpopModel instances.
            age_group (Optional[int]):
                The age group to select. If None, values are summed across
                all age groups.
            risk_group (Optional[int]):
                The risk group to select. If None, values are summed across
                all risk groups.
            output_csv_filename (Optional[str]):
                If provided, saves the resulting DataFrame as a CSV.

        Returns:
            (pd.DataFrame):
                A pandas DataFrame where rows represent the replication and columns indicate the
                simulation day (timepoint) of recording. DataFrame values are the StateVariable's
                current_val or the sum of the StateVariable's current_val across subpopulations,
                age groups, or risk groups (the combination of what is summed over is
                specified by the user -- details are in the part of this docstring describing
                this function's parameters).

        Side Effects:
            - Connects to an SQLite database to retrieve data.
            - Writes output to a CSV file if `output_csv_filename` is specified.
        """

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

    def run_with_random_inputs(self,
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

        self.sample_random_inputs(num_reps,
                                  random_inputs_RNG,
                                  random_inputs_spec)

        self.create_results_sql_table()

        self.create_inputs_realizations_sql_tables()

        self.simulate_and_log_reps(num_reps,
                                   simulation_end_day,
                                   days_between_save_history,
                                   False,
                                   output_csv_filename)

    def run_with_sequences_inputs(self,
                                  num_reps: int,
                                  simulation_end_day: int,
                                  sequences_inputs_spec: dict,
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
            sequences_inputs_spec (dict):
                dictionary of dictionaries that stores user-specified deterministic
                sequences for inputs -- must follow specific structure.
                Keys are SubpopModel names, values are dictionaries.
                These second-layer dictionaries' keys are strings corresponding
                to input names (they must match names in each SubpopModel's
                SubpopParams to be valid) and values are list-like, where the ith
                element corresponds to the ith random sample for that input.
            days_between_save_history (positive int):
                indicates how often to save simulation results.
            output_csv_filename (str):
                if specified, must be valid filename with suffix ".csv" --
                experiment results are saved to this CSV file
        """

        self.inputs_realizations = sequences_inputs_spec

        self.create_results_sql_table()

        self.create_inputs_realizations_sql_tables()

        self.simulate_and_log_reps(num_reps,
                                   simulation_end_day,
                                   days_between_save_history,
                                   False,
                                   output_csv_filename)

    def sample_random_inputs(self,
                             num_reps: int,
                             random_inputs_RNG: np.random.Generator,
                             random_inputs_spec: dict):
        """
        Randomly and independently samples inputs specified by keys of
        random_inputs_spec according to uniform distribution with lower
        and upper bounds specified by values of random_inputs_spec.
        Stores random realizations in inputs_realizations attribute.
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

        inputs_realizations = self.inputs_realizations

        for subpop_name, subpop_model in self.metapop_model.subpop_models.items():

            compartments_names_list = list(subpop_model.compartments.keys())
            epi_metrics_names_list = list(subpop_model.epi_metrics.keys())
            params_names_list = [field.name for field in fields(subpop_model.params)]

            if not check_is_subset_list(random_inputs_spec[subpop_name],
                                        compartments_names_list + epi_metrics_names_list + params_names_list):
                raise (f"\"random_inputs_spec[\"{subpop_name}\"]\" keys are not a subset "
                       "of the state variables and parameters on SubpopModel \"{subpop_name}\" -- "
                       "modify \"random_inputs_spec\" and re-initialize experiment.")

        for subpop_name in self.metapop_model.subpop_models.keys():
            for state_var_name, bounds in random_inputs_spec[subpop_name].items():
                lower_bd = bounds[0]
                upper_bd = bounds[1]
                inputs_realizations[subpop_name][state_var_name] = \
                    random_inputs_RNG.uniform(low=lower_bd,
                                              high=upper_bd,
                                              size=num_reps)

    def log_current_vals_to_sql(self,
                                rep: int,
                                cursor: sqlite3.Cursor) -> None:

        for subpop_model in self.experiment_subpop_models:
            for state_var_name in self.state_variables_to_record:
                data = format_current_val_for_sql(subpop_model,
                                                  state_var_name,
                                                  rep)

                cursor.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?)", data)

    def log_inputs_to_sql(self,
                          cursor: sqlite3.Cursor):

        for subpop_model in self.experiment_subpop_models:
            table_name = f'"{subpop_model.name}_INPUTS"'

            # Get the column names (dynamically, based on table)
            cursor.execute(f"PRAGMA table_info({table_name})")

            columns_info = cursor.fetchall()
            column_names = [col[1] for col in columns_info]  # Extract column names from the table info

            # Create a placeholder string for the dynamic query
            placeholders = ", ".join(["?" for _ in column_names])  # Number of placeholders matches number of columns

            # Create the dynamic INSERT statement
            sql_statement = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({placeholders})"

            subpop_inputs_realizations = self.inputs_realizations[subpop_model.name]

            inputs_vals_over_reps_list = [subpop_inputs_realizations[input_name] for input_name in column_names]

            data = np.column_stack(inputs_vals_over_reps_list).tolist()

            cursor.executemany(sql_statement, data)

    def apply_inputs_to_model(self,
                              rep: int):

        for subpop_model in self.experiment_subpop_models:

            params = subpop_model.params

            for input_name, input_val in self.inputs_realizations[subpop_model.name].items():

                dimensions = (params.num_age_groups, params.num_risk_groups)

                if input_name in subpop_model.all_state_variables.keys():
                    subpop_model.all_state_variables[input_name].current_val = np.full(dimensions,
                                                                                       input_val[rep])
                else:
                    if np.isscalar(subpop_model.params[input_name]):
                        subpop_model.params[input_name] = input_val[rep]
                    else:
                        subpop_model.params[input_name] = np.full(dimensions,
                                                                  input_val[rep])

    def simulate_and_log_reps(self,
                              num_reps: int,
                              simulation_end_day: int,
                              days_between_save_history: int,
                              inputs_are_static: bool,
                              output_csv_filename: str = None):

        # Override each subpop config's save_daily_history attribute --
        #   set it to False -- because we will manually save history
        #   to results database according to user-defined
        #   days_between_save_history for all subpops
        for subpop_model in self.experiment_subpop_models:
            subpop_model.config.save_daily_history = False

        model = self.model

        conn = sqlite3.connect(self.database_filename)
        cursor = conn.cursor()

        # Loop through replications
        for rep in range(num_reps):

            model.reset_simulation()

            if not inputs_are_static:
                self.apply_inputs_to_model(rep)
                self.log_inputs_to_sql(cursor)

            while model.current_simulation_day < simulation_end_day:
                model.simulate_until_day(min(model.current_simulation_day + days_between_save_history,
                                             simulation_end_day))

                self.log_current_vals_to_sql(rep, cursor)

        self.results_df = get_sql_table_as_df(conn, "SELECT * FROM results", int(1e4))

        if output_csv_filename:
            self.results_df.to_csv(output_csv_filename)

        conn.commit()
        conn.close()

    def create_results_sql_table(self):

        # Make sure user is not overwriting database
        if os.path.exists(self.database_filename):
            raise ExperimentError("Database already exists! Overwriting is not allowed. "
                                  "Delete existing .db file or change database_filename "
                                  "attribute.")

        # Connect to the SQLite database and create database
        # Create a cursor object to execute SQL commands
        # Initialize a table with columns given by column_names
        # Commit changes and close the connection
        conn = sqlite3.connect(self.database_filename)
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

    def create_inputs_realizations_sql_tables(self):

        conn = sqlite3.connect(self.database_filename)
        cursor = conn.cursor()

        for subpop_name in self.inputs_realizations.keys():
            table_name = f'"{subpop_name}_INPUTS"'

            column_names = self.inputs_realizations[subpop_name].keys()

            # Construct the column definitions dynamically
            column_definitions = ", ".join([f'"{col}" FLOAT' for col in column_names])

            # SQL statement to create table
            sql_statement = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                rep INTEGER PRIMARY KEY AUTOINCREMENT,
                {column_definitions}
            )
            """

            cursor.execute(sql_statement)

        conn.commit()
        conn.close()
