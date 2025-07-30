from .utils import np, sc, Optional, List, sqlite3, functools, os, pd, fields
from .base_components import SubpopModel, MetapopModel

class ExperimentError(Exception):
    """Custom exceptions for experiment errors."""
    pass


def check_is_subset_list(listA: list,
                         listB: list) -> bool:
    """
    Params:
        listA (list):
            list-like of elements to check if subset of listB.
        listB (list):
            list-like of elements.

    Returns:
        True if listA is a subset of listB, and False otherwise.
    """

    return all(item in listB for item in listA)


def format_current_val_for_sql(subpop_model: SubpopModel,
                               state_var_name: str,
                               rep: int) -> list:
    """
    Processes current_val of given subpop_model's `StateVariable`
    specified by `state_var_name`. Current_val is an A x R
    numpy array (for age-risk) -- this function "unpacks" it into an
    (A x R, 1) numpy array (a column vector). Converts metadata
    (subpop_name, state_var_name, `rep`, and current_simulation_day)
    into list of A x R rows, where each row has 7 elements, for
    consistent row formatting for batch SQL insertion.

    Params:
        subpop_model (SubpopModel):
            SubpopModel to record.
        state_var_name (str):
            StateVariable name to record.
        rep (int):
            replication counter to record.

    Returns:
        data (list):
            list of A x R rows, where each row is a list of 7 elements
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
    Returns a pandas DataFrame containing data from specified SQL table,
    retrieved using the provided database connection. Reads in SQL rows
    in batches of size `chunk_size` to avoid memory issues for very large
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

    Returns:
        DataFrame containing data from specified SQL table,
        or empty DataFrame if table does not exist.
    """

    chunks = []

    try:
        for chunk in pd.read_sql_query(sql_query,
                                       conn,
                                       chunksize=chunk_size,
                                       params=sql_query_params):
            chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)

    # Handle exception gracefully -- print a warning and
    #   return an empty DataFrame if table given by sql_query
    #   does not exist
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            print(f"Warning: table does not exist for query: {sql_query}. "
                  f"Returning empty DataFrame.")
            df = pd.DataFrame()

    return df


class Experiment:
    """
    Class to manage running multiple simulation replications
    on a `SubpopModel` or `MetapopModel` instance and query its results.

    Also allows running a batch of simulation replications on a
    deterministic sequence of values for a given input
    (for example, to see how output changes as a function of
    a given input).

    Also handles random sampling of inputs from a uniform
    distribution.

    NOTE:
        If an input is an A x R array (for age-risk),
        the current functionality does not support sampling individual
        age-risk elements separately. Instead, a single scalar value
        is sampled at a time for the entire input.
        See `self.sample_random_inputs` method for more details.

    Params:
        experiment_subpop_models (tuple):
            tuple of `SubpopModel` instances associated with the `Experiment`.
            If the `Experiment` is for a `MetapopModel`, then this tuple
            contains all the associated `SubpopModel` instances
            that comprise that `MetapopModel.` If the `Experiment` is for
            a `SubpopModel` only, then this tuple contains only that
            particular `SubpopModel`.
        inputs_realizations (dict):
            dictionary of dictionaries that stores user-specified deterministic
            sequences for inputs or realizations of random input sampling --
            keys are `SubpopModel` names, values are dictionaries.
            These second-layer dictionaries' keys are strings corresponding
            to input names (they must match names in each `SubpopModel`'s
            `SubpopParams` to be valid) and values are list-like, where the ith
            element corresponds to the ith random sample for that input.
        results_df (pd.DataFrame):
            DataFrame holding simulation results from each
            `simulation` replication
        has_been_run (bool):
            indicates if `self.run_static_inputs`, `self.run_random_inputs`,
            or `self.run_sequences_of_inputs` has been executed.

    See `__init__` docstring for other attributes.
    """

    def __init__(self,
                 model: SubpopModel | MetapopModel,
                 state_variables_to_record: list,
                 database_filename: str):

        """
        Params:
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

        self.model = model
        self.state_variables_to_record = state_variables_to_record
        self.database_filename = database_filename

        self.has_been_run = False

        # Create experiment_subpop_models tuple
        # If model is MetapopModel instance, then this tuple is a list
        #   of all associated SubpopModel instances
        # If model is a SubpopModel instance, then this tuple
        #   only contains that SubpopModel.
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

        for subpop_model in self.experiment_subpop_models:
            self.inputs_realizations[subpop_model.name] = {}

            # Make sure the state variables to record are valid -- the names
            #   of the state variables to record must match actual state variables
            #   on each SubpopModel
            if not check_is_subset_list(state_variables_to_record,
                                        subpop_model.all_state_variables.keys()):
                raise ExperimentError(
                    f"\"state_variables_to_record\" list is not a subset "
                    "of the state variables on SubpopModel \"{subpop_name}\" -- "
                    "modify \"state_variables_to_record\" and re-initialize experiment.")

    def run_static_inputs(self,
                          num_reps: int,
                          simulation_end_day: int,
                          days_between_save_history: int = 1,
                          results_filename: str = None):
        """
        Runs the associated `SubpopModel` or `MetapopModel` for a
        given number of independent replications until `simulation_end_day`.
        Parameter values and initial values are the same across
        simulation replications. User can specify how often to save the
        history and a CSV file in which to store this history.

        Params:
            num_reps (positive int):
                number of independent simulation replications
                to run in an experiment.
            simulation_end_day (positive int):
                stop simulation at simulation_end_day (i.e. exclusive,
                simulate up to but not including simulation_end_day).
            days_between_save_history (positive int):
                indicates how often to save simulation results.
            results_filename (str):
                if specified, must be valid filename with suffix ".csv" --
                experiment results are saved to this CSV file.
        """

        if self.has_been_run:
            raise ExperimentError("Experiment has already been run. "
                                  "Create a new Experiment instance to simulate "
                                  "more replications.")

        else:
            self.has_been_run = True

            self.create_results_sql_table()

            self.simulate_reps_and_save_results(reps=num_reps,
                                                end_day=simulation_end_day,
                                                days_per_save=days_between_save_history,
                                                inputs_are_static=True,
                                                filename=results_filename)

    def run_random_inputs(self,
                          num_reps: int,
                          simulation_end_day: int,
                          random_inputs_RNG: np.random.Generator,
                          random_inputs_spec: dict,
                          days_between_save_history: int = 1,
                          results_filename: str = None,
                          inputs_filename_suffix: str = None):
        """
        Runs the associated `SubpopModel` or `MetapopModel` for a
        given number of independent replications until `simulation_end_day`,
        where certain parameter values or initial values are
        independently randomly sampled for each replication.
        Random sampling details (which inputs to sample and what
        Uniform distribution lower and upper bounds to use for each input)
        are specified in `random_inputs_spec`, and random sampling of inputs
        uses `random_inputs_RNG`. User can specify how often to save the
        history and a CSV file in which to store this history.
        User can also specify a filename suffix to name CSV files
        in which to store realizations of inputs.

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
                names in each `SubpopModel`'s `SubpopParams` or associated
                `EpiMetric` or `Compartment` instances to be valid)
                and values are 2-tuples of nonnegative floats corresponding
                to lower and upper bounds for sampling that input from a
                uniform distribution.
            days_between_save_history (positive int):
                indicates how often to save simulation results.
            results_filename (str):
                if specified, must be valid filename with suffix ".csv" --
                experiment results are saved to this CSV file
            inputs_filename_suffix (str):
                if specified, must be a valid filename with suffix ".csv" --
                one inputs CSV is generated for each `SubpopModel` with the
                filename "{name}_{inputs_filename}" where name is the name
                of the `SubpopModel` -- saves the values of each parameter that
                varies between replications
        """

        if self.has_been_run:
            raise ExperimentError("Experiment has already been run. "
                                  "Create a new Experiment instance to simulate "
                                  "more replications.")

        else:
            self.has_been_run = True

            self.sample_random_inputs(num_reps,
                                      random_inputs_RNG,
                                      random_inputs_spec)

            self.create_results_sql_table()
            self.create_inputs_realizations_sql_tables()

            self.simulate_reps_and_save_results(reps=num_reps,
                                                end_day=simulation_end_day,
                                                days_per_save=days_between_save_history,
                                                inputs_are_static=False,
                                                filename=results_filename)

            if inputs_filename_suffix:
                self.write_inputs_csvs(inputs_filename_suffix)

    def run_sequences_of_inputs(self,
                                num_reps: int,
                                simulation_end_day: int,
                                sequences_of_inputs: dict,
                                days_between_save_history: int = 1,
                                results_filename: str = None,
                                inputs_filename_suffix: str = None):
        """
        Runs the associated `SubpopModel` or `MetapopModel` for a
        given number of independent replications until `simulation_end_day`,
        where certain parameter values or initial values deterministically
        change between replications, according to `sequences_of_inputs`.
        User can specify how often to save the history and a CSV file
        in which to store this history. User can also specify a filename
        suffix to name CSV files in which to store sequences of inputs.

        Params:
            num_reps (positive int):
                number of independent simulation replications
                to run in an experiment.
            simulation_end_day (positive int):
                stop simulation at `simulation_end_day` (i.e. exclusive,
                simulate up to but not including `simulation_end_day`).
            sequences_of_inputs (dict):
                dictionary of dictionaries that stores user-specified deterministic
                sequences for inputs -- must follow specific structure.
                Keys are `SubpopModel` names, values are dictionaries.
                These second-layer dictionaries' keys are strings corresponding
                to input names (they must match names in each `SubpopModel`'s
                `SubpopParams` to be valid) and values are list-like, where the ith
                element corresponds to the ith random sample for that input.
            days_between_save_history (positive int):
                indicates how often to save simulation results.
            results_filename (str):
                if specified, must be valid filename with suffix ".csv" --
                experiment results are saved to this CSV file
            inputs_filename_suffix (str):
                if specified, must be a valid filename with suffix ".csv" --
                one inputs CSV is generated for each `SubpopModel` with the
                filename "{name}_{inputs_filename}" where name is the name
                of the `SubpopModel` -- saves the values of each parameter that
                varies between replications
        """

        if self.has_been_run:
            raise ExperimentError("Experiment has already been run. "
                                  "Create a new Experiment instance to simulate "
                                  "more replications.")

        else:
            self.has_been_run = True

            self.inputs_realizations = sequences_of_inputs

            self.create_results_sql_table()
            self.create_inputs_realizations_sql_tables()

            self.simulate_reps_and_save_results(reps=num_reps,
                                                end_day=simulation_end_day,
                                                days_per_save=days_between_save_history,
                                                inputs_are_static=False,
                                                filename=results_filename)

            if inputs_filename_suffix:
                self.write_inputs_csvs(inputs_filename_suffix)

    def sample_random_inputs(self,
                             total_reps: int,
                             RNG: np.random.Generator,
                             spec: dict):
        """
        Randomly and independently samples inputs specified by keys of
        `spec` according to uniform distribution with lower
        and upper bounds specified by values of `spec`.
        Stores random realizations in `self.inputs_realizations` attribute.
        Uses `RNG` to sample inputs.

        Params:
            total_reps (positive int):
                number of independent simulation replications
                to run in an experiment -- corresponds to number of
                Uniform random variables to draw for each
                state variable.
            RNG (np.random.Generator):
                random number generator used to sample random
                inputs -- for reproducibility, it is recommended to
                use a distinct RNG for sampling random inputs different
                from the RNG for simulating the experiment/model.
            spec (dict):
                random inputs' specification -- stores details
                for random input sampling -- keys are strings
                corresponding to input names (they must match
                names in each `SubpopModel`'s `SubpopParams` or associated
                `EpiMetric` or `Compartment` instances to be valid)
                and values are 2-tuples of nonnegative floats corresponding
                to lower and upper bounds for sampling that input from a
                uniform distribution.

        NOTE:
            If an input is an A x R array (for age-risk),
            the current functionality does not support sampling individual
            age-risk elements separately. Instead, a single scalar value
            is sampled at a time for the entire input. Consequently,
            if an A x R input is chosen to be randomly sampled,
            all its elements will have the same sampled value.

            If a user wants to sample some age-risk elements separately,
            they should create new inputs for these elements. "Inputs"
            refers to both parameters (in `SubpopParams`) and initial values
            of `Compartment` and `EpiMetric` instances. For example, if the model
            has a parameter `H_to_R_rate` that is 2x1 (2 age groups, 1 risk group)
            and the user wants to sample each element separately, they should create
            two parameters: `H_to_R_rate_age_group_1` and `H_to_R_rate_age_group_2.`
            These should be added to the relevant `SubpopParams` instance and
            input dictionary/file used to create the `SubpopParams` instance.
            The user can then specify both parameters to be randomly sampled
            and specify the lower and upper bounds accordingly.

            TODO: allow sampling individual age-risk elements separately
            without creating new parameters for each element.

            (Developer note: the difficulty is not with randomly sampling
            arrays, but rather storing arrays in SQL -- SQL tables only
            support atomic values.)
        """

        inputs_realizations = self.inputs_realizations

        for subpop_model in self.experiment_subpop_models:

            subpop_name = subpop_model.name

            compartments_names_list = list(subpop_model.compartments.keys())
            epi_metrics_names_list = list(subpop_model.epi_metrics.keys())
            params_names_list = [field.name for field in fields(subpop_model.params)]

            if not check_is_subset_list(spec[subpop_name],
                                        compartments_names_list + epi_metrics_names_list + params_names_list):
                raise (f"\"random_inputs_spec[\"{subpop_name}\"]\" keys are not a subset "
                       "of the state variables and parameters on SubpopModel \"{subpop_name}\" -- "
                       "modify \"random_inputs_spec\" and re-initialize experiment.")

            for state_var_name, bounds in spec[subpop_name].items():
                lower_bd = bounds[0]
                upper_bd = bounds[1]
                inputs_realizations[subpop_name][state_var_name] = \
                    RNG.uniform(low=lower_bd,
                                high=upper_bd,
                                size=total_reps)

    def get_state_var_df(self,
                         state_var_name: str,
                         subpop_name: str = None,
                         age_group: int = None,
                         risk_group: int = None,
                         results_filename: str = None) -> pd.DataFrame:
        """
        Get pandas DataFrame of recorded values of `StateVariable` given by
        `state_var_name`, in the `SubpopModel` given by `subpop_name`,
        for the age-risk group given by `age_group` and `risk_group`.
        If `subpop_name` is not specified, then values are summed across all
        associated subpopulations. Similarly, if `age_group` (or `risk_group`)
        is not specified, then values are summed across all age groups
        (or risk groups).

        Args:
            state_var_name (str):
                Name of the `StateVariable` to retrieve.
            subpop_name (Optional[str]):
                The name of the `SubpopModel` for filtering. If None, values are
                summed across all `SubpopModel` instances.
            age_group (Optional[int]):
                The age group to select. If None, values are summed across
                all age groups.
            risk_group (Optional[int]):
                The risk group to select. If None, values are summed across
                all risk groups.
            results_filename (Optional[str]):
                If provided, saves the resulting DataFrame as a CSV.

        Returns:
            A pandas DataFrame where rows represent the replication and columns indicate the
            simulation day (timepoint) of recording. DataFrame values are the `StateVariable`'s
            current_val or the sum of the `StateVariable`'s current_val across subpopulations,
            age groups, or risk groups (the combination of what is summed over is
            specified by the user -- details are in the part of this docstring describing
            this function's parameters).
        """

        if state_var_name not in self.state_variables_to_record:
            raise ExperimentError("\"state_var_name\" is not in \"self.state_variables_to_record\" --"
                                  "function call is invalid.")

        conn = sqlite3.connect(self.database_filename)

        # Query all results table entries where state_var_name matches
        # This will return results across all subpopulations, age groups,
        #   and risk groups
        df = get_sql_table_as_df(conn,
                                 "SELECT * FROM results WHERE state_var_name = ?",
                                 chunk_size=int(1e4),
                                 sql_query_params=(state_var_name,))

        conn.close()

        # Define filter conditions
        filters = {
            "subpop_name": subpop_name,
            "age_group": age_group,
            "risk_group": risk_group
        }

        # Filter DataFrame based on user-specified conditions
        #   (for example, if user specifies subpop_name, return subset of
        #   DataFrame where subpop_name matches)
        conditions = [(df[col] == value) for col, value in filters.items() if value is not None]
        df_filtered = df if not conditions else df[np.logical_and.reduce(conditions)]

        # Group DataFrame based on unique combinations of "rep" and "timepoint" columns
        # Then sum (numeric values only), return the "value" column, and reset the index
        #   so that "rep" and "timepoint" become regular columns and are not the index
        df_aggregated = \
            df_filtered.groupby(["rep",
                                 "timepoint"]).sum(numeric_only=True)["value"].reset_index()

        # breakpoint()

        # Use pivot() function to reshape the DataFrame for its final form
        # The "timepoint" values are spread across new columns
        #   (creating a column for each unique timepoint).
        # The "value" column populates the corresponding cells.
        df_final = df_aggregated.pivot(index="rep",
                                       columns="timepoint",
                                       values="value")

        if results_filename:
            df_final.to_csv(results_filename)

        return df_final

    def log_current_vals_to_sql(self,
                                rep_counter: int,
                                experiment_cursor: sqlite3.Cursor) -> None:
        """
        For each subpopulation and state variable to record
        associated with this `Experiment`, save current values to
        "results" table in SQL database specified by `experiment_cursor`.

        Params:
            rep_counter (int):
                Current replication ID.
            experiment_cursor (sqlite3.Cursor):
                Cursor object connected to the database
                where results should be inserted.
        """

        for subpop_model in self.experiment_subpop_models:
            for state_var_name in self.state_variables_to_record:
                data = format_current_val_for_sql(subpop_model,
                                                  state_var_name,
                                                  rep_counter)
                experiment_cursor.executemany(
                    "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?)", data)

    def log_inputs_to_sql(self,
                          experiment_cursor: sqlite3.Cursor):
        """
        For each subpopulation, add a new table to SQL
        database specified by `experiment_cursor`. Each table
        contains information on inputs that vary across
        replications (either due to random sampling or
        user-specified deterministic sequence). Each table
        contains inputs information from `Experiment` attribute
        `self.inputs_realizations` for a given subpopulation.

        Params:
            experiment_cursor (sqlite3.Cursor):
                Cursor object connected to the database
                where results should be inserted.
        """

        for subpop_model in self.experiment_subpop_models:
            table_name = f'"{subpop_model.name}_inputs"'

            # Get the column names (dynamically, based on table)
            experiment_cursor.execute(f"PRAGMA table_info({table_name})")

            # Extract column names from the table info
            # But exclude the column name "rep"
            columns_info = experiment_cursor.fetchall()
            column_names = [col[1] for col in columns_info if col[1] != "rep"]

            # Create a placeholder string for the dynamic query
            placeholders = ", ".join(["?" for _ in column_names])  # Number of placeholders matches number of columns

            # Create the dynamic INSERT statement
            sql_statement = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({placeholders})"

            # Create list of lists -- each nested list contains a sequence of values
            #   for that particular input
            subpop_inputs_realizations = self.inputs_realizations[subpop_model.name]
            inputs_vals_over_reps_list = \
                [np.array(subpop_inputs_realizations[input_name]).reshape(-1,1) for input_name in column_names]
            inputs_vals_over_reps_list = np.hstack(inputs_vals_over_reps_list)

            experiment_cursor.executemany(sql_statement, inputs_vals_over_reps_list)

    def apply_inputs_to_model(self,
                              rep_counter: int):
        """
        Changes inputs (parameters or initial values) for a given
        replication according to `self.inputs_realizations` attribute.

        Specifically, for each subpopulation, this function retrieves
        the corresponding values from `self.inputs_realizations` and applies them to either:
        (a) the subpopulation's `StateVariable`, if the input corresponds to a state variable,
        or (b) the subpopulation's `params` attribute, if the input corresponds to a model
        parameter. If the parameter or state variable is multidimensional
        (e.g., varies across age or risk groups), it is assigned a numpy array of
        the appropriate shape with the replicated input value.

        Params:
            rep_counter (int):
                Replication ID, used to retrieve the correct
                realizations of inputs for the current run.
        """

        for subpop_model in self.experiment_subpop_models:

            params = subpop_model.params

            for input_name, input_val in self.inputs_realizations[subpop_model.name].items():

                dimensions = (params.num_age_groups, params.num_risk_groups)

                if input_name in subpop_model.all_state_variables.keys():
                    subpop_model.all_state_variables[input_name].current_val = np.full(dimensions,
                                                                                       input_val[rep_counter])
                else:
                    if np.isscalar(getattr(subpop_model.params, input_name)):
                        setattr(subpop_model.params, input_name, input_val[rep_counter])
                    else:
                        setattr(subpop_model.params, input_name, np.full(dimensions, input_val[rep_counter]))

    def simulate_reps_and_save_results(self,
                                       reps: int,
                                       end_day: int,
                                       days_per_save: int,
                                       inputs_are_static: bool,
                                       filename: str = None):
        """
        Helper function that executes main loop over
        replications in `Experiment` and saves results.

        Params:
            reps (int):
                number of independent simulation replications
                to run in an experiment.
            end_day (int):
                stop simulation at end_day (i.e. exclusive,
                simulate up to but not including end_day).
            days_per_save (int):
                indicates how often to save simulation results.
            inputs_are_static (bool):
                indicates if inputs are same across replications.
            filename (str):
                if specified, must be valid filename with suffix ".csv" --
                experiment results are saved to this CSV file.
        """

        # Override each subpop config's save_daily_history attribute --
        #   set it to False -- because we will manually save history
        #   to results database according to user-defined
        #   days_between_save_history for all subpops
        for subpop_model in self.experiment_subpop_models:
            subpop_model.config.save_daily_history = False

        model = self.model

        # Connect to SQL database
        conn = sqlite3.connect(self.database_filename)
        cursor = conn.cursor()

        # Loop through replications
        for rep in range(reps):

            # Reset model and clear its history
            model.reset_simulation()

            # Apply new values of inputs, if some
            #   inputs change between replications
            if not inputs_are_static:
                self.apply_inputs_to_model(rep)
                self.log_inputs_to_sql(cursor)

            # Simulate model and save results every `days_per_save` days
            while model.current_simulation_day < end_day:
                model.simulate_until_day(min(model.current_simulation_day + days_per_save,
                                             end_day))

                self.log_current_vals_to_sql(rep, cursor)

        self.results_df = get_sql_table_as_df(conn, "SELECT * FROM results", chunk_size=int(1e4))

        if filename:
            self.results_df.to_csv(filename)

        # Commit changes to database and close
        conn.commit()
        conn.close()

    def create_results_sql_table(self):
        """
        Create SQL database and save to `self.database_filename`.
        Create table named `results` with columns `subpop_name`,
        `state_var_name`, `age_group`, `risk_group`, `rep`, `timepoint`,
        and `value` to store results from each replication of experiment.
        """

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
        """
        Create tables in SQL database given by `self.database_filename`
        to store realizations of inputs that change across
        replications. There is one table per associated subpopulation.
        """

        conn = sqlite3.connect(self.database_filename)
        cursor = conn.cursor()

        for subpop_name in self.inputs_realizations.keys():
            table_name = f'"{subpop_name}_inputs"'

            column_names = self.inputs_realizations[subpop_name].keys()

            # Construct the column definitions dynamically
            column_definitions = ", ".join([f'"{col}" FLOAT' for col in column_names])

            # SQL statement to create table
            # Each table has a column called "rep" which is the primary
            #   key and automatically increments every time a row is added
            sql_statement = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                rep INTEGER PRIMARY KEY AUTOINCREMENT,
                {column_definitions}
            )
            """

            cursor.execute(sql_statement)

        conn.commit()
        conn.close()

    def write_inputs_csvs(self,
                          suffix: str):
        """
        For each subpopulation, writes a CSV (with a filename
        based on provided suffix) containing values of inputs
        across replications.

        Params:
            suffix (str):
                Common suffix used to generate CSV filenames.
        """

        for subpop_model in self.experiment_subpop_models:
            conn = sqlite3.connect(self.database_filename)

            table_name = f"{subpop_model.name}_inputs"

            subpop_inputs_df = get_sql_table_as_df(conn, f"SELECT * FROM {table_name}", chunk_size=int(1e4))

            subpop_inputs_df.to_csv(f"{subpop_model.name}_{suffix}", index=False)
