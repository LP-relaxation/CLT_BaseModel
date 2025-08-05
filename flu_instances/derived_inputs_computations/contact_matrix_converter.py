import numpy as np
import pandas as pd

# Texas population counts
##########################
# 0-1, 1-2, ..., 84-85, then 84 is all lumped together (sum for all of those age groups)

tx_pop = pd.read_csv("2023_ASRE_Texas_Population.csv")

tx_pop = tx_pop[tx_pop["County"] == "STATE OF TEXAS"] # State-wide only
tx_pop = tx_pop[["Age", "Total"]] # We only need Age and Total [population] here
tx_pop = tx_pop[tx_pop["Age"] != "All Ages"] # Drop the row with all ages

age_range_85_plus = [f"{age} Years" for age in range(85, 95)] + ["95 Years +"]
rows_85_plus = tx_pop[tx_pop["Age"].isin(age_range_85_plus)]
summed_row = rows_85_plus.select_dtypes(include='number').sum()
summed_row["Age"] = "85 Years"

# Drop all rows in the age_range
tx_pop = tx_pop[~tx_pop["Age"].isin(age_range_85_plus)]

# Append the new summed row
tx_pop = pd.concat([tx_pop, pd.DataFrame([summed_row])], ignore_index=True)

# MOBS Contact Matrix
#####################

original_cm = pd.read_csv("United_States_subnational_Texas_M_overall_contact_matrix_85.csv", header=None)

# Our model's age groups
# - 0-4
# - 5-17
# - 18-49
# - 50-64
# - 65+

age_groups_strs = [str(i) for i in range(85)]

original_cm.columns = age_groups_strs
original_cm.index = age_groups_strs

new_contact_matrix = np.zeros((5, 5))

texas_ix_to_age_groups_map = {0: [str(i) for i in range(5)],
                              1: [str(i) for i in range(5, 17)],
                              2: [str(i) for i in range(18, 49)],
                              3: [str(i) for i in range(50, 64)],
                              4: [str(i) for i in range(65, 85)]}


def get_new_contact_matrix_entry_population_weighted(new_row_ix,
                                                     new_col_ix,
                                                     age_group_populations_map,
                                                     ix_to_age_groups_map):
    '''
    Only rows are weighted!
    This is based on guidance from Remy, Kaiming, Shraddha 04/02/2025
        group CLT meeting.

    :param new_row_ix:
    :param new_col_ix:
    :param age_group_populations_map:
    :param ix_to_age_groups_map:
    :return:
    '''

    original_cm_subset = original_cm.loc[ix_to_age_groups_map[new_row_ix]][ix_to_age_groups_map[new_col_ix]]

    row_age_group_populations = {key: age_group_populations_map[key] for key in ix_to_age_groups_map[new_row_ix]}

    for row in row_age_group_populations.keys():
        original_cm_subset.loc[row] = original_cm_subset.loc[row] * row_age_group_populations[row]

    numerator = original_cm_subset.sum().sum()

    denom = sum([val for val in row_age_group_populations.values()])

    return numerator / denom


texas_age_group_populations = {}
texas_age_group_populations[str(0)] = float(tx_pop[tx_pop["Age"] == "< 1 Year"]["Total"])

for i in range(1, 85):
    texas_age_group_populations[str(i)] = float(tx_pop[tx_pop["Age"] == str(i) + " Years"]["Total"])


for row in range(5):
    for col in range(5):
        new_contact_matrix[row, col] = get_new_contact_matrix_entry_population_weighted(row,
                                                                                        col,
                                                                                        texas_age_group_populations,
                                                                                        texas_ix_to_age_groups_map)

breakpoint()
