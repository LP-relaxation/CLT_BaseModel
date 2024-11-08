import numpy as np
import matplotlib.pyplot as plt


def create_basic_compartment_history_plot(sim_model):
    plt.clf()
    plt.figure(1)
    for compartment in sim_model.compartments:
        history_vals_list = [np.sum(age_risk_group_entry) for age_risk_group_entry in compartment.history_vals_list]
        plt.plot(history_vals_list, label=compartment.name, alpha=0.6)
    plt.legend()
    plt.xlabel("Days")
    plt.ylabel("Number of individuals")
    plt.show()