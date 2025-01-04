from .utils import np
import matplotlib.pyplot as plt


def create_basic_compartment_history_plot(sim_model,
                                          savefig_filename=None):
    plt.clf()
    plt.figure(1)
    for name, compartment in sim_model.compartments.items():
        history_vals_list = [np.sum(age_risk_group_entry) for age_risk_group_entry in compartment.history_vals_list]
        plt.plot(history_vals_list, label=name, alpha=0.6)
    plt.legend()
    plt.xlabel("Days")
    plt.ylabel("Number of individuals")

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=1200)

    plt.show()