import matplotlib.pyplot as plt

def create_basic_compartment_history_plot(sim_model):
    plt.clf()
    plt.figure(1)
    for name, compartment in sim_model.name_to_epi_compartment_dict.items():
        plt.plot(compartment.history_vals_list, label=name, alpha=0.6)
    plt.legend()
    plt.xlabel("Days")
    plt.ylabel("Number of individuals")
    plt.show()