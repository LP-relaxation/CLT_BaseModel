import matplotlib.pyplot as plt

def create_basic_compartment_history_plot(sim_model):
    plt.figure(1)
    for name, compartment in sim_model.name_to_compartment_dict.items():
        plt.plot(compartment.history_vals_list, label=name)
    plt.legend()
    plt.xlabel("Days")
    plt.xlim([0, 400])
    plt.ylabel("Number of individuals")
    plt.show()