from .utils import np
from .base_components import SubpopModel, MetapopModel
import matplotlib.pyplot as plt
import matplotlib


def plot_subpop_epi_metrics(subpop_model: SubpopModel,
                            ax: matplotlib.axes.Axes = None,
                            savefig_filename: str = None):
    """
    Plots EpiMetric history for a single subpopulation model on the given axis.

    Args:
        subpop_model (SubpopModel):
            Subpopulation model containing compartments.
        ax (matplotlib.axes.Axes):
            Matplotlib axis to plot on.
        savefig_filename (str):
            Optional filename to save the figure.
    """

    for name, epi_metric in subpop_model.epi_metrics.items():
        # Compute summed history values for each age-risk group
        history_vals_list = [np.average(age_risk_group_entry) for
                             age_risk_group_entry in epi_metric.history_vals_list]

        # Plot data with a label
        ax.plot(history_vals_list, label=name, alpha=0.6)

    # Set axis title and labels
    ax.set_title(f"{subpop_model.name}")
    ax.set_xlabel("Days")
    ax.set_ylabel("Epi Metric Value")
    ax.legend()

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=1200)
        plt.show()


def plot_metapop_epi_metrics(metapop_model: MetapopModel,
                             savefig_filename=None):
    """
    Plots the EpiMetric data for a metapopulation model.

    Args:
        metapop_model (MetapopModel):
            Metapopulation model containing compartments.
        savefig_filename: Optional filename to save the figure.
    """

    num_plots = len(metapop_model.subpop_models)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Create figure and axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    axes = axes.flatten()

    # Iterate over subpop models and plot
    for ix, (subpop_name, subpop_model) in enumerate(metapop_model.subpop_models.items()):
        plot_subpop_epi_metrics(subpop_model, axes[ix])

    # Turn off any unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])  # Remove empty subplot

    # Adjust layout and save/show the figure
    plt.tight_layout()

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=1200)

    plt.show()


def plot_subpop_total_infected_deaths(subpop_model: SubpopModel,
                                      ax: matplotlib.axes.Axes = None,
                                      savefig_filename: str = None):
    """
    Plots data for a single subpopulation model on the given axis.

    Args:
        subpop_model (SubpopModel):
            Subpopulation model containing compartments.
        ax (matplotlib.axes.Axes):
            Matplotlib axis to plot on.
        savefig_filename (str):
            Optional filename to save the figure.
    """

    if not ax:
        fig, axes = plt.subplots()

    infected_compartments_history = [subpop_model.compartments[compartment_name].history_vals_list
                                     for compartment_name in ["IP", "IA", "IS", "H"]]

    total_infected = np.sum(np.asarray(infected_compartments_history), axis=(0, 2, 3))

    deaths = [np.sum(age_risk_group_entry)
              for age_risk_group_entry
              in subpop_model.compartments.D.history_vals_list]

    # Plot data with a label
    ax.plot(total_infected, label="IA + IA + IS + H", alpha=0.6)
    ax.plot(deaths, label="D", alpha=0.6)

    # Set axis title and labels
    ax.set_title(f"{subpop_model.name}")
    ax.set_xlabel("Days")
    ax.set_ylabel("Number of individuals")
    ax.legend()

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=1200)
        plt.show()


def plot_metapop_total_infected_deaths(metapop_model: MetapopModel,
                                       savefig_filename=None):
    """
    Plots the total infected (IP+IS+IA) and deaths data for a metapopulation model.

    Args:
        metapop_model (MetapopModel):
            Metapopulation model containing compartments.
        savefig_filename: Optional filename to save the figure.
    """

    num_plots = len(metapop_model.subpop_models)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Create figure and axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    axes = axes.flatten()

    # Iterate over subpop models and plot
    for ix, (subpop_name, subpop_model) in enumerate(metapop_model.subpop_models.items()):
        plot_subpop_total_infected_deaths(subpop_model, axes[ix])

    # Turn off any unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])  # Remove empty subplot

    # Adjust layout and save/show the figure
    plt.tight_layout()

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=1200)

    plt.show()


def plot_subpop_basic_compartment_history(subpop_model: SubpopModel,
                                          ax: matplotlib.axes.Axes = None,
                                          savefig_filename: str = None):
    """
    Plots data for a single subpopulation model on the given axis.

    Args:
        subpop_model (SubpopModel):
            Subpopulation model containing compartments.
        ax (matplotlib.axes.Axes):
            Matplotlib axis to plot on.
        savefig_filename (str):
            Optional filename to save the figure.
    """

    if not ax:
        fig, axes = plt.subplots()

    for name, compartment in subpop_model.compartments.items():
        # Compute summed history values for each age-risk group
        history_vals_list = [np.sum(age_risk_group_entry) for age_risk_group_entry in compartment.history_vals_list]

        # Plot data with a label
        ax.plot(history_vals_list, label=name, alpha=0.6)

    # Set axis title and labels
    ax.set_title(f"{subpop_model.name}")
    ax.set_xlabel("Days")
    ax.set_ylabel("Number of individuals")
    ax.legend()

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=1200)
        plt.show()


def plot_metapop_basic_compartment_history(metapop_model: MetapopModel,
                                           savefig_filename=None):
    """
    Plots the compartment data for a metapopulation model.

    Args:
        metapop_model (MetapopModel):
            Metapopulation model containing compartments.
        savefig_filename: Optional filename to save the figure.
    """

    num_plots = len(metapop_model.subpop_models)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Create figure and axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    axes = axes.flatten()

    # Iterate over subpop models and plot
    for ix, (subpop_name, subpop_model) in enumerate(metapop_model.subpop_models.items()):
        plot_subpop_basic_compartment_history(subpop_model, axes[ix])

    # Turn off any unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])  # Remove empty subplot

    # Adjust layout and save/show the figure
    plt.tight_layout()

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=1200)

    plt.show()
