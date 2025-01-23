from .utils import np
from .base_components import SubpopModel, MetapopModel
import matplotlib.pyplot as plt
import matplotlib
import functools


def plot_subpop_decorator(plot_func):
    """
    Decorator to handle common subpopulation plotting tasks.
    """

    @functools.wraps(plot_func)
    def wrapper(subpop_model: SubpopModel,
                ax: matplotlib.axes.Axes = None,
                savefig_filename: str = None):
        """
        Args:
            subpop_model (SubpopModel):
                SubpopModel to plot.
            ax (matplotlib.axes.Axes):
                Matplotlib axis to plot on.
            savefig_filename (str):
                Optional filename to save the figure.
        """

        ax_provided = ax

        # If no axis is provided, create own axis
        if ax is None:
            fig, ax = plt.subplots()

        plot_func(subpop_model=subpop_model, ax=ax)

        if savefig_filename:
            plt.savefig(savefig_filename, dpi=1200)

        if ax_provided is None:
            plt.show()

    return wrapper


def plot_metapop_decorator(plot_func):
    """
    Decorator to handle common metapopulation plotting tasks.
    """

    @functools.wraps(plot_func)
    def wrapper(metapop_model: MetapopModel,
                savefig_filename = None):

        num_plots = len(metapop_model.subpop_models)
        num_cols = 2
        num_rows = (num_plots + num_cols - 1) // num_cols

        # Create figure and axes
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
        axes = axes.flatten()

        plot_func(metapop_model=metapop_model, axes=axes)

        # Turn off any unused subplots
        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])  # Remove empty subplot

        # Adjust layout and save/show the figure
        plt.tight_layout()

        if savefig_filename:
            plt.savefig(savefig_filename, dpi=1200)

        plt.show()

    return wrapper


@plot_subpop_decorator
def plot_subpop_epi_metrics(subpop_model: SubpopModel,
                            ax: matplotlib.axes.Axes = None):
    """
    Plots EpiMetric history for a single subpopulation model on the given axis.

    Args:
        subpop_model (SubpopModel):
            Subpopulation model containing compartments.
        ax (matplotlib.axes.Axes):
            Matplotlib axis to plot on.
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


@plot_metapop_decorator
def plot_metapop_epi_metrics(metapop_model: MetapopModel,
                             axes: matplotlib.axes.Axes):
    """
    Plots the EpiMetric data for a metapopulation model.

    Args:
        metapop_model (MetapopModel):
            Metapopulation model containing compartments.
        axes (matplotlib.axes.Axes):
            Matplotlib axes to plot on.
    """

    for ix, (subpop_name, subpop_model) in enumerate(metapop_model.subpop_models.items()):
        plot_subpop_epi_metrics(subpop_model, axes[ix])


@plot_subpop_decorator
def plot_subpop_total_infected_deaths(subpop_model: SubpopModel,
                                      ax: matplotlib.axes.Axes = None):
    """
    Plots data for a single subpopulation model on the given axis.

    Args:
        subpop_model (SubpopModel):
            Subpopulation model containing compartments.
        ax (matplotlib.axes.Axes):
            Matplotlib axis to plot on.
    """

    infected_compartment_names = [name for name in subpop_model.compartments.keys() if
                                  "I" in name or "H" in name]

    infected_compartments_history = [subpop_model.compartments[compartment_name].history_vals_list
                                     for compartment_name in infected_compartment_names]

    total_infected = np.sum(np.asarray(infected_compartments_history), axis=(0, 2, 3))

    ax.plot(total_infected, label="Total infected", alpha=0.6)

    if "D" in subpop_model.compartments.keys():
        deaths = [np.sum(age_risk_group_entry)
                  for age_risk_group_entry
                  in subpop_model.compartments.D.history_vals_list]

        ax.plot(deaths, label="D", alpha=0.6)

    ax.set_title(f"{subpop_model.name}")
    ax.set_xlabel("Days")
    ax.set_ylabel("Number of individuals")
    ax.legend()


@plot_metapop_decorator
def plot_metapop_total_infected_deaths(metapop_model: MetapopModel,
                                       axes: matplotlib.axes.Axes):
    """
    Plots the total infected (IP+IS+IA) and deaths data for a metapopulation model.

    Args:
        metapop_model (MetapopModel):
            Metapopulation model containing compartments.
        axes (matplotlib.axes.Axes):
            Matplotlib axes to plot on.
    """

    # Iterate over subpop models and plot
    for ix, (subpop_name, subpop_model) in enumerate(metapop_model.subpop_models.items()):
        plot_subpop_total_infected_deaths(subpop_model, axes[ix])


@plot_subpop_decorator
def plot_subpop_basic_compartment_history(subpop_model: SubpopModel,
                                          ax: matplotlib.axes.Axes = None):
    """
    Plots data for a single subpopulation model on the given axis.

    Args:
        subpop_model (SubpopModel):
            Subpopulation model containing compartments.
        ax (matplotlib.axes.Axes):
            Matplotlib axis to plot on.
    """

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


@plot_metapop_decorator
def plot_metapop_basic_compartment_history(metapop_model: MetapopModel,
                                           axes: matplotlib.axes.Axes = None):
    """
    Plots the compartment data for a metapopulation model.

    Args:
        metapop_model (MetapopModel):
            Metapopulation model containing compartments.
        axes (matplotlib.axes.Axes):
            Matplotlib axes to plot on.
    """

    # Iterate over subpop models and plot
    for ix, (subpop_name, subpop_model) in enumerate(metapop_model.subpop_models.items()):
        plot_subpop_basic_compartment_history(subpop_model, axes[ix])
