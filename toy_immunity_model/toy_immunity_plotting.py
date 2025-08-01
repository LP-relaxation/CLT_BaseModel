import matplotlib.pyplot as plt
import numpy as np

import toy_immunity_components as imm


def make_graph_set(model: imm.ToyImmunitySubpopModel):

    plt.clf()
    plt.figure(figsize=(8, 12))

    plt.subplot(4, 1, 1)
    plt.plot(np.sum(np.asarray(model.compartments.S.history_vals_list), axis=(1,2)), label="S")
    plt.plot(np.sum(np.asarray(model.compartments.I.history_vals_list), axis=(1,2)), label="I")
    plt.plot(np.sum(np.asarray(model.compartments.H.history_vals_list), axis=(1,2)), label="H")
    plt.plot(np.sum(np.asarray(model.compartments.R.history_vals_list), axis=(1,2)), label="R")
    plt.title("Simulated compartment populations")
    plt.xlabel("Day")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.subplot(4, 1, 2)
    plt.plot(np.sum(np.asarray(model.epi_metrics.M.history_vals_list), axis=(1,2)), label="M")
    plt.plot(np.sum(np.asarray(model.epi_metrics.Mv.history_vals_list), axis=(1,2)), label="Mv")
    plt.title("Simulated immunity")
    plt.xlabel("Day")

    plt.subplot(4, 1, 3)
    plt.plot(np.sum(np.asarray(model.transition_variables.S_to_I.history_vals_list), axis=(1,2)), label="S to I")
    plt.plot(np.sum(np.asarray(model.transition_variables.I_to_H.history_vals_list), axis=(1,2)), label="I to H")
    plt.title("Simulated incidence and hospital admits")
    plt.xlabel("Day")

    plt.subplot(4, 1, 4)
    plt.plot(np.sum(np.asarray(model.transition_variables.R_to_S.history_vals_list), axis=(1,2)))
    plt.title("Simulated R to S")
    plt.xlabel("Day")

    plt.tight_layout()

    plt.show()


def plot_comparison(ax, model1_data, model2_data, labels, title, xlabel="Day"):
    # Helper function for `make_comparison_graph_set`

    lines = []
    for label in labels:
        m1_vals = np.sum(np.asarray(model1_data[label].history_vals_list), axis=(1,2))
        line, = ax.plot(m1_vals, label=label)
        lines.append((label, line))

    for label, line in lines:
        m2_vals = np.sum(np.asarray(model2_data[label].history_vals_list), axis=(1,2))
        ax.plot(m2_vals, marker="o", markevery=100, markersize=2, linestyle=":", label=f"{label}, model 2",
                color=line.get_color(), alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend(loc='upper left')


def make_comparison_graph_set(model1: imm.ToyImmunitySubpopModel,
                              model2: imm.ToyImmunitySubpopModel):
    # Function to graph two models to compare outputs (to check identifiability)

    plt.figure(figsize=(8, 12))

    # Subplot 1: Compartments
    ax1 = plt.subplot(4, 1, 1)
    plot_comparison(
        ax1,
        model1.compartments,
        model2.compartments,
        labels=["S", "I", "H", "R"],
        title="Simulated compartment populations"
    )

    # Subplot 2: Immunity metrics
    ax2 = plt.subplot(4, 1, 2)
    plot_comparison(
        ax2,
        model1.epi_metrics,
        model2.epi_metrics,
        labels=["M", "Mv"],
        title="Simulated immunity"
    )

    # Subplot 3: Incidence and hospital admits
    ax3 = plt.subplot(4, 1, 3)
    plot_comparison(
        ax3,
        model1.transition_variables,
        model2.transition_variables,
        labels=["S_to_I", "I_to_H"],
        title="Simulated incidence and hospital admits"
    )

    # Subplot 4: R to S
    ax4 = plt.subplot(4, 1, 4)
    plot_comparison(
        ax4,
        model1.transition_variables,
        model2.transition_variables,
        labels=["R_to_S"],
        title="Simulated R to S"
    )

    plt.tight_layout()
    plt.show()


def changing_param_val_graph(model, param_type, param_name, param_vals_list, end_day=240):

    plt.clf()

    markers_list = ['o', 's', '^', 'v', 'D', 'x', '+', '*']
    blue_colors_list = ['#F0F8FF', '#CCCCFF', '#B0E0E6', '#87CEFA', '#87CEEB', '#ADD8E6', '#89CFF0', '#AFEEEE']
    red_colors_list = ['#FFE4E1', '#FFF0F5', '#F08080', '#FA8072', '#FFA07A', '#CD5C5C', '#DC143C', '#B22222']
    green_colors_list = [
        '#E0F2E9',  # pastel mint
        '#D0F0C0',  # tea green
        '#98FB98',  # pale green
        '#90EE90',  # light green
        '#66CDAA',  # medium aquamarine
        '#3CB371',  # medium sea green
        '#2E8B57',  # sea green
        '#006400',  # dark green
    ]
    alpha_list = np.linspace(1.0, 0.4, num=len(param_vals_list))

    # ---- Run all simulations once and store outputs ----
    sim_results = []

    for i, param_val in enumerate(param_vals_list):
        model.reset_simulation()

        if param_type == "param":
            setattr(model.params, param_name, param_val)
        elif param_type == "init_cond":
            try:
                setattr(getattr(model.epi_metrics, param_name), "init_val", param_val)
            except AttributeError:
                setattr(getattr(model.compartments, param_name), "init_val", param_val)

        model.simulate_until_day(end_day)

        sim_results.append({
            "S": np.copy(np.sum(np.asarray(model.compartments.S.history_vals_list), axis=(1,2))),
            "I": np.copy(np.sum(np.asarray(model.compartments.I.history_vals_list), axis=(1,2))),
            "H": np.copy(np.sum(np.asarray(model.compartments.H.history_vals_list), axis=(1,2))),
            "R": np.copy(np.sum(np.asarray(model.compartments.R.history_vals_list), axis=(1,2))),
            "M": np.copy(np.sum(np.asarray(model.epi_metrics.M.history_vals_list), axis=(1,2))),
            "Mv": np.copy(np.sum(np.asarray(model.epi_metrics.Mv.history_vals_list), axis=(1,2))),
            "param_val": param_val
        })

    # ---- Set up plots ----
    fig, axes = plt.subplots(6, 1, figsize=(10, 12))
    plot_items = [
        ("S", axes[0], blue_colors_list, "Susceptible"),
        ("I", axes[1], red_colors_list, "Infected"),
        ("H", axes[2], red_colors_list, "Hospitalized"),
        ("R", axes[3], blue_colors_list, "Recovered"),
        ("M", axes[4], green_colors_list, "Infection-Induced Immunity M"),
        ("Mv", axes[5], green_colors_list, "Vaccine-Induced Immunity Mv")
    ]

    for i, result in enumerate(sim_results):
        alpha = alpha_list[i]
        marker = None if i == 0 else markers_list[i % len(markers_list)]
        label_suffix = f", {param_name} {result['param_val']}"

        for key, ax, color_list, title in plot_items:
            if key in result:
                ax.plot(result[key], label=f"{key}{label_suffix}", color=color_list[i % len(color_list)],
                        alpha=alpha, marker=marker, markersize=4, markevery=21)
                ax.set_title(title)
                ax.set_xlabel("Day")
                ax.legend()

    fig.tight_layout()
    plt.show()