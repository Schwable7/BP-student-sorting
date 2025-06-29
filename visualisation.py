import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from constants import BEAM_SEARCH, SIMULATED_ANNEALING, EA_DEAP, EA_OWN

# ============================================================================
# Global font‑size constants – adjust once, propagate everywhere
# ============================================================================
FONT_TITLE = 16   # Plot / subplot titles
FONT_LABEL = 16   # Axis labels (x / y)
FONT_TICKS = 14   # Tick‑label text on both axes
FONT_LEGEND = 12  # Legend text

# Optionally push defaults to rcParams so library / third‑party plots match too
plt.rcParams.update({
    "axes.titlesize": FONT_TITLE,
    "axes.labelsize": FONT_LABEL,
    "xtick.labelsize": FONT_TICKS,
    "ytick.labelsize": FONT_TICKS,
    "legend.fontsize": FONT_LEGEND,
})

# ----------------------------------------------------------------------------
# 1. Multi‑metric six‑subplot visualisation
# ----------------------------------------------------------------------------

def visualize_multiplot(costs, size_devs, boys_devs, girls_devs,
                        together_penalties, not_together_penalties,
                        max_iterations, filename, dataset):
    """Six metrics, 3×2 grid."""
    plt.figure(figsize=(15, 12))

    plots = [
        ("Celková fitness", costs, "Celková fitness", "blue"),
        ("Velikost třídy - odchylka", size_devs, "Velikost třídy - odchylka", "orange"),
        ("Chlapci - odchylka", boys_devs, "Chlapci - odchylka", "green"),
        ("Dívky - odchylka", girls_devs, "Dívky - odchylka", "purple"),
        ("Spolu - penalizace", together_penalties, "Spolu - penalizace", "red"),
        ("Ne-spolu - penalizace", not_together_penalties, "Ne-spolu - penalizace", "brown"),
    ]

    for i, (title, data, ylabel, color) in enumerate(plots, start=1):
        plt.subplot(3, 2, i)
        plt.plot(range(max_iterations), data, label=title, color=color)
        plt.xlabel("Iterace", fontsize=FONT_LABEL)
        plt.ylabel(ylabel, fontsize=FONT_LABEL)
        plt.title(f"{title} vs Iterace", fontsize=FONT_TITLE)
        plt.xticks(fontsize=FONT_TICKS)
        plt.yticks(fontsize=FONT_TICKS)
        plt.legend(fontsize=FONT_LEGEND)

    plt.tight_layout()
    plt.savefig(f"output_data/{dataset}/visualisation/{filename}")
    plt.close()

# ----------------------------------------------------------------------------
# 2. Simplified 4‑subplot variant
# ----------------------------------------------------------------------------

def visualize_multiplot_simple(costs, size_devs, boys_devs, girls_devs,
                               max_iterations, filename, dataset):
    """Four metrics, 2×2 grid."""
    plt.figure(figsize=(15, 12))

    plots = [
        ("Celková fitness", costs, "Celková fitness", "blue"),
        ("Velikost třídy - odchylka", size_devs, "Velikost třídy - odchylka", "orange"),
        ("Chlapci - odchylka", boys_devs, "Chlapci - odchylka", "green"),
        ("Dívky - odchylka", girls_devs, "Dívky - odchylka", "purple"),
    ]

    for i, (title, data, ylabel, color) in enumerate(plots, start=1):
        plt.subplot(2, 2, i)
        plt.plot(range(max_iterations), data, label=title, color=color)
        plt.xlabel("Iterations", fontsize=FONT_LABEL)
        plt.ylabel(ylabel, fontsize=FONT_LABEL)
        plt.title(f"{title} vs Iterations", fontsize=FONT_TITLE)
        plt.xticks(fontsize=FONT_TICKS)
        plt.yticks(fontsize=FONT_TICKS)
        plt.legend(fontsize=FONT_LEGEND)

    plt.tight_layout()
    plt.savefig(f"output_data/{dataset}/visualisation/{filename}")
    plt.close()

# ----------------------------------------------------------------------------
# 3. Heatmap of hall‑of‑fame assignments
# ----------------------------------------------------------------------------

def plot_hall_of_fame_heatmap(hall_of_fame, filename, dataset):
    hof_matrix = np.array(hall_of_fame).T  # Students × Solutions

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        hof_matrix,
        cmap="viridis",
        annot=False,
        cbar=True,
        linewidths=0.5,
        xticklabels=[f"Řešení {i + 1}" for i in range(len(hall_of_fame))],
        yticklabels=False,
    )

    plt.xlabel("Hall of Fame", fontsize=FONT_LABEL)
    plt.ylabel("Studenti", fontsize=FONT_LABEL)
    plt.title("Hall of Fame - Rozřazení studentů do tříd (Heatmap)", fontsize=FONT_TITLE)
    plt.savefig(f"output_data/{dataset}/visualisation/{filename}")
    plt.close()

# ----------------------------------------------------------------------------
# 4. Fitness/diversity progress (DEAP logbook version)
# ----------------------------------------------------------------------------

def _line_plot(generations, series, labels, ylabel, title, filename, dataset):
    plt.figure(figsize=(10, 5))
    for data, label in zip(series, labels):
        plt.plot(generations, data, label=label, linestyle="-")

    plt.xlabel("Generace", fontsize=FONT_LABEL)
    plt.ylabel(ylabel, fontsize=FONT_LABEL)
    plt.title(title, fontsize=FONT_TITLE)
    plt.legend(fontsize=FONT_LEGEND)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=FONT_TICKS)
    plt.yticks(fontsize=FONT_TICKS)
    plt.tight_layout()
    plt.savefig(f"output_data/{dataset}/visualisation/{filename}")
    plt.close()


def plot_fitness_progress(logbook, filename, dataset):
    generations = logbook.select("gen")
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")

    _line_plot(
        generations,
        [min_fitness, avg_fitness],
        ["Min Fitness", "Avg Fitness"],
        "Fitness Score",
        "Evolution of Fitness over Generations",
        filename,
        dataset
    )


def plot_diversity_progress(logbook, filename, dataset):
    generations = logbook.select("gen")
    diversity = logbook.select("diversity")

    _line_plot(
        generations,
        [diversity],
        ["Diverzita"],
        "Skóre Diverzity",
        "Diverzita populace v průběhu generací",
        filename,
        dataset
    )

# ----------------------------------------------------------------------------
# 5. Fitness/diversity progress (own logbook dict version)
# ----------------------------------------------------------------------------

def plot_fitness_progress_own(logbook, filename, dataset):
    generations = [log["gen"] for log in logbook]
    min_fitness = [log["min"] for log in logbook]
    avg_fitness = [log["avg"] for log in logbook]

    _line_plot(
        generations,
        [min_fitness, avg_fitness],
        ["Min Fitness", "Avg Fitness"],
        "Fitness Score",
        "Evolution of Fitness over Generations",
        filename,
        dataset
    )


def plot_diversity_progress_own(logbook, filename, dataset):
    generations = [log["gen"] for log in logbook]
    diversity = [log["diversity"] for log in logbook]

    _line_plot(
        generations,
        [diversity],
        ["Diverzita"],
        "Skóre Diverzity",
        "Diverzita populace v průběhu generací",
        filename,
        dataset
    )

# ----------------------------------------------------------------------------
# 6. Relative statistics bar plot
# ----------------------------------------------------------------------------

def plot_relative_statistics(stats: dict, filename: str, dataset: str):
    class_indices = np.arange(len(stats["class"]))
    width = 0.1
    bar_count = 7

    plt.figure(figsize=(14, 6))

    bar_data = [
        ("Celkový počet", stats["total_ratio"], "#1f77b4"),
        ("Chlapci", stats["boys_ratio"], "#4e79a7"),
        ("Dívky", stats["girls_ratio"], "#f28e2c"),
        ("Odklad", stats["deferred_ratio"], "#e15759"),
        ("Spec.uč.potř.", stats["disabilities_ratio"], "#76b7b2"),
        ("Nadání", stats["talented_ratio"], "#59a14f"),
        ("Mateřský jazyk", stats["diff_lang_ratio"], "#edc948"),
    ]

    for i, (label, values, color) in enumerate(bar_data):
        offset = (i - bar_count / 2) * width + width / 2
        plt.bar(class_indices + offset, values, width=width, label=label, color=color, alpha=0.9)

    plt.xlabel("Třída", fontsize=FONT_LABEL)
    plt.ylabel("Proporce", fontsize=FONT_LABEL)
    plt.title("Relativní rozložení charakteristik studentů ve třídách", fontsize=FONT_TITLE)
    plt.xticks(class_indices, stats["class"], rotation=0, fontsize=FONT_TICKS)
    plt.yticks(fontsize=FONT_TICKS)

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    # basic dataset
    if dataset.__contains__("large"):
        plt.ylim(0, 0.3)
    else:
        plt.ylim(0, 0.6)
    # large dataset
    # plt.ylim(0, 0.3)

    plt.legend(title="Charakteristika", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f"output_data/{dataset}/visualisation/{filename}", bbox_inches="tight")
    plt.close()

# ----------------------------------------------------------------------------
# 7. Aggregate fitness comparison (combined + separate)
# ----------------------------------------------------------------------------

def compare_algorithms_graph(all_runs: dict[str, list[list[float]]], filename: str, dataset):
    plt.figure(figsize=(12, 8))

    colors = {
        BEAM_SEARCH: "purple",
        SIMULATED_ANNEALING: "red",
        EA_DEAP: "blue",
        EA_OWN: "green",
    }

    for algo_name, runs in all_runs.items():
        runs_array = np.array(runs)
        mean_vals = runs_array.mean(axis=0)
        std_vals = runs_array.std(axis=0)
        upper = mean_vals + std_vals
        lower = mean_vals - std_vals

        color = colors.get(algo_name, None)
        plt.fill_between(range(len(mean_vals)), lower, upper, alpha=0.2, label=f"{algo_name} ±1 SD", color=color)
        plt.plot(mean_vals, label=f"{algo_name} Průměr", color=color)

    plt.xlabel("Iterace", fontsize=FONT_LABEL)
    plt.ylabel("Fitness", fontsize=FONT_LABEL)
    plt.title("Srovnání algoritmů – vývoj fitness v čase", fontsize=FONT_TITLE)
    plt.legend(fontsize=FONT_LEGEND)
    plt.grid(True)
    plt.xticks(fontsize=FONT_TICKS)
    plt.yticks(fontsize=FONT_TICKS)
    plt.tight_layout()
    plt.savefig(f"output_data/{dataset}/visualisation/{filename}")
    plt.close()



def compare_algorithms_separate_graph(all_runs: dict[str, list[list[float]]], filename: str, dataset):
    """One subplot per algorithm (max 4), fixed axes."""
    colors = {
        BEAM_SEARCH: "purple",
        SIMULATED_ANNEALING: "red",
        EA_DEAP: "blue",
        EA_OWN: "green",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (algo_name, runs) in enumerate(all_runs.items()):
        if idx >= 4:
            break

        ax = axes[idx]
        runs_array = np.array(runs)
        mean_vals = runs_array.mean(axis=0)
        min_vals = runs_array.min(axis=0)
        max_vals = runs_array.max(axis=0)

        x_vals = range(len(mean_vals))
        color = colors.get(algo_name, "gray")

        ax.fill_between(x_vals, min_vals, max_vals, alpha=0.2, color=color, label="min–max")
        ax.plot(x_vals, mean_vals, color=color, label="průměr")

        ax.set_xlim(left=0)

        # basic dataset
        # ax.set_ylim(top=40, bottom=0)

        # skewed dataset
        # ax.set_ylim(top=0.03, bottom=0.003)

        # large dataset
        # ax.set_ylim(top=0.025, bottom=0)

        ax.set_title(algo_name, fontsize=FONT_TITLE)
        ax.set_xlabel("Iterace", fontsize=FONT_LABEL)
        ax.set_ylabel("Fitness", fontsize=FONT_LABEL)
        ax.grid(True)
        ax.legend(fontsize=FONT_LEGEND)
        ax.tick_params(axis="both", labelsize=FONT_TICKS)

    # Remove unused subplots if fewer than 4 algorithms
    for j in range(len(all_runs), 4):
        fig.delaxes(axes[j])

    fig.suptitle("Srovnání algoritmů – vývoj fitness v čase", fontsize=FONT_TITLE)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"output_data/{dataset}/visualisation/{filename}")
    plt.close(fig)


def compare_algorithms_sd(all_runs: dict[str, list[list[float]]], filename: str, dataset):
    """One subplot per algorithm (max 4), fixed axes."""
    colors = {
        BEAM_SEARCH: "purple",
        SIMULATED_ANNEALING: "red",
        EA_DEAP: "blue",
        EA_OWN: "green",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (algo_name, runs) in enumerate(all_runs.items()):
        if idx >= 4:
            break

        ax = axes[idx]
        runs_array = np.array(runs)
        mean_vals = runs_array.mean(axis=0)
        std_vals = runs_array.std(axis=0)
        upper = mean_vals + std_vals
        lower = mean_vals - std_vals

        x_vals = range(len(mean_vals))
        color = colors.get(algo_name, "gray")

        ax.fill_between(x_vals, lower, upper, alpha=0.2, color=color, label="±1 SD")
        ax.plot(x_vals, mean_vals, color=color, label="Průměr")

        ax.set_xlim(left=0)

        # basic dataset
        # ax.set_ylim(top=30, bottom=0)

        # skewed dataset
        # ax.set_ylim(top=0.03, bottom=0.003)

        # large dataset
        # ax.set_ylim(top=0.025, bottom=0)

        ax.set_title(algo_name, fontsize=FONT_TITLE)
        ax.set_xlabel("Iterace", fontsize=FONT_LABEL)
        ax.set_ylabel("Fitness", fontsize=FONT_LABEL)
        ax.grid(True)
        ax.legend(fontsize=FONT_LEGEND)
        ax.tick_params(axis="both", labelsize=FONT_TICKS)

    # Remove unused subplots if fewer than 4 algorithms
    for j in range(len(all_runs), 4):
        fig.delaxes(axes[j])

    fig.suptitle("Srovnání algoritmů – vývoj fitness v čase", fontsize=FONT_TITLE)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"output_data/{dataset}/visualisation/{filename}")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 8. Diversity comparison across algorithms
# ----------------------------------------------------------------------------

def compare_diversity_progress(all_logbooks: dict[str, list], filename: str, dataset):
    plt.figure(figsize=(12, 6))

    colors = {
        EA_DEAP: "blue",
        EA_OWN: "green"
    }

    for algo_name, logbooks in all_logbooks.items():

        runs_array = np.array(logbooks)  # shape: (num_runs, num_generations)
        mean_vals = np.mean(runs_array, axis=0)
        min_vals = np.min(runs_array, axis=0)
        max_vals = np.max(runs_array, axis=0)

        num_generations = runs_array.shape[1]
        generations = list(range(num_generations))  # [0, 1, ..., n]

        color = colors.get(algo_name, None)

        # Fill between min and max
        plt.fill_between(generations, min_vals, max_vals, alpha=0.2, label=f"{algo_name} (rozsah)", color=color)
        # Plot mean
        plt.plot(generations, mean_vals, label=f"{algo_name} (průměr)", color=color)

    plt.xlabel("Generace", fontsize=FONT_LABEL)
    plt.ylabel("Diverzita populace", fontsize=FONT_LABEL)
    plt.title("Srovnání diverzity mezi algoritmy", fontsize=FONT_TITLE)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=FONT_LEGEND)
    plt.xticks(fontsize=FONT_TICKS)
    plt.yticks(fontsize=FONT_TICKS)
    plt.tight_layout()
    plt.savefig(f"output_data/{dataset}/visualisation/{filename}")
    plt.close()
