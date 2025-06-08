import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_multiplot(costs, size_devs, boys_devs, girls_devs, together_penalties, not_together_penalties, max_iterations, filename):
    plt.figure(figsize=(15, 12))

    # Fitness Cost
    plt.subplot(3, 2, 1)
    plt.plot(range(max_iterations), costs, label="Fitness Cost", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Cost")
    plt.title("Fitness Cost vs Iterations")
    plt.legend()

    # Size Deviation
    plt.subplot(3, 2, 2)
    plt.plot(range(max_iterations), size_devs, label="Size Deviation", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Size Deviation")
    plt.title("Class Size Deviation vs Iterations")
    plt.legend()

    # Boys Deviation
    plt.subplot(3, 2, 3)
    plt.plot(range(max_iterations), boys_devs, label="Boys Deviation", color="green")
    plt.xlabel("Iterations")
    plt.ylabel("Boys Deviation")
    plt.title("Boys Deviation vs Iterations")
    plt.legend()

    # Girls Deviation
    plt.subplot(3, 2, 4)
    plt.plot(range(max_iterations), girls_devs, label="Girls Deviation", color="purple")
    plt.xlabel("Iterations")
    plt.ylabel("Girls Deviation")
    plt.title("Girls Deviation vs Iterations")
    plt.legend()

    # Together Penalties
    plt.subplot(3, 2, 5)
    plt.plot(range(max_iterations), together_penalties, label="Together penalty", color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Together penalty")
    plt.title("Together penalty vs Iterations")
    plt.legend()

    # Not together penalties
    plt.subplot(3, 2, 6)
    plt.plot(range(max_iterations), not_together_penalties, label="Not Together penalty", color="brown")
    plt.xlabel("Iterations")
    plt.ylabel("Not Together penalty")
    plt.title("Not Together penalty vs Iterations")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"output_data/visualisation/{filename}")


def visualize_multiplot_simple(costs, size_devs, boys_devs, girls_devs, max_iterations, filename):
    plt.figure(figsize=(15, 12))

    # Fitness Cost
    plt.subplot(2, 2, 1)
    plt.plot(range(max_iterations), costs, label="Fitness Cost", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Cost")
    plt.title("Fitness Cost vs Iterations")
    plt.legend()

    # Size Deviation
    plt.subplot(2, 2, 2)
    plt.plot(range(max_iterations), size_devs, label="Size Deviation", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Size Deviation")
    plt.title("Class Size Deviation vs Iterations")
    plt.legend()

    # Boys Deviation
    plt.subplot(2, 2, 3)
    plt.plot(range(max_iterations), boys_devs, label="Boys Deviation", color="green")
    plt.xlabel("Iterations")
    plt.ylabel("Boys Deviation")
    plt.title("Boys Deviation vs Iterations")
    plt.legend()

    # Girls Deviation
    plt.subplot(2, 2, 4)
    plt.plot(range(max_iterations), girls_devs, label="Girls Deviation", color="purple")
    plt.xlabel("Iterations")
    plt.ylabel("Girls Deviation")
    plt.title("Girls Deviation vs Iterations")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"output_data/visualisation/{filename}")


def visualise_individual(individual: list, filename):
    plt.figure(figsize=(10, 8))
    plt.plot(individual)
    plt.savefig(f"output_data/visualisation/{filename}")


def visualize_bs(costs, size_devs, boys_devs, girls_devs, max_iterations, filename):
    plt.figure(figsize=(12, 8))

    # Fitness Cost
    plt.subplot(2, 2, 1)
    plt.plot(range(max_iterations), costs, label="Fitness Cost", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Cost")
    plt.title("Fitness Cost vs Iterations")
    plt.legend()

    # Size Deviation
    plt.subplot(2, 2, 2)
    plt.plot(range(max_iterations), size_devs, label="Size Deviation", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Size Deviation")
    plt.title("Class Size Deviation vs Iterations")
    plt.legend()

    # Boys Deviation
    plt.subplot(2, 2, 3)
    plt.plot(range(max_iterations), boys_devs, label="Boys Deviation", color="green")
    plt.xlabel("Iterations")
    plt.ylabel("Boys Deviation")
    plt.title("Boys Deviation vs Iterations")
    plt.legend()

    # Girls Deviation
    plt.subplot(2, 2, 4)
    plt.plot(range(max_iterations), girls_devs, label="Girls Deviation", color="purple")
    plt.xlabel("Iterations")
    plt.ylabel("Girls Deviation")
    plt.title("Girls Deviation vs Iterations")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"output_data/visualisation/{filename}")


def plot_hall_of_fame(hall_of_fame, num_students, filename):
    plt.figure(figsize=(20, 10))

    # Assign unique colors for each solution in the Hall of Fame
    colors = plt.cm.get_cmap('tab10', len(hall_of_fame))

    for idx, individual in enumerate(hall_of_fame):
        x = np.arange(num_students)  # Student indices
        y = individual  # Assigned classes
        plt.scatter(x, y, label=f"Solution {idx + 1}", alpha=0.6, color=colors(idx), s=20)

    plt.xlabel("Students")
    plt.ylabel("Classes")
    plt.title("Hall of Fame - Student Class Assignments")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(f"output_data/visualisation/{filename}")


def plot_hall_of_fame_heatmap(hall_of_fame, filename):
    # Convert the hall of fame list into a NumPy array (shape: num_students x num_solutions)
    hof_matrix = np.array(hall_of_fame).T  # Transpose to have students as rows, solutions as columns

    plt.figure(figsize=(10, 8))

    # Create heatmap using Seaborn
    sns.heatmap(hof_matrix, cmap="viridis", annot=False, cbar=True, linewidths=0.5,
                xticklabels=[f"Sol {i + 1}" for i in range(len(hall_of_fame))],
                yticklabels=False)  # Hide student labels for large data

    plt.xlabel("Hall of Fame Solutions")
    plt.ylabel("Students")
    plt.title("Hall of Fame - Student Class Assignments (Heatmap)")
    plt.savefig(f"output_data/visualisation/{filename}")


def plot_fitness_progress(logbook, filename):
    generations = logbook.select("gen")
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")

    plt.figure(figsize=(10, 5))
    plt.plot(generations, min_fitness, label="Min Fitness", linestyle="-")
    plt.plot(generations, avg_fitness, label="Avg Fitness", linestyle="-")

    plt.xlabel("Generations")
    plt.ylabel("Fitness Score")
    plt.title("Evolution of Fitness over Generations")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"output_data/visualisation/{filename}")


def plot_diversity_progress(logbook, filename):
    generations = logbook.select("gen")
    diversity = logbook.select("diversity")

    plt.figure(figsize=(10, 5))
    plt.plot(generations, diversity, linestyle="-", color="purple")

    plt.xlabel("Generations")
    plt.ylabel("Diversity Score")
    plt.title("Diversity of Population Over Generations")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"output_data/visualisation/{filename}")



def plot_fitness_progress_own(logbook, filename):
    generations = [log["gen"] for log in logbook]
    min_fitness = [log["min"] for log in logbook]
    avg_fitness = [log["avg"] for log in logbook]

    plt.figure(figsize=(10, 5))
    plt.plot(generations, min_fitness, label="Min Fitness", linestyle="-")
    plt.plot(generations, avg_fitness, label="Avg Fitness", linestyle="-")

    plt.xlabel("Generations")
    plt.ylabel("Fitness Score")
    plt.title("Evolution of Fitness over Generations")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"output_data/visualisation/{filename}")


def plot_diversity_progress_own(logbook, filename):
    generations = [log["gen"] for log in logbook]
    diversity = [log["diversity"] for log in logbook]

    plt.figure(figsize=(10, 5))
    plt.plot(generations, diversity, linestyle="-", color="purple")

    plt.xlabel("Generations")
    plt.ylabel("Diversity Score")
    plt.title("Diversity of Population Over Generations")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"output_data/visualisation/{filename}")


def plot_mutation_crossover(logbook, filename):
    generations = logbook.select("gen")
    mutations = logbook.select("mutations")
    crossovers = logbook.select("crossovers")

    plt.figure(figsize=(10, 5))
    plt.plot(generations, mutations, label="Mutations", linestyle="-", color="red")
    plt.plot(generations, crossovers, label="Crossovers", linestyle="-", color="blue")

    plt.xlabel("Generations")
    plt.ylabel("Count")
    plt.title("Mutation and Crossover Rates Over Generations")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"output_data/visualisation/{filename}")


def plot_relative_statistics(stats: dict, filename: str):
    class_indices = np.arange(len(stats["class"]))  # X-axis positions
    width = 0.4  # Width of bars

    plt.figure(figsize=(10, 5))

    plt.bar(class_indices - width / 2, stats["boys_ratio"], width=width, label="Boys Ratio", color="blue", alpha=0.7)
    plt.bar(class_indices + width / 2, stats["girls_ratio"], width=width, label="Girls Ratio", color="pink", alpha=0.7)

    plt.xlabel("Class")
    plt.ylabel("Proportion")
    plt.title("Relative Boys/Girls Distribution in Classes")
    plt.xticks(class_indices, stats["class"])
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.savefig(f"output_data/visualisation/{filename}")


def visualize_all(costs_sa, costs_bs, logbook, max_iterations_sa, max_iterations_bs, filename):
    plt.figure(figsize=(12, 8))

    # Fitness Cost SA
    plt.subplot(3, 1, 1)
    plt.plot(range(max_iterations_sa), costs_sa, label="Fitness Cost - SA", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Cost SA")
    plt.title("Fitness Cost SA vs Iterations")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Fitness Cost BS
    plt.subplot(3, 1, 2)
    plt.plot(range(max_iterations_bs), costs_bs, label="Fitness Cost - BS", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Cost BS")
    plt.title("Fitness Cost BS vs Iterations")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Fitness Progress EA
    generations = logbook.select("gen")
    min_fitness = logbook.select("min")

    plt.subplot(3, 1, 3)
    plt.plot(generations, min_fitness, label="Min Fitness", linestyle="-")

    plt.xlabel("Generations")
    plt.ylabel("Fitness Score")
    plt.title("Evolution of Fitness over Generations")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"output_data/visualisation/{filename}")
