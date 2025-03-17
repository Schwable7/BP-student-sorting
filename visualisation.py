import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from constants import GENERATIONS


def visualize_sa(costs, size_devs, boys_devs, girls_devs, probabilities, temperatures, max_iterations, filename):
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

    # Acceptance Probability
    plt.subplot(3, 2, 5)
    plt.plot(range(len(probabilities)), probabilities, label="Acceptance Probability", color="brown")
    plt.xlabel("Iterations")
    plt.ylabel("Acceptance Probability")
    plt.title("Decrease in Acceptance Probability Over Time")
    plt.legend()

    # Temperature
    plt.subplot(3, 2, 6)
    plt.plot(range(max_iterations), temperatures, label="Temperature", color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Temperature")
    plt.title("Temperature vs Iterations")
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
    """
    Plots the Hall of Fame solutions with students on X-axis and classes on Y-axis.

    :param filename: Name of the output file.
    :param hall_of_fame: List of best individuals (assignments).
    :param num_students: Number of students.
    """
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
    """
    Plots a heatmap of the Hall of Fame solutions showing student-class assignments.

    :param hall_of_fame: List of best individuals (assignments).
    :param filename: Name of the output file.
    """
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
    """
    Plots the evolution of fitness values (min and avg fitness over generations).

    :param logbook: Logbook object from DEAP containing evolution history.
    """
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


def print_hall_of_fame(hall_of_fame):
    for i, (classes, score) in enumerate(hall_of_fame):
        print(f"\n--- Hall of Fame Solution {i + 1} --- (Fitness Score: {score})")

        # Create a DataFrame to show class assignments
        data = []
        for class_idx, cls in enumerate(classes):
            for student in cls:
                data.append([class_idx + 1, student["student_uid"], student["pohlavi"]])  # Add more attributes if needed

        df = pd.DataFrame(data, columns=["Class", "Student ID", "Gender"])

        # Display using Pandas formatting
        print(df.to_string(index=False))  # Print without row index for clarity
