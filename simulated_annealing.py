import random
import math
import logging
from datetime import datetime

from constants import NUM_CLASSES, INITIAL_TEMP, COOLING_RATE, MAX_ITERATIONS, HALL_OF_FAME_SIZE
from data_exporter import export_hall_of_fame
from fitness import fitness, fitness_simple
from helper_functions import convert_classes_to_individual, swap_or_move, compute_relative_statistics, \
    print_relative_stats, print_total_stats
from student_loader import load_students
from visualisation import visualize_multiplot, plot_hall_of_fame_heatmap, plot_relative_statistics, \
    visualize_multiplot_simple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def simulated_annealing(students: list[dict], num_classes: int, initial_temp, cooling_rate,
                        max_iterations) -> tuple[list[list], list[float]]:
    # Randomly assign students to classes
    classes = [[] for _ in range(num_classes)]
    random.shuffle(students)

    for student in students:
        classes[random.randint(0, num_classes - 1)].append(student)

    fitness_dict = fitness(classes)  # Calculate initial cost
    together_penalty = fitness_dict["together_penalty"]
    not_together_penalty = fitness_dict["not_together_penalty"]
    # fitness_dict = fitness_simple(classes)
    current_score = fitness_dict["total_cost"]
    size_dev = fitness_dict["size_dev"]
    boys_dev = fitness_dict["boys_dev"]
    girls_dev = fitness_dict["girls_dev"]

    temp = initial_temp

    logging.info(f"Starting Simulated Annealing: Initial cost = {current_score}")

    costs, size_devs, boys_devs, girls_devs, tgthr_penalties, not_tgthr_penalties, probabilities, temperatures, hall_of_fame = [], [], [], [], [], [], [], [], []

    for iteration in range(max_iterations):
        new_classes = swap_or_move(classes, num_classes)

        new_fitness_dict = fitness(new_classes)  # Calculate new cost
        new_together_penalty = new_fitness_dict["together_penalty"]
        new_not_together_penalty = new_fitness_dict["not_together_penalty"]
        # new_fitness_dict = fitness_simple(new_classes)  # Calculate new cost
        new_score = new_fitness_dict["total_cost"]
        new_size_dev = new_fitness_dict["size_dev"]
        new_boys_dev = new_fitness_dict["boys_dev"]
        new_girls_dev = new_fitness_dict["girls_dev"]

        # Accept new solution if it is better or with probability exp((current_score - new_score) / temp)
        acceptance_probability = (current_score - new_score) / temp
        # acceptance_probability = math.exp((current_score - new_score) / temp)
        probabilities.append(acceptance_probability)

        if new_score < current_score or acceptance_probability > random.random():
            classes, current_score, size_dev, boys_dev, girls_dev, together_penalty, not_together_penalty = (
                new_classes, new_score, new_size_dev, new_boys_dev, new_girls_dev, new_together_penalty,
                new_not_together_penalty
            )
            # classes, current_score, size_dev, boys_dev, girls_dev = (
            #     new_classes, new_score, new_size_dev, new_boys_dev, new_girls_dev
            # )
            # logging.info(f"Iteration {iteration}: Accepted new state with cost {new_score}, Temperature = {temp:.2f}")

            if not any(hof[0] == classes for hof in hall_of_fame):
                hall_of_fame.append((classes, new_score))
                hall_of_fame = sorted(hall_of_fame, key=lambda x: x[1])[:HALL_OF_FAME_SIZE]

        # else:
        # logging.info(f"Iteration {iteration}: Rejected new state with cost {new_score}, Temperature = {temp:.2f}")

        costs.append(current_score)
        size_devs.append(size_dev)
        boys_devs.append(boys_dev)
        girls_devs.append(girls_dev)
        temperatures.append(temp)
        tgthr_penalties.append(together_penalty)
        not_tgthr_penalties.append(not_together_penalty)

        temp *= cooling_rate

        # if iteration % 100 == 0:
        # logging.info(f"Iteration {iteration}: Current best cost = {current_score}, Temperature = {temp:.2f}")

    logging.info(f"Final solution found with cost {current_score}")

    # Convert and plot hall of fame
    hall_of_fame_converted = [convert_classes_to_individual(hof[0]) for hof in hall_of_fame]
    plot_hall_of_fame_heatmap(hall_of_fame_converted, f"SA/HoF_{datetime.now().timestamp()}.png")

    export_hall_of_fame(hall_of_fame, f"SA/HoF_{datetime.now().timestamp()}.xlsx")

    visualize_multiplot(
        costs, size_devs, boys_devs, girls_devs, tgthr_penalties, not_tgthr_penalties, max_iterations,
        f"SA/multi_plot_{datetime.now().timestamp()}.png"
    )

    # visualize_multiplot_simple(
    #     costs, size_devs, boys_devs, girls_devs, max_iterations,
    #     f"SA/multi_plot_simple_{datetime.now().timestamp()}.png"
    # )

    return classes, costs


if __name__ == "__main__":
    students = load_students("input_data/students_04.xlsx")
    for i in range(1):
        sorted_classes, costs = simulated_annealing(
            students=students,
            num_classes=NUM_CLASSES,
            initial_temp=INITIAL_TEMP,
            cooling_rate=COOLING_RATE,
            max_iterations=MAX_ITERATIONS
        )

        # Compute print and visualise relative statistics
        relative_stats = compute_relative_statistics(sorted_classes)
        print_relative_stats(relative_stats)
        plot_relative_statistics(relative_stats, f"SA/relative_distribution_{datetime.now().timestamp()}.png")

        # Print total statistics
        print_total_stats(students, sorted_classes)
