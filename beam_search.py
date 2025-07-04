import logging
import random
from datetime import datetime, timedelta

from constants import NUM_CLASSES, BEAM_WIDTH, HALL_OF_FAME_SIZE, STUDENTS_PATH, BEAM_ITERATIONS
from data_exporter import export_hall_of_fame
from fitness import fitness, fitness_simple
from helper_functions import convert_classes_to_individual, swap_or_move, compute_relative_statistics, \
    print_relative_stats, print_total_stats
from student_loader import load_students
from visualisation import plot_hall_of_fame_heatmap, plot_relative_statistics, visualize_multiplot, \
    visualize_multiplot_simple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_neighbors(classes: list[list], num_neighbors: int) -> list[list[list]]:
    neighbors = []

    for _ in range(num_neighbors):
        num_classes = len(classes)
        new_classes = swap_or_move(classes, num_classes)

        neighbors.append(new_classes)

    return neighbors


def beam_search(students: list[dict], num_classes: int, beam_width: int, max_iterations: int, dataset) -> tuple[
    list[list], list[float], timedelta, dict]:
    start_time = datetime.now()
    classes = [[] for _ in range(num_classes)]
    random.shuffle(students)

    for student in students:
        classes[random.randint(0, num_classes - 1)].append(student)
    fitness_dict = fitness(classes)
    together_penalty = fitness_dict["together_penalty"]
    not_together_penalty = fitness_dict["not_together_penalty"]
    # fitness_dict = fitness_simple(classes)
    initial_cost = fitness_dict["total_cost"]
    size_dev = fitness_dict["size_dev"]
    boys_dev = fitness_dict["boys_dev"]
    girls_dev = fitness_dict["girls_dev"]

    beam = [(classes, initial_cost, size_dev, boys_dev, girls_dev, together_penalty, not_together_penalty)]
    # beam = [(classes, initial_cost, size_dev, boys_dev, girls_dev)]

    logging.info(f"Starting Beam Search: Initial cost = {initial_cost}")

    costs, size_devs, boys_devs, girls_devs, tgthr_penalties, not_tgthr_penalties, hall_of_fame = [], [], [], [], [], [], []

    best_cost = initial_cost  # Track the best cost found
    best_iteration_found = 0  # Track the iteration number when the best cost is found

    for iteration in range(max_iterations):
        candidates = []
        for state, _, _, _, _, _, _ in beam:
        # for state, _, _, _, _ in beam:
            neighbors = generate_neighbors(state, num_neighbors=beam_width * 2)  # Generate twice as many candidates
            for neighbor in neighbors:
                neighbor_fitness_dict = fitness(neighbor)
                together_penalty = neighbor_fitness_dict["together_penalty"]
                not_together_penalty = neighbor_fitness_dict["not_together_penalty"]
                # neighbor_fitness_dict = fitness_simple(neighbor)
                cost = neighbor_fitness_dict["total_cost"]
                size_dev = neighbor_fitness_dict["size_dev"]
                boys_dev = neighbor_fitness_dict["boys_dev"]
                girls_dev = neighbor_fitness_dict["girls_dev"]

                candidates.append(
                    (neighbor, cost, size_dev, boys_dev, girls_dev, together_penalty, not_together_penalty))
                # candidates.append((neighbor, cost, size_dev, boys_dev, girls_dev))

        # Select the best k solutions
        beam = sorted(candidates, key=lambda x: x[1])[:beam_width]

        # Track best solution
        best_classes, current_best_cost, best_size_dev, best_boys_dev, best_girls_dev, together_penalty, not_together_penalty = \
            beam[0]

        # best_classes, current_best_cost, best_size_dev, best_boys_dev, best_girls_dev = beam[0]

        costs.append(current_best_cost)
        size_devs.append(best_size_dev)
        boys_devs.append(best_boys_dev)
        girls_devs.append(best_girls_dev)
        tgthr_penalties.append(together_penalty)
        not_tgthr_penalties.append(not_together_penalty)

        if not any(hof[0] == classes for hof in hall_of_fame):
            hall_of_fame.append((best_classes, current_best_cost))
            hall_of_fame = sorted(hall_of_fame, key=lambda x: x[1])[:HALL_OF_FAME_SIZE]

        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_iteration_found = iteration  # Store the iteration number of improvement

        if iteration % 100 == 0:
            logging.info(f"Iteration {iteration}: Best cost = {current_best_cost}")

    logging.info(f"Final solution found with cost {best_cost} at iteration {best_iteration_found}")
    end_time = datetime.now()
    execution_time = end_time - start_time
    logging.info(f"Beam Search completed in {execution_time} seconds.")

    hall_of_fame_converted = [convert_classes_to_individual(hof[0]) for hof in hall_of_fame]
    plot_hall_of_fame_heatmap(hall_of_fame_converted, f"BS/HoF_{datetime.now().timestamp()}.png", dataset)

    export_hall_of_fame(hall_of_fame, f"BS/HoF_{datetime.now().timestamp()}.xlsx")

    visualize_multiplot(costs, size_devs, boys_devs, girls_devs, tgthr_penalties, not_tgthr_penalties, max_iterations,
                        f"BS/multi_plot_{datetime.now().timestamp()}.png", dataset)

    # visualize_multiplot_simple(
    #     costs, size_devs, boys_devs, girls_devs, max_iterations,
    #     f"BS/multi_plot_simple_{datetime.now().timestamp()}.png", dataset
    # )

    relative_stats = compute_relative_statistics(students, best_classes)
    print_relative_stats(relative_stats)
    plot_relative_statistics(relative_stats, f"BS/relative_distribution_{datetime.now().timestamp()}.png", dataset)
    # Print total statistics
    print_total_stats(students, best_classes)
    return best_classes, costs, execution_time, relative_stats


if __name__ == "__main__":
    students = load_students(STUDENTS_PATH)
    for i in range(1):
        sorted_classes, costs, execution_time, relative_stats = beam_search(
            students=students,
            num_classes=NUM_CLASSES,
            beam_width=BEAM_WIDTH,
            max_iterations=BEAM_ITERATIONS,
            dataset="basic"  # Replace with actual dataset if needed
        )


