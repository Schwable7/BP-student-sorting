import copy
import logging
import random
from datetime import datetime

import deap.algorithms
import deap.base
import deap.creator
import deap.tools

from constants import NUM_CLASSES, POPULATION_SIZE, CX_PROB, MUT_PROB
from fitness import fitness
from helper_functions import compute_relative_statistics, \
    print_relative_stats, print_total_stats, compute_population_diversity_2, convert_classes_to_individual
from student_loader import load_students
from visualisation import plot_hall_of_fame_heatmap, plot_fitness_progress, plot_diversity_progress, \
    plot_relative_statistics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define DEAP classes for the optimization
deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
deap.creator.create("Individual", list, fitness=deap.creator.FitnessMin)


def create_individual(students: list[dict], num_classes: int) -> list[list[dict]]:
    students_copy = copy.deepcopy(students)  # Avoid in-place modification
    random.shuffle(students_copy)
    classes = [[] for _ in range(num_classes)]

    for student in students_copy:
        classes[random.randint(0, num_classes - 1)].append(student)

    return deap.creator.Individual(classes)

def evaluate(individual: list[list[dict]]) -> tuple[float]:
    fitness_dict = fitness(individual, False)  # Pass directly as a structured list
    return fitness_dict["total_cost"],  # Fitness must be a tuple


def mutate(individual: list[list[dict]]) -> tuple[list[list[dict]]]:
    individual = copy.deepcopy(individual)  # Clone to avoid side effects
    if random.random() < MUT_PROB:
        class_a, class_b = random.sample(range(len(individual)), 2)
        if individual[class_a]:
            student = random.choice(individual[class_a])
            individual[class_a].remove(student)
            individual[class_b].append(student)
    return individual,


def crossover(ind1: list[list[dict]], ind2: list[list[dict]]) -> tuple[list[list[dict]], list[list[dict]]]:
    ind1 = copy.deepcopy(ind1)
    ind2 = copy.deepcopy(ind2)

    if random.random() < CX_PROB:
        class_idx = random.randint(0, len(ind1) - 1)
        num_students = min(len(ind1[class_idx]), len(ind2[class_idx]), 3)

        if num_students > 0:
            students_from_1 = random.sample(ind1[class_idx], num_students)
            students_from_2 = random.sample(ind2[class_idx], num_students)

            for student in students_from_1:
                ind1[class_idx].remove(student)
            for student in students_from_2:
                ind2[class_idx].remove(student)

            ind1[class_idx].extend(students_from_2)
            ind2[class_idx].extend(students_from_1)

    return ind1, ind2


mutation_counter = {"count": 0}
crossover_counter = {"count": 0}


def counted_crossover(ind1, ind2):
    crossover_counter["count"] += 1
    return crossover(ind1, ind2)


def counted_mutation(ind):
    mutation_counter["count"] += 1
    return mutate(ind)


def get_mutation_count(_):
    return mutation_counter["count"]


def get_crossover_count(_):
    return crossover_counter["count"]


def evolution(students: list[dict]) -> list[list]:
    toolbox = deap.base.Toolbox()
    toolbox.register("individual", create_individual, students, NUM_CLASSES)
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", deap.tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    population = toolbox.population(n=POPULATION_SIZE)

    # Apply fitness function to the initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    logging.info("Starting Genetic Algorithm Optimization")

    stats = deap.tools.Statistics(lambda ind: ind)
    stats.register("min", lambda pop: min(ind.fitness.values[0] for ind in pop))
    stats.register("avg", lambda pop: sum(ind.fitness.values[0] for ind in pop) / len(pop))
    stats.register("diversity", compute_population_diversity_2)

    hall_of_fame = deap.tools.HallOfFame(5)  # Store the best individual

    population, logbook = deap.algorithms.eaSimple(
        population, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB, ngen=50,
        stats=stats, halloffame=hall_of_fame, verbose=True
    )

    hall_of_fame_converted = [convert_classes_to_individual(hof) for hof in hall_of_fame]

    best_individual = copy.deepcopy(hall_of_fame[0])
    plot_fitness_progress(logbook, f"EA2/fitness_progress_{datetime.now().timestamp()}.png")
    plot_diversity_progress(logbook, f"EA2/diversity_progress_{datetime.now().timestamp()}.png")
    plot_hall_of_fame_heatmap(hall_of_fame_converted, filename=f"EA2/HoF_heatmap_{datetime.now().timestamp()}.png")

    logging.info(f"Final solution found with cost {best_individual.fitness.values[0]}")

    return best_individual  # Now it returns the structured class list


def validate_solution(original_students: list[dict], sorted_classes: list[list[dict]]) -> bool:
    original_ids = {s["id"] for s in original_students}
    result_ids = [s["id"] for class_ in sorted_classes for s in class_]

    if len(result_ids) != len(original_ids):
        logging.error(f"Mismatch in total number of students: expected {len(original_ids)}, got {len(result_ids)}")
        return False

    result_id_counts = {}
    for sid in result_ids:
        result_id_counts[sid] = result_id_counts.get(sid, 0) + 1

    duplicates = [sid for sid, count in result_id_counts.items() if count > 1]
    missing = list(original_ids - set(result_ids))

    if duplicates:
        logging.error(f"Duplicate student IDs found: {duplicates}")
    if missing:
        logging.error(f"Missing student IDs: {missing}")

    return not duplicates and not missing


if __name__ == "__main__":
    students = load_students("input_data/students_03.xlsx")
    sorted_classes = evolution(students)
    if not validate_solution(students, sorted_classes):
        logging.error("Final result is invalid! Students were lost or duplicated.")
    else:
        logging.info("Validation passed: each student appears exactly once.")

    # Compute print and visualise relative statistics
    relative_stats = compute_relative_statistics(sorted_classes)
    print_relative_stats(relative_stats)
    plot_relative_statistics(relative_stats, f"EA2/relative_distribution_{datetime.now().timestamp()}.png")

    # Print total statistics
    print_total_stats(students, sorted_classes)
