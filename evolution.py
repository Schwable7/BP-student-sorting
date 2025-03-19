import random
import logging
from datetime import datetime

import deap.base
import deap.creator
import deap.tools
import deap.algorithms

from constants import NUM_CLASSES, POPULATION_SIZE, GENERATIONS, CX_PROB, MUT_PROB
from fitness import fitness
from helper_functions import convert_individual_to_classes, compute_population_diversity, compute_relative_statistics, \
    print_relative_stats, print_total_stats
from student_loader import load_students
from visualisation import plot_hall_of_fame_heatmap, plot_fitness_progress, plot_diversity_progress, \
    plot_mutation_crossover, plot_relative_statistics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define DEAP classes for the optimization
deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
deap.creator.create("Individual", list, fitness=deap.creator.FitnessMin)


def create_individual(students: list[dict], num_classes: int) -> list[int]:
    assignment = [random.randint(0, num_classes - 1) for _ in students]
    return deap.creator.Individual(assignment)


def evaluate(individual: list[int], students: list[dict], num_classes: int) -> tuple[float]:
    classes = convert_individual_to_classes(individual, students, num_classes)

    score, size_dev, boys_dev, girls_dev = fitness(classes, False)
    return score,  # Must be a tuple


mutation_counter = {"count": 0}
crossover_counter = {"count": 0}


def counted_crossover(ind1, ind2):
    crossover_counter["count"] += 1
    return deap.tools.cxOnePoint(ind1, ind2)


def counted_mutation(ind):
    mutation_counter["count"] += 1
    return mutate(ind, NUM_CLASSES)


def mutate(individual: list[int], num_classes: int) -> tuple[list[int]]:
    """ Mutates an individual by randomly changing a student's class assignment. """
    if random.random() < MUT_PROB:
        idx = random.randint(0, len(individual) - 1)
        individual[idx] = random.randint(0, num_classes - 1)
    return individual,


def get_mutation_count(_):
    return mutation_counter["count"]


def get_crossover_count(_):
    return crossover_counter["count"]


def evolution(students: list[dict]) -> list[list]:

    toolbox = deap.base.Toolbox()
    toolbox.register("individual", create_individual, students, NUM_CLASSES)
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", deap.tools.cxOnePoint)
    toolbox.register("mutate", mutate, num_classes=NUM_CLASSES)
    toolbox.register("select", deap.tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, students=students, num_classes=NUM_CLASSES)
    toolbox.register("mate", counted_crossover)
    toolbox.register("mutate", counted_mutation)

    population = toolbox.population(n=POPULATION_SIZE)

    # Apply fitness function to the initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    logging.info("Starting Genetic Algorithm Optimization")

    stats = deap.tools.Statistics(lambda ind: ind)
    stats.register("min", lambda pop: min(ind.fitness.values[0] for ind in pop))
    stats.register("avg", lambda pop: sum(ind.fitness.values[0] for ind in pop) / len(pop))
    stats.register("diversity", compute_population_diversity)
    stats.register("mutations", get_mutation_count)
    stats.register("crossovers", get_crossover_count)

    hall_of_fame = deap.tools.HallOfFame(5)  # Store the best individual

    population, logbook = deap.algorithms.eaSimple(
        population, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB, ngen=GENERATIONS,
        stats=stats, halloffame=hall_of_fame, verbose=True
    )

    best_individual = hall_of_fame[0]
    plot_fitness_progress(logbook, f"EA/fitness_progress_{datetime.now().timestamp()}.png")
    plot_diversity_progress(logbook, f"EA/diversity_progress_{datetime.now().timestamp()}.png")
    plot_mutation_crossover(logbook, f"EA/mutation_crossover_{datetime.now().timestamp()}.png")
    plot_hall_of_fame_heatmap(hall_of_fame, filename=f"EA/HoF_heatmap_{datetime.now().timestamp()}.png")

    best_classes = [[] for _ in range(NUM_CLASSES)]
    for student_idx, class_idx in enumerate(best_individual):
        best_classes[class_idx].append(students[student_idx])

    final_cost = fitness(best_classes, False)[0]
    logging.info(f"Final solution found with cost {final_cost}")

    return best_classes


if __name__ == "__main__":
    students = load_students("input_data/students_02.xlsx")
    sorted_classes = evolution(students)

    # Compute print and visualise relative statistics
    relative_stats = compute_relative_statistics(sorted_classes)
    print_relative_stats(relative_stats)
    plot_relative_statistics(relative_stats, f"EA/relative_distribution_{datetime.now().timestamp()}.png")

    # Print total statistics
    print_total_stats(students, sorted_classes)
