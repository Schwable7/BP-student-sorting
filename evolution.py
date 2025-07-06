import logging
import random
from datetime import datetime, timedelta

import deap.algorithms
import deap.base
import deap.creator
import deap.tools
import pandas as pd
from deap.tools import Logbook

from constants import NUM_CLASSES, POPULATION_SIZE, GENERATIONS, CX_PROB, MUT_PROB, STUDENTS_PATH, TOURNAMENT_SIZE
from fitness import fitness, fitness_simple
from helper_functions import convert_individual_to_classes, compute_population_diversity, compute_relative_statistics, \
    print_relative_stats, print_total_stats
from student_loader import load_students
from visualisation import plot_hall_of_fame_heatmap, plot_fitness_progress, plot_diversity_progress, \
    plot_relative_statistics, visualize_multiplot, visualize_multiplot_simple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define DEAP classes for the optimization
deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
deap.creator.create("Individual", list, fitness=deap.creator.FitnessMin)

fitness_components_log = {}


def create_individual(students: list[dict], num_classes: int) -> list[int]:
    assignment = [random.randint(0, num_classes - 1) for _ in students]
    return deap.creator.Individual(assignment)


def evaluate(individual: list[int], students: list[dict], num_classes: int) -> tuple[float]:
    classes = convert_individual_to_classes(individual, students, num_classes)
    fitness_result = fitness(classes, False)
    # fitness_result = fitness_simple(classes, False)

    # Uložíme přímo do jednotlivce
    individual.fitness_components = fitness_result

    return fitness_result["total_cost"],  # Musí být tuple


mutation_counter = {"count": 0}
crossover_counter = {"count": 0}


def counted_crossover(ind1, ind2):
    crossover_counter["count"] += 1
    return deap.tools.cxOnePoint(ind1, ind2)


def counted_mutation(ind, num_classes: int):
    mutation_counter["count"] += 1
    return mutate(ind, num_classes)


def mutate(individual: list[int], num_classes: int, mut_prob: float) -> tuple[list[int]]:
    """ Mutates an individual by randomly changing a student's class assignment. """
    if random.random() < mut_prob:
        idx = random.randint(0, len(individual) - 1)
        individual[idx] = random.randint(0, num_classes - 1)
    return individual,


def get_mutation_count(_):
    return mutation_counter["count"]


def get_crossover_count(_):
    return crossover_counter["count"]


def evolution(students: list[dict], dataset: str, mut_prob: float, cx_prob: float, tournament_size: int, num_classes: int = NUM_CLASSES, generations: int = GENERATIONS) -> tuple[list[list], Logbook, timedelta, dict]:
    start_time = datetime.now()
    toolbox = deap.base.Toolbox()
    toolbox.register("individual", create_individual, students, num_classes)
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", deap.tools.cxOnePoint)
    toolbox.register("mutate", mutate, num_classes=num_classes, mut_prob=mut_prob)
    toolbox.register("select", deap.tools.selTournament, tournsize=tournament_size)
    toolbox.register("evaluate", evaluate, students=students, num_classes=num_classes)
    toolbox.register("mate", counted_crossover)
    toolbox.register("mutate", counted_mutation, num_classes=num_classes)

    population = toolbox.population(n=POPULATION_SIZE)

    # Apply fitness function to the initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    logging.info("Starting Genetic Algorithm Optimization")

    def extract_best_component(component: str):
        def extractor(population):
            best_ind = min(population, key=lambda ind: ind.fitness.values[0])
            return getattr(best_ind, "fitness_components", {}).get(component, 0.0)  # fallback = 0.0

        return extractor

    stats = deap.tools.Statistics(lambda ind: ind)
    stats.register("min", lambda pop: min(ind.fitness.values[0] for ind in pop))
    stats.register("avg", lambda pop: sum(ind.fitness.values[0] for ind in pop) / len(pop))
    stats.register("diversity", compute_population_diversity)
    stats.register("mutations", get_mutation_count)
    stats.register("crossovers", get_crossover_count)

    # Add tracking for fitness subcomponents
    for comp in [
        "size_dev", "boys_dev", "girls_dev", "deferred_dev",
        "disabilities_dev", "talent_dev", "diff_lang_dev",
        "together_penalty", "not_together_penalty"
    ]:
    # for comp in [
    #     "size_dev", "boys_dev", "girls_dev", "deferred_dev",
    #     "disabilities_dev", "talent_dev", "diff_lang_dev"
    # ]:
        stats.register(f"best_{comp}", extract_best_component(comp))

    hall_of_fame = deap.tools.HallOfFame(5)  # Store the best individual

    population, logbook = deap.algorithms.eaSimple(
        population, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=generations,
        stats=stats, halloffame=hall_of_fame, verbose=True
    )
    end_time = datetime.now()
    execution_time = end_time - start_time
    logging.info(f"Optimization completed in {execution_time}")
    best_individual = hall_of_fame[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plot_fitness_progress(logbook, f"EA/fitness_progress_{timestamp}.png", dataset)
    plot_diversity_progress(logbook, f"EA/diversity_progress_{timestamp}.png", dataset)
    plot_hall_of_fame_heatmap(hall_of_fame, f"EA/HoF_heatmap_{timestamp}.png", dataset)

    best_classes = [[] for _ in range(num_classes)]
    for student_idx, class_idx in enumerate(best_individual):
        best_classes[class_idx].append(students[student_idx])

    final_cost = fitness(best_classes, False)["total_cost"]
    # final_cost = fitness_simple(best_classes, False)["total_cost"]

    logging.info(f"Final solution found with cost {final_cost}")
    df = pd.DataFrame(logbook)
    costs = df["min"].tolist()
    size_devs = df["best_size_dev"].tolist()
    boys_devs = df["best_boys_dev"].tolist()
    girls_devs = df["best_girls_dev"].tolist()
    together_penalties = df["best_together_penalty"].tolist()
    not_together_penalties = df["best_not_together_penalty"].tolist()

    visualize_multiplot(costs, size_devs, boys_devs, girls_devs, together_penalties, not_together_penalties, generations + 1, f"EA/multiplot_{timestamp}.png", dataset)
    # visualize_multiplot_simple(costs, size_devs, boys_devs, girls_devs, generations + 1,f"EA/multi_plot_simple_{timestamp}.png", dataset)

    # Compute print and visualise relative statistics
    relative_stats = compute_relative_statistics(students, best_classes)
    print_relative_stats(relative_stats)
    plot_relative_statistics(relative_stats, f"EA/relative_distribution_{datetime.now().timestamp()}.png", dataset)
    # Print total statistics
    print_total_stats(students, best_classes)
    return best_classes, logbook, execution_time, relative_stats


if __name__ == "__main__":
    students = load_students(STUDENTS_PATH)
    for i in range(1):
        sorted_classes, logbook, execution_time, relative_stats = evolution(students, "basic")


