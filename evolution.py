import random
import logging
from datetime import datetime

import deap.base
import deap.creator
import deap.tools
import deap.algorithms

from constants import NUM_CLASSES, POPULATION_SIZE, GENERATIONS, CX_PROB, MUT_PROB
from fitness import fitness
from student_loader import load_students
from visualisation import visualize_sa, visualise_individual

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define DEAP classes for the optimization
deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
deap.creator.create("Individual", list, fitness=deap.creator.FitnessMin)


def create_individual(students: list[dict], num_classes: int) -> list[int]:
    assignment = [random.randint(0, num_classes - 1) for _ in students]
    return deap.creator.Individual(assignment)


def evaluate(individual: list[int], students: list[dict], num_classes: int) -> tuple[float]:
    classes = [[] for _ in range(num_classes)]

    for student_idx, class_idx in enumerate(individual):
        classes[class_idx].append(students[student_idx])

    score, size_dev, boys_dev, girls_dev = fitness(classes)
    return score,  # Must be a tuple


def mutate(individual: list[int], num_classes: int) -> tuple[list[int]]:
    """ Mutates an individual by randomly changing a student's class assignment. """
    if random.random() < MUT_PROB:
        idx = random.randint(0, len(individual) - 1)
        individual[idx] = random.randint(0, num_classes - 1)
    return individual,


def evolution() -> list[list]:
    students = load_students("input_data/students_02.xlsx")

    toolbox = deap.base.Toolbox()
    toolbox.register("individual", create_individual, students, NUM_CLASSES)
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", deap.tools.cxOnePoint)
    toolbox.register("mutate", mutate, num_classes=NUM_CLASSES)
    toolbox.register("select", deap.tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, students=students, num_classes=NUM_CLASSES)

    population = toolbox.population(n=POPULATION_SIZE)

    # Apply fitness function to the initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    logging.info("Starting Genetic Algorithm Optimization")

    stats = deap.tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", min)
    stats.register("avg", lambda x: sum(x) / len(x))

    hall_of_fame = deap.tools.HallOfFame(5)  # Store the best individual

    population, logbook = deap.algorithms.eaSimple(
        population, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB, ngen=GENERATIONS,
        stats=stats, halloffame=hall_of_fame, verbose=True
    )

    best_individual = hall_of_fame[0]
    visualise_individual(population, f"best_individual_{datetime.now().timestamp()}.png")
    best_classes = [[] for _ in range(NUM_CLASSES)]
    for student_idx, class_idx in enumerate(best_individual):
        best_classes[class_idx].append(students[student_idx])

    final_cost = fitness(best_classes)[0]
    logging.info(f"Final solution found with cost {final_cost}")

    return best_classes


if __name__ == "__main__":
    sorted_classes = evolution()
    for i, cls in enumerate(sorted_classes):
        print(f"Class {i + 1}: {len(cls)} students, "
              f"Boys = {sum(1 for s in cls if s['pohlavi'] == 'K')}, "
              f"Girls = {sum(1 for s in cls if s['pohlavi'] == 'D')}")
