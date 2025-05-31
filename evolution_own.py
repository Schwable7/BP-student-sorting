import copy
import logging
import random
from datetime import datetime

from constants import MUT_PROB, CX_PROB, POPULATION_SIZE, NUM_CLASSES, GENERATIONS, TOURNAMENT_SIZE, ID, ELITE_COUNT, \
    HALL_OF_FAME_SIZE
from fitness import fitness
from helper_functions import compute_relative_statistics, print_relative_stats, print_total_stats, calculate_diversity, \
    convert_classes_to_individual
from student_loader import load_students
from visualisation import plot_relative_statistics, plot_hall_of_fame_heatmap, plot_fitness_progress_own, \
    plot_diversity_progress_own


def generate_random_solution(students: list[dict]) -> list[list[dict]]:
    classes = [[] for _ in range(NUM_CLASSES)]
    for student in students:
        random.choice(classes).append(student)
    return classes


def tournament_selection(pop_with_fitness: list[tuple[list[list[dict]], float]], k: int) -> list[list[list[dict]]]:
    selected = []
    for _ in range(POPULATION_SIZE):
        competitors = random.sample(pop_with_fitness, k)
        winner = min(competitors, key=lambda x: x[1])
        selected.append(copy.deepcopy(winner[0]))
    return selected


def crossover(parent1: list[list[dict]], parent2: list[list[dict]]) -> tuple[list[list[dict]], list[list[dict]]]:
    student_map = {}

    for cls_index, cls in enumerate(parent1):
        for student in cls:
            sid = student[ID]
            student_map.setdefault(sid, []).append(cls_index)

    for cls_index, cls in enumerate(parent2):
        for student in cls:
            sid = student[ID]
            student_map.setdefault(sid, []).append(cls_index)

    def build_child():
        classes = [[] for _ in range(NUM_CLASSES)]
        for sid, choices in student_map.items():
            chosen_class = max(set(choices), key=choices.count) if choices else random.randint(0, NUM_CLASSES - 1)
            student = next(s for cls in (parent1 + parent2) for s in cls if s[ID] == sid)
            classes[chosen_class].append(student)
        return classes

    return build_child(), build_child()


def mutate(individual: list[list[dict]]):
    if random.random() < 0.5:
        # Swap two students between two different classes
        class_indices = random.sample(range(NUM_CLASSES), 2)
        cls1, cls2 = individual[class_indices[0]], individual[class_indices[1]]
        if cls1 and cls2:
            s1, s2 = random.choice(cls1), random.choice(cls2)
            cls1.remove(s1)
            cls2.remove(s2)
            cls1.append(s2)
            cls2.append(s1)
    else:
        # Move student from largest to smallest class
        largest = max(individual, key=len)
        smallest = min(individual, key=len)
        if largest and largest != smallest:
            student = random.choice(largest)
            largest.remove(student)
            smallest.append(student)


def balance_classes(classes: list[list[dict]]) -> list[list[dict]]:
    student_ids = set()
    unique_students = []

    for cls in classes:
        for student in cls:
            if student[ID] not in student_ids:
                student_ids.add(student[ID])
                unique_students.append(student)

    # Clear and reassign randomly, but preserve number of classes
    new_classes = [[] for _ in range(NUM_CLASSES)]
    for student in unique_students:
        random.choice(new_classes).append(student)

    return new_classes


def evolutionary_algorithm(students: list[dict]):
    # Initialize population
    population = [generate_random_solution(students) for _ in range(POPULATION_SIZE)]

    best_solution = None
    best_fitness = float('inf')

    logbook = []  # will hold dictionaries per generation
    hall_of_fame = []  # will hold best solutions (student → class ID) over time

    for generation in range(GENERATIONS):
        # Evaluate fitness of population
        evaluated = [(individual, fitness(individual, False)["total_cost"]) for individual in population]
        evaluated.sort(key=lambda x: x[1])
        # Select elites
        elites = [copy.deepcopy(individual) for individual, _ in evaluated[:ELITE_COUNT]]

        # Track best solution
        gen_best_fitness = evaluated[0][1]
        gen_avg_fitness = sum(f for _, f in evaluated) / len(evaluated)
        if evaluated[0][1] < best_fitness:
            best_solution = evaluated[0][0]
            best_fitness = evaluated[0][1]

        # Log fitness & diversity
        logbook.append({
            "gen": generation,
            "min": gen_best_fitness,
            "avg": gen_avg_fitness,
            "diversity": calculate_diversity(population)
        })

        # Save hall of fame mapping
        hof_mapping = convert_classes_to_individual(evaluated[0][0])
        hall_of_fame.append(hof_mapping)
        hall_of_fame = sorted(hall_of_fame, key=lambda x: x[1])[:HALL_OF_FAME_SIZE]

        print(
            f"Generation {generation} - Best fitness: {evaluated[0][1]} Average fitness: {sum(f for _, f in evaluated) / len(evaluated)}")

        # Selection (Tournament)
        selected = tournament_selection(evaluated, TOURNAMENT_SIZE)

        # Create next generation
        next_generation = []

        while len(next_generation) < POPULATION_SIZE:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)

            if random.random() < CX_PROB:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            if random.random() < MUT_PROB:
                mutate(child1)
            if random.random() < MUT_PROB:
                mutate(child2)

            next_generation.extend([child1, child2])

        next_generation = elites + next_generation[:POPULATION_SIZE - ELITE_COUNT]
        population = next_generation

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    plot_fitness_progress_own(logbook, f"EA2/fitness_progress_{timestamp}.png")
    plot_diversity_progress_own(logbook, f"EA2/diversity_progress_{timestamp}.png")
    plot_hall_of_fame_heatmap(hall_of_fame, f"EA2/hall_of_fame_{timestamp}.png")

    return best_solution


# === MAIN ===

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    students = load_students("input_data/students_03.xlsx")
    best = evolutionary_algorithm(students)

    relative_stats = compute_relative_statistics(best)
    print_relative_stats(relative_stats)
    plot_relative_statistics(relative_stats, f"EA2/relative_distribution_{datetime.now().timestamp()}.png")
    print_total_stats(students, best)
