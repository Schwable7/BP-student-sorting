import copy
import logging
import random
from datetime import datetime

from constants import MUT_PROB, CX_PROB, POPULATION_SIZE, NUM_CLASSES, GENERATIONS, TOURNAMENT_SIZE, ID, ELITE_COUNT, \
    HALL_OF_FAME_SIZE, STUDENTS_PATH
from fitness import fitness, fitness_simple
from helper_functions import compute_relative_statistics, print_relative_stats, print_total_stats, calculate_diversity, \
    convert_classes_to_individual
from student_loader import load_students
from visualisation import plot_relative_statistics, plot_hall_of_fame_heatmap, plot_fitness_progress_own, \
    plot_diversity_progress_own, visualize_multiplot, visualize_multiplot_simple


def generate_random_solution(students: list[dict], num_classes) -> list[list[dict]]:
    classes = [[] for _ in range(num_classes)]
    for student in students:
        random.choice(classes).append(student)
    return classes


def tournament_selection(pop_with_fitness: list[tuple[list[list[dict]], float]], k: int) -> list[list[list[dict]]]:
    selected = []
    for _ in range(POPULATION_SIZE):
        competitors = random.sample(pop_with_fitness, k)
        winner = min(competitors, key=lambda x: x[1]["total_cost"])
        selected.append(copy.deepcopy(winner[0]))
    return selected


def crossover(parent1: list[list[dict]], parent2: list[list[dict]], num_classes) -> tuple[list[list[dict]], list[list[dict]]]:
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
        classes = [[] for _ in range(num_classes)]
        for sid, choices in student_map.items():
            chosen_class = max(set(choices), key=choices.count) if choices else random.randint(0, num_classes - 1)
            student = next(s for cls in (parent1 + parent2) for s in cls if s[ID] == sid)
            classes[chosen_class].append(student)
        return classes

    return build_child(), build_child()


def mutate(individual: list[list[dict]], num_classes):
    if random.random() < 0.5:
        # Swap two students between two different classes
        class_indices = random.sample(range(num_classes), 2)
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


def balance_classes(classes: list[list[dict]], num_classes) -> list[list[dict]]:
    student_ids = set()
    unique_students = []

    for cls in classes:
        for student in cls:
            if student[ID] not in student_ids:
                student_ids.add(student[ID])
                unique_students.append(student)

    # Clear and reassign randomly, but preserve number of classes
    new_classes = [[] for _ in range(num_classes)]
    for student in unique_students:
        random.choice(new_classes).append(student)

    return new_classes


def evolutionary_algorithm(students: list[dict], dataset, mut_prob: float, cx_prob: float, tournament_size: int, elite_count: int, num_classes: int = NUM_CLASSES, generations: int = GENERATIONS):
    start_time = datetime.now()
    # Initialize population
    population = [generate_random_solution(students, num_classes) for _ in range(POPULATION_SIZE)]

    best_solution = None
    best_fitness = float('inf')

    logbook = []  # will hold dictionaries per generation
    hall_of_fame = []  # will hold best solutions (student → class ID) over time
    costs, size_devs, boys_devs, girls_devs, tgthr_penalties, not_tgthr_penalties, probabilities, temperatures = [], [], [], [], [], [], [], []

    for generation in range(generations):
        # Evaluate fitness of population
        evaluated = [(individual, fitness(individual, False)) for individual in population]
        # evaluated = [(individual, fitness_simple(individual, False)) for individual in population]
        evaluated.sort(key=lambda x: x[1]["total_cost"])
        # Select elites
        elites = [copy.deepcopy(individual) for individual, _ in evaluated[:elite_count]]

        # Track best solution
        fitness_dict = evaluated[0][1]
        gen_best_fitness = fitness_dict["total_cost"]

        size_dev = fitness_dict["size_dev"]
        boys_dev = fitness_dict["boys_dev"]
        girls_dev = fitness_dict["girls_dev"]
        together_penalty = fitness_dict["together_penalty"]
        not_together_penalty = fitness_dict["not_together_penalty"]

        gen_avg_fitness = sum(f["total_cost"] for _, f in evaluated) / len(evaluated)
        if gen_best_fitness < best_fitness:
            best_solution = evaluated[0][0]
            best_fitness = gen_best_fitness

        costs.append(gen_best_fitness)
        size_devs.append(size_dev)
        boys_devs.append(boys_dev)
        girls_devs.append(girls_dev)
        tgthr_penalties.append(together_penalty)
        not_tgthr_penalties.append(not_together_penalty)

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
            f"Generation {generation} - Best fitness: {gen_best_fitness} Average fitness: {sum(f["total_cost"] for _, f in evaluated) / len(evaluated)}")

        # Selection (Tournament)
        selected = tournament_selection(evaluated, tournament_size)

        # Create next generation
        next_generation = []

        while len(next_generation) < POPULATION_SIZE:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)

            if random.random() < cx_prob:
                child1, child2 = crossover(parent1, parent2, num_classes)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            if random.random() < mut_prob:
                mutate(child1, num_classes)
            if random.random() < mut_prob:
                mutate(child2, num_classes)

            next_generation.extend([child1, child2])

        next_generation = elites + next_generation[:POPULATION_SIZE - elite_count]
        population = next_generation

    end_time = datetime.now()
    execution_time = end_time - start_time
    logging.info(f"Optimization completed in {execution_time} seconds")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    plot_fitness_progress_own(logbook, f"EA2/fitness_progress_{timestamp}.png", dataset)
    plot_diversity_progress_own(logbook, f"EA2/diversity_progress_{timestamp}.png", dataset)
    plot_hall_of_fame_heatmap(hall_of_fame, f"EA2/hall_of_fame_{timestamp}.png", dataset)
    visualize_multiplot(costs, size_devs, boys_devs, girls_devs, tgthr_penalties,
                        not_tgthr_penalties, generations, f"EA2/multi_plot_{timestamp}.png", dataset)

    # visualize_multiplot_simple(
    #     costs, size_devs, boys_devs, girls_devs, generations,
    #     f"EA2/multi_plot_simple_{datetime.now().timestamp()}.png", dataset
    # )

    relative_stats = compute_relative_statistics(students, best_solution)
    print_relative_stats(relative_stats)
    plot_relative_statistics(relative_stats, f"EA2/relative_distribution_{datetime.now().timestamp()}.png", dataset)
    print_total_stats(students, best_solution)
    return best_solution, costs, logbook, execution_time, relative_stats


# === MAIN ===

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    students = load_students(STUDENTS_PATH)
    for i in range(1):
        best, costs, logbook, execution_time, relative_stats = evolutionary_algorithm(students, "basic")


