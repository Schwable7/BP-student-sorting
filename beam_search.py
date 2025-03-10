import logging
import random
from datetime import datetime

from constants import NUM_CLASSES, MAX_ITERATIONS, BEAM_WIDTH
from fitness import fitness
from student_loader import load_students
from visualisation import visualize_bs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_neighbors(classes: list[list], num_neighbors: int) -> list[list[list]]:
    neighbors = []

    for _ in range(num_neighbors):
        num_classes = len(classes)

        new_classes = [cls[:] for cls in classes]  # Deep copy

        if random.random() < 0.5:  # Swap students
            class_a, class_b = random.sample(range(num_classes), 2)
            if new_classes[class_a] and new_classes[class_b]:
                idx_a, idx_b = (
                    random.randint(0, len(new_classes[class_a]) - 1),
                    random.randint(0, len(new_classes[class_b]) - 1),
                )
                new_classes[class_a][idx_a], new_classes[class_b][idx_b] = (
                    new_classes[class_b][idx_b],
                    new_classes[class_a][idx_a],
                )
        else:  # Move a student
            class_from, class_to = random.sample(range(num_classes), 2)
            if new_classes[class_from]:
                idx = random.randint(0, len(new_classes[class_from]) - 1)
                student = new_classes[class_from].pop(idx)
                new_classes[class_to].append(student)

        neighbors.append(new_classes)

    return neighbors


def beam_search(students: list[dict], num_classes: int, beam_width: int, max_iterations: int) -> list[list]:
    classes = [[] for _ in range(num_classes)]
    random.shuffle(students)

    for student in students:
        classes[random.randint(0, num_classes - 1)].append(student)

    initial_cost, size_dev, boys_dev, girls_dev = fitness(classes)
    beam = [(classes, initial_cost, size_dev, boys_dev, girls_dev)]

    logging.info(f"Starting Beam Search: Initial cost = {initial_cost}")

    costs, size_devs, boys_devs, girls_devs = [], [], [], []

    best_cost = initial_cost  # Track the best cost found
    best_iteration_found = 0  # Track the iteration number when the best cost is found

    for iteration in range(max_iterations):
        candidates = []
        for state, _, _, _, _ in beam:
            neighbors = generate_neighbors(state, num_neighbors=beam_width * 2)  # Generate twice as many candidates
            for neighbor in neighbors:
                cost, size_dev, boys_dev, girls_dev = fitness(neighbor)
                candidates.append((neighbor, cost, size_dev, boys_dev, girls_dev))

        # Select the best k solutions
        beam = sorted(candidates, key=lambda x: x[1])[:beam_width]

        # Track best solution
        best_classes, current_best_cost, best_size_dev, best_boys_dev, best_girls_dev = beam[0]

        costs.append(current_best_cost)
        size_devs.append(best_size_dev)
        boys_devs.append(best_boys_dev)
        girls_devs.append(best_girls_dev)

        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_iteration_found = iteration  # Store the iteration number of improvement

        if iteration % 100 == 0:
            logging.info(f"Iteration {iteration}: Best cost = {current_best_cost}")

    logging.info(f"Final solution found with cost {best_cost} at iteration {best_iteration_found}")

    visualize_bs(costs, size_devs, boys_devs, girls_devs, max_iterations, f"BS_plot_{datetime.now().timestamp()}.png")

    return best_classes


if __name__ == "__main__":
    students = load_students("input_data/students_02.xlsx")

    for i in range(5):
        sorted_classes = beam_search(
            students=students,
            num_classes=NUM_CLASSES,
            beam_width=BEAM_WIDTH,
            max_iterations=100,
        )

        print(f"Total number of students: {len(students)}")
        print(f"Num of boys = {sum(1 for s in students if s['pohlavi'] == 'K')}, "
              f"Num of girls = {sum(1 for s in students if s['pohlavi'] == 'D')}")

        for i, cls in enumerate(sorted_classes):
            print(f"Class {i + 1}: {len(cls)} students, Boys = {sum(1 for s in cls if s['pohlavi'] == 'K')}, "
                  f"Girls = {sum(1 for s in cls if s['pohlavi'] == 'D')}")
