import random
import math
import logging
from datetime import datetime

from constants import NUM_CLASSES, INITIAL_TEMP, COOLING_RATE, MAX_ITERATIONS
from fitness import fitness
from student_loader import load_students
from visualisation import visualize_sa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def simulated_annealing(students: list[dict], num_classes: int, initial_temp, cooling_rate,
                        max_iterations) -> list[list]:
    # Randomly assign students to classes
    classes = [[] for _ in range(num_classes)]
    random.shuffle(students)

    # Assign students to classes randomly
    for student in students:
        classes[random.randint(0, num_classes - 1)].append(student)

    current_score, size_dev, boys_dev, girls_dev = fitness(classes)  # Calculate initial cost
    temp = initial_temp

    logging.info(f"Starting Simulated Annealing: Initial cost = {current_score}")

    costs = []
    size_devs = []
    boys_devs = []
    girls_devs = []
    temperatures = []
    probabilities = []

    for iteration in range(max_iterations):
        new_classes = [cls[:] for cls in classes]  # copy current solution

        if random.random() < 0.5:  # 50% probability: swap or move
            # Swap two random students between two classes
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

        else:
            # Move a student from one class to another
            class_from, class_to = random.sample(range(num_classes), 2)
            if new_classes[class_from]:  # Ensure the source class is not empty
                idx = random.randint(0, len(new_classes[class_from]) - 1)
                student = new_classes[class_from].pop(idx)  # Remove student from original class
                new_classes[class_to].append(student)  # Add student to new class

        new_score, new_size_dev, new_boys_dev, new_girls_dev = fitness(new_classes)  # Calculate new cost

        # Accept new solution if it is better or with probability exp((current_score - new_score) / temp)
        acceptance_probability = math.exp((current_score - new_score) / temp)
        probabilities.append(acceptance_probability)
        if new_score < current_score or acceptance_probability > random.random():
            classes, current_score, size_dev, boys_dev, girls_dev = new_classes, new_score, new_size_dev, new_boys_dev, new_girls_dev
            logging.info(f"Iteration {iteration}: Accepted new state with cost {new_score}, Temperature = {temp:.2f}")
        else:
            logging.info(f"Iteration {iteration}: Rejected new state with cost {new_score}, Temperature = {temp:.2f}")

        costs.append(current_score)
        size_devs.append(size_dev)
        boys_devs.append(boys_dev)
        girls_devs.append(girls_dev)
        temperatures.append(temp)

        temp *= cooling_rate

        if iteration % 100 == 0:
            logging.info(f"Iteration {iteration}: Current best cost = {current_score}, Temperature = {temp:.2f}")

    logging.info(f"Final solution found with cost {current_score}")

    visualize_sa(
        costs, size_devs, boys_devs, girls_devs, probabilities, temperatures, max_iterations,
        f"SA_plot_{datetime.now().timestamp()}.png"
    )
    return classes


if __name__ == "__main__":
    students = load_students("input_data/students_02.xlsx")
    for i in range(5):
        sorted_classes = simulated_annealing(
            students=students,
            num_classes=NUM_CLASSES,
            initial_temp=INITIAL_TEMP,
            cooling_rate=COOLING_RATE,
            max_iterations=MAX_ITERATIONS
        )
        print(f"Total number of students: {len(students)}")
        print(f"Num of boys = {sum(1 for s in students if s['pohlavi'] == 'K')}, "
              f"Num of girls = {sum(1 for s in students if s['pohlavi'] == 'D')}")
        for i, cls in enumerate(sorted_classes):
            print(f"Class {i + 1}: {len(cls)} students, Boys = {sum(1 for s in cls if s['pohlavi'] == 'K')}, "
                  f"Girls = {sum(1 for s in cls if s['pohlavi'] == 'D')}")
