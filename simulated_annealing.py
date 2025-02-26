import random
import math
import logging
from datetime import datetime

import matplotlib.pyplot as plt

from constants import NUM_CLASSES, INITIAL_TEMP, COOLING_RATE, MAX_ITERATIONS
from student_loader import load_students

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fitness(classes: list[list]) -> float:
    class_sizes = [len(cls) for cls in classes]  # Number of students in each class
    boys_balance = [sum(1 for s in cls if s["pohlavi"] == "K") for cls in classes]  # Number of boys in each class
    girls_balance = [sum(1 for s in cls if s["pohlavi"] == "D") for cls in classes]

    # Deviation from average class size
    size_std_dev = sum(abs(size - sum(class_sizes) / len(class_sizes)) for size in class_sizes)

    # Deviation from average gender distribution
    boys_std_dev = sum(abs(boys - sum(boys_balance) / len(boys_balance)) for boys in boys_balance)
    girls_std_dev = sum(abs(girls - sum(girls_balance) / len(girls_balance)) for girls in girls_balance)
    gender_std_dev = boys_std_dev + girls_std_dev

    total_cost = size_std_dev + gender_std_dev
    logging.info(
        f"Fitness evaluation -> Size deviation: {size_std_dev}, Gender deviation: {gender_std_dev}, Total cost: {total_cost}")

    return total_cost


def simulated_annealing(students: list[dict], num_classes: int, initial_temp, cooling_rate,
                        max_iterations) -> list[list]:
    # Randomly assign students to classes
    classes = [[] for _ in range(num_classes)]
    random.shuffle(students)

    # Assign students to classes randomly
    for student in students:
        classes[random.randint(0, num_classes - 1)].append(student)

    current_score = fitness(classes)  # Calculate initial cost
    temp = initial_temp

    logging.info(f"Starting Simulated Annealing: Initial cost = {current_score}")

    costs = []
    temperatures = []
    exps = []

    for iteration in range(max_iterations):
        new_classes = [cls[:] for cls in classes]  # copy current solution

        if random.random() < 0.5:  # 50% probability: swap or move
            # Swap two random students between two classes
            class_a, class_b = random.sample(range(num_classes), 2)
            if new_classes[class_a] and new_classes[class_b]:
                idx_a, idx_b = (random.randint(0, len(new_classes[class_a]) - 1),
                                random.randint(0, len(new_classes[class_b]) - 1))
                new_classes[class_a][idx_a], new_classes[class_b][idx_b] = new_classes[class_b][idx_b], \
                    new_classes[class_a][idx_a]

        else:
            # Move a student from one class to another
            class_from, class_to = random.sample(range(num_classes), 2)
            if new_classes[class_from]:  # Ensure the source class is not empty
                idx = random.randint(0, len(new_classes[class_from]) - 1)
                student = new_classes[class_from].pop(idx)  # Remove student from original class
                new_classes[class_to].append(student)  # Add student to new class

        new_score = fitness(new_classes)  # Calculate new cost

        # Accept new solution if it is better or with probability exp((current_score - new_score) / temp)
        exp = math.exp((current_score - new_score) / temp)
        exps.append(exp)
        if new_score < current_score or exp > random.random():
            classes, current_score = new_classes, new_score
            logging.info(f"Iteration {iteration}: Accepted new state with cost {new_score}, Temperature = {temp:.2f}")
        else:
            logging.info(f"Iteration {iteration}: Rejected new state with cost {new_score}, Temperature = {temp:.2f}")

        costs.append(current_score)
        temperatures.append(temp)
        temp *= cooling_rate

        if iteration % 100 == 0:
            logging.info(f"Iteration {iteration}: Current best cost = {current_score}, Temperature = {temp:.2f}")

    logging.info(f"Final solution found with cost {current_score}")

    visualise_all(costs, exps, temperatures, max_iterations, f"combined_plot_{datetime.now().timestamp()}.png")

    return classes


def visualise_all(costs, exps, temperatures, max_iterations, filename="combined_plot.png"):
    plt.figure(figsize=(15, 10))

    # Fitness Cost
    plt.subplot(2, 2, 1)  # (rows, columns, index)
    plt.plot(range(max_iterations), costs, label="Fitness Cost", color='blue')
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Cost")
    plt.title("Fitness Cost vs Iterations")
    plt.legend()

    # Exponential Value
    plt.subplot(2, 2, 2)
    plt.plot(range(len(exps)), exps, label="Acceptance probability", color='green')
    plt.xlabel("Iterations")
    plt.ylabel("Acceptance Probability")
    plt.title("Decrease in Acceptance Probability Over Time")
    plt.legend()

    # Temperature
    plt.subplot(2, 2, 3)
    plt.plot(range(max_iterations), temperatures, label="Temperature", color='red')
    plt.xlabel("Iterations")
    plt.ylabel("Temperature")
    plt.title("Temperature vs Iterations")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"output_data/{filename}")


if __name__ == "__main__":
    students = load_students("input_data/students_02.xlsx")
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
