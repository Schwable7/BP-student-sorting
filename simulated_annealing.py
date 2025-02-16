import random
import math
import logging

from constants import NUM_CLASSES, INITIAL_TEMP, COOLING_RATE, MAX_ITERATIONS
from student_loader import load_students

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fitness(classes: list[list]) -> float:
    class_sizes = [len(cls) for cls in classes]
    gender_balance = [sum(1 for s in cls if s["pohlavi"] == "K") for cls in classes]

    size_std_dev = sum(abs(size - sum(class_sizes) / len(class_sizes)) for size in class_sizes)
    gender_std_dev = sum(abs(gender - sum(gender_balance) / len(gender_balance)) for gender in gender_balance)

    total_cost = size_std_dev + gender_std_dev
    logging.info(
        f"Fitness evaluation -> Size deviation: {size_std_dev}, Gender deviation: {gender_std_dev}, Total cost: {total_cost}")

    return total_cost


def simulated_annealing(students: list[dict], num_classes: int, initial_temp, cooling_rate,
                        max_iterations) -> list[list]:
    # Randomly assign students to classes
    classes = [[] for _ in range(num_classes)]
    random.shuffle(students)

    # Assign students to classes in a round-robin fashion
    for i, student in enumerate(students):
        classes[i % num_classes].append(student)

    current_score = fitness(classes)  # Calculate initial cost
    temp = initial_temp

    logging.info(f"Starting Simulated Annealing: Initial cost = {current_score}")

    for iteration in range(max_iterations):
        new_classes = [cls[:] for cls in classes]  # copy current solution

        # Swap two random students from two random classes
        class_a, class_b = random.sample(range(num_classes), 2)  # select two random classes
        if new_classes[class_a] and new_classes[class_b]:  # ensure classes are not empty
            idx_a, idx_b = random.randint(0, len(new_classes[class_a]) - 1), random.randint(0, len(
                new_classes[class_b]) - 1)  # select two random students
            new_classes[class_a][idx_a], new_classes[class_b][idx_b] = new_classes[class_b][idx_b], \
                new_classes[class_a][idx_a]  # swap students

        new_score = fitness(new_classes)  # Calculate new cost

        # Accept new solution if it is better or with probability exp((current_score - new_score) / temp)
        if new_score < current_score or math.exp((current_score - new_score) / temp) > random.random():
            classes, current_score = new_classes, new_score
            logging.info(f"Iteration {iteration}: Accepted new state with cost {new_score}, Temperature = {temp:.2f}")
        else:
            logging.info(f"Iteration {iteration}: Rejected new state with cost {new_score}, Temperature = {temp:.2f}")

        temp *= cooling_rate  # Cool down temperature

        if iteration % 100 == 0:
            logging.info(f"Iteration {iteration}: Current best cost = {current_score}, Temperature = {temp:.2f}")

    logging.info(f"Final solution found with cost {current_score}")
    return classes


if __name__ == "__main__":
    students = load_students("students.xlsx")
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
