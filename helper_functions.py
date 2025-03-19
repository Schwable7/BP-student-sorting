import random

import pandas as pd


def convert_classes_to_individual(classes: list[list[dict]]) -> list[int]:
    """
    Converts a nested list of class assignments into a sorted list where each element represents
    a student's assigned class. The list is sorted by student ID.

    :param classes: List of classes, where each class contains a list of students (dicts with 'id' key).
    :return: A sorted list where each entry represents a student's class assignment.
    """
    students = []  # List to store (student_id, assigned_class) tuples

    # Collect all students and their assigned class
    for class_idx, student_list in enumerate(classes):
        for student in student_list:
            students.append((student["student_uid"], class_idx))

    # Sort students by their ID
    students.sort(key=lambda x: x[0])

    # Extract only the class assignments in sorted order
    return [class_idx for _, class_idx in students]


def convert_individual_to_classes(individual: list[int], students: list[dict], num_classes: int) -> list[list]:
    classes = [[] for _ in range(num_classes)]
    for student_idx, class_idx in enumerate(individual):
        classes[class_idx].append(students[student_idx])
    return classes


def swap_or_move(classes, num_classes):
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
    return new_classes


def compute_population_diversity(population):
    """ Measures diversity by counting unique individuals. """
    unique_solutions = len(set(tuple(ind) for ind in population))
    return unique_solutions / len(population)


def compute_population_diversity_2(population):
    """ Measures diversity by counting unique individuals. """
    unique_solutions = len(set(tuple(convert_classes_to_individual(ind)) for ind in population))
    return unique_solutions / len(population)


def compute_relative_statistics(classes: list[list[dict]]) -> dict:
    stats = {
        "class": [],
        "boys_ratio": [],
        "girls_ratio": [],
    }

    for class_idx, cls in enumerate(classes):
        total_students = len(cls)
        if total_students == 0:
            boys_ratio = 0
            girls_ratio = 0
        else:
            num_boys = sum(1 for s in cls if s["pohlavi"] == "K")
            num_girls = sum(1 for s in cls if s["pohlavi"] == "D")

            boys_ratio = num_boys / total_students
            girls_ratio = num_girls / total_students

        stats["class"].append(class_idx + 1)  # Class index starting from 1
        stats["boys_ratio"].append(boys_ratio)
        stats["girls_ratio"].append(girls_ratio)

    return stats


def print_total_stats(students: list[dict], classes: list[list[dict]]):
    print(f"Total number of students: {len(students)}")
    print(f"Num of boys = {sum(1 for s in students if s['pohlavi'] == 'K')}, "
          f"Num of girls = {sum(1 for s in students if s['pohlavi'] == 'D')}")
    for i, cls in enumerate(classes):
        print(f"Class {i + 1}: {len(cls)} students, Boys = {sum(1 for s in cls if s['pohlavi'] == 'K')}, "
              f"Girls = {sum(1 for s in cls if s['pohlavi'] == 'D')}")


def print_relative_stats(relative_stats: dict):
    print("Relative Boys/Girls Distribution:")
    for idx, (boys_ratio, girls_ratio) in enumerate(
            zip(relative_stats["boys_ratio"], relative_stats["girls_ratio"])):
        print(f"Class {idx + 1}: Boys {boys_ratio:.2%}, Girls {girls_ratio:.2%}")


def print_hall_of_fame(hall_of_fame):
    for i, (classes, score) in enumerate(hall_of_fame):
        print(f"\n--- Hall of Fame Solution {i + 1} --- (Fitness Score: {score})")

        # Create a DataFrame to show class assignments
        data = []
        for class_idx, cls in enumerate(classes):
            for student in cls:
                data.append([class_idx + 1, student["student_uid"], student["pohlavi"]])

        df = pd.DataFrame(data, columns=["Class", "Student ID", "Gender"])

        # Display using Pandas formatting
        print(df.to_string(index=False))  # Print without row index for clarity
