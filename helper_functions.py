import csv
import logging
import random
import statistics
from datetime import datetime

import pandas as pd

from constants import GENDER, MALE, FEMALE, ID, DEFERRAL, LEARNING_DISABILITIES, TALENT, DIFF_MOTHER_LANG


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
            students.append((student[ID], class_idx))

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


def compute_relative_statistics(students: list[dict], classes: list[list[dict]]) -> dict:
    total_boys = sum(1 for s in students if s[GENDER] == MALE)
    total_girls = sum(1 for s in students if s[GENDER] == FEMALE)
    total_deferred = sum(1 for s in students if s[DEFERRAL] == 1)
    total_disabilities = sum(1 for s in students if s[LEARNING_DISABILITIES] == 1)
    total_talented = sum(1 for s in students if s[TALENT] == 1)
    total_diff_lang = sum(1 for s in students if s[DIFF_MOTHER_LANG] == 1)

    stats = {
        "class": [],
        "total_ratio": [],
        "boys_ratio": [],
        "girls_ratio": [],
        "deferred_ratio": [],
        "disabilities_ratio": [],
        "talented_ratio": [],
        "diff_lang_ratio": []
    }

    for i, cls in enumerate(classes):
        boys = sum(1 for s in cls if s[GENDER] == MALE)
        girls = sum(1 for s in cls if s[GENDER] == FEMALE)
        deferred = sum(1 for s in cls if s[DEFERRAL] == 1)
        disabilities = sum(1 for s in cls if s[LEARNING_DISABILITIES] == 1)
        talented = sum(1 for s in cls if s[TALENT] == 1)
        diff_lang = sum(1 for s in cls if s[DIFF_MOTHER_LANG] == 1)

        boys_pct = (boys / total_boys) if total_boys > 0 else 0
        girls_pct = (girls / total_girls) if total_girls > 0 else 0
        deferred_pct = (deferred / total_deferred) if total_deferred > 0 else 0
        disabilities_pct = (disabilities / total_disabilities) if total_disabilities > 0 else 0
        talented_pct = (talented / total_talented) if total_talented > 0 else 0
        diff_lang_pct = (diff_lang / total_diff_lang) if total_diff_lang > 0 else 0
        total_ratio = len(cls) / len(students) if len(students) > 0 else 0
        stats["total_ratio"].append(total_ratio)
        stats["class"].append(i + 1)
        stats["boys_ratio"].append(boys_pct)
        stats["girls_ratio"].append(girls_pct)
        stats["deferred_ratio"].append(deferred_pct)
        stats["disabilities_ratio"].append(disabilities_pct)
        stats["talented_ratio"].append(talented_pct)
        stats["diff_lang_ratio"].append(diff_lang_pct)

    return stats


def print_total_stats_simple(students: list[dict], classes: list[list[dict]]):
    print(f"Total number of students: {len(students)}")
    print(f"Num of boys = {sum(1 for s in students if s[GENDER] == MALE)}, "
          f"Num of girls = {sum(1 for s in students if s[GENDER] == FEMALE)}")
    for i, cls in enumerate(classes):
        print(f"Class {i + 1}: {len(cls)} students, Boys = {sum(1 for s in cls if s[GENDER] == MALE)}, "
              f"Girls = {sum(1 for s in cls if s[GENDER] == FEMALE)}")


def print_total_stats(students: list[dict], classes: list[list[dict]]):
    total_boys = sum(1 for s in students if s[GENDER] == MALE)
    total_girls = sum(1 for s in students if s[GENDER] == FEMALE)
    total_deferred = sum(1 for s in students if s[DEFERRAL] == 1)
    total_disabilities = sum(1 for s in students if s[LEARNING_DISABILITIES] == 1)
    total_talented = sum(1 for s in students if s[TALENT] == 1)
    total_diff_lang = sum(1 for s in students if s[DIFF_MOTHER_LANG] == 1)

    print(f"Total number of students: {len(students)}")
    print(f"Num of boys = {total_boys}, Num of girls = {total_girls}")
    print(f"Num of deferred = {total_deferred}")
    print(f"Num of students with learning disabilities = {total_disabilities}")
    print(f"Num of talented students = {total_talented}")
    print(f"Num of students with different mother language = {total_diff_lang}\n")

    print("Class statistics with relative distributions (% of total category in class):")

    for i, cls in enumerate(classes):
        boys = sum(1 for s in cls if s[GENDER] == MALE)
        girls = sum(1 for s in cls if s[GENDER] == FEMALE)
        deferred = sum(1 for s in cls if s[DEFERRAL] == 1)
        disabilities = sum(1 for s in cls if s[LEARNING_DISABILITIES] == 1)
        talented = sum(1 for s in cls if s[TALENT] == 1)
        diff_lang = sum(1 for s in cls if s[DIFF_MOTHER_LANG] == 1)

        boys_pct = (boys / total_boys * 100) if total_boys > 0 else 0
        girls_pct = (girls / total_girls * 100) if total_girls > 0 else 0
        deferred_pct = (deferred / total_deferred * 100) if total_deferred > 0 else 0
        disabilities_pct = (disabilities / total_disabilities * 100) if total_disabilities > 0 else 0
        talented_pct = (talented / total_talented * 100) if total_talented > 0 else 0
        diff_lang_pct = (diff_lang / total_diff_lang * 100) if total_diff_lang > 0 else 0

        print(f"Class {i + 1}: {len(cls)} students")
        print(f"  Boys: {boys} ({boys_pct:.2f}%)")
        print(f"  Girls: {girls} ({girls_pct:.2f}%)")
        print(f"  Deferred: {deferred} ({deferred_pct:.2f}%)")
        print(f"  Learning Disabilities: {disabilities} ({disabilities_pct:.2f}%)")
        print(f"  Talented: {talented} ({talented_pct:.2f}%)")
        print(f"  Different Mother Language: {diff_lang} ({diff_lang_pct:.2f}%)\n")


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
                data.append([class_idx + 1, student[ID], student[GENDER]])

        df = pd.DataFrame(data, columns=["Class", "Student ID", "Gender"])

        # Display using Pandas formatting
        print(df.to_string(index=False))  # Print without row index for clarity


def calculate_diversity(population: list[list[list[dict]]]) -> float:
    signatures = set()
    for individual in population:
        sig = tuple(sorted(frozenset(s[ID] for s in cls) for cls in individual))
        signatures.add(sig)
    return len(signatures) / len(population)


def convert_to_class_vector(individual: list[list[dict]]) -> list[int]:
    class_vector = {}
    for class_idx, cls in enumerate(individual):
        for student in cls:
            class_vector[student[ID]] = class_idx
    # Return as a list ordered by student ID (sorted)
    return [class_vector[sid] for sid in sorted(class_vector)]


def export_fitness_summary_to_csv(
    all_costs: dict[str, list[list[float]]],
    all_exec_times: dict[str, list[float]],
    filename: str,
    output_dir: str = "output_data",
) -> None:
    """
    Write one CSV row per algorithm with:
      mean_fitness ± std,  min_fitness,  max_fitness,  mean_exec_time
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_filename = f"{output_dir}/{filename}"

    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Algoritmus",
                "Průměrná fitness ± Směr. Odchylka",
                "Nejlepší Fitness",
                "Nejhorší Fitness",
                "Průměrný čas (s)",
                "Počet iterací"
            ]
        )

        for algo, runs_costs in all_costs.items():
            # best fitness in every run
            best_per_run = [min(costs) for costs in runs_costs]
            iterations = len(runs_costs[0]) if runs_costs else 0

            mean_fit = statistics.mean(best_per_run)
            std_fit = statistics.stdev(best_per_run) if len(best_per_run) > 1 else 0.0
            min_fit = min(best_per_run)
            max_fit = max(best_per_run)

            mean_time = statistics.mean(all_exec_times[algo])

            writer.writerow(
                [
                    algo,
                    f"{mean_fit:.5f} ± {std_fit:.5f}",
                    f"{min_fit:.5f}",
                    f"{max_fit:.5f}",
                    f"{mean_time:.2f}",
                    f"{iterations}"
                ]
            )

    logging.info("Fitness summary saved to %s", csv_filename)


def save_all_statistics(
    all_statistics: dict[str, list[dict]],
    filename: str,
    output_format: str = "csv"   # or "xlsx"
):
    def class_row(algo: str, run_idx: int, class_idx: int, cls: list[dict], totals: dict) -> dict:
        """Return one row for the output DataFrame for a single class."""
        boys   = sum(1 for s in cls if s[GENDER] == MALE)
        girls  = sum(1 for s in cls if s[GENDER] == FEMALE)
        defer  = sum(1 for s in cls if s[DEFERRAL] == 1)
        disab  = sum(1 for s in cls if s[LEARNING_DISABILITIES] == 1)
        talent = sum(1 for s in cls if s[TALENT] == 1)
        dlang  = sum(1 for s in cls if s[DIFF_MOTHER_LANG] == 1)

        pct = lambda cnt, tot: round(cnt / tot * 100, 2) if tot else 0.0

        return {
            "Algorithm"            : algo,
            "Run"                  : run_idx + 1,
            "Class"                : class_idx + 1,
            "Size"                 : len(cls),

            "Boys"                 : boys,
            "Boys %"               : pct(boys, totals["boys"]),
            "Girls"                : girls,
            "Girls %"              : pct(girls, totals["girls"]),
            "Deferred"             : defer,
            "Deferred %"           : pct(defer, totals["deferred"]),
            "Learning Disabilities": disab,
            "Disabilities %"       : pct(disab, totals["disabilities"]),
            "Talented"             : talent,
            "Talented %"           : pct(talent, totals["talented"]),
            "Different Mother Language": dlang,
            "Diff Lang %"          : pct(dlang, totals["diff_lang"]),
        }

    rows: list[dict] = []

    for algo, runs in all_statistics.items():
        for run_idx, run_data in enumerate(runs):
            students = run_data["students"]
            classes = run_data["classes"]

            totals = {
                "boys":        sum(1 for s in students if s[GENDER] == MALE),
                "girls":       sum(1 for s in students if s[GENDER] == FEMALE),
                "deferred":    sum(1 for s in students if s[DEFERRAL] == 1),
                "disabilities":sum(1 for s in students if s[LEARNING_DISABILITIES] == 1),
                "talented":    sum(1 for s in students if s[TALENT] == 1),
                "diff_lang":   sum(1 for s in students if s[DIFF_MOTHER_LANG] == 1),
            }

            for class_idx, cls in enumerate(classes):
                rows.append(class_row(algo, run_idx, class_idx, cls, totals))

    df = pd.DataFrame(rows)

    if output_format.lower() == "xlsx":
        df.to_excel(f"output_data/{filename}.xlsx", index=False)
    else:
        df.to_csv(f"output_data/{filename}.csv", index=False)

    print(f"✅ Saved {len(df)} rows to {filename}.{output_format.lower()}")