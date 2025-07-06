import csv
import logging
from datetime import datetime

from beam_search import beam_search
from constants import NUM_CLASSES, INITIAL_TEMP, COOLING_RATE, MAX_ITERATIONS, BEAM_WIDTH, STUDENTS_PATH, BEAM_SEARCH, \
    SIMULATED_ANNEALING, EA_DEAP, EA_OWN, BEAM_ITERATIONS, CX_PROB, MUT_PROB, TOURNAMENT_SIZE, ELITE_COUNT
from evolution import evolution
from evolution_own import evolutionary_algorithm
from helper_functions import export_fitness_summary_to_csv, save_all_statistics
from simulated_annealing import simulated_annealing
from student_loader import load_students
from visualisation import compare_algorithms_graph, compare_diversity_progress, \
    compare_algorithms_separate_graph, compare_algorithms_sd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_all(students_path: str, dataset: str):
    students = load_students("input_data/" + students_path)
    classes_sa, costs_sa, time_sa, stats_sa = simulated_annealing(
        students=students,
        num_classes=NUM_CLASSES,
        initial_temp=INITIAL_TEMP,
        cooling_rate=COOLING_RATE,
        max_iterations=MAX_ITERATIONS,
        dataset=dataset,
    )

    classes_bs, costs_bs, time_bs, stats_bs = beam_search(
        students=students,
        num_classes=NUM_CLASSES,
        beam_width=BEAM_WIDTH,
        max_iterations=BEAM_ITERATIONS,
        dataset=dataset,
    )

    classes_ea, logbook_deap, time_ea_deap, stats_ea_deap = evolution(
        students=students,
        dataset=dataset,
        mut_prob=MUT_PROB,
        cx_prob=CX_PROB,
        tournament_size=TOURNAMENT_SIZE,
        num_classes=NUM_CLASSES,
        generations=GENERATIONS
    )

    classes_ea_own, costs_ea, logbook_own, time_ea_own, stats_ea_own = evolutionary_algorithm(
        students=students,
        dataset=dataset,
        mut_prob=MUT_PROB,
        cx_prob=CX_PROB,
        tournament_size=TOURNAMENT_SIZE,
        elite_count=ELITE_COUNT,
        num_classes=NUM_CLASSES,
        generations=GENERATIONS
    )

    return {
        "fitness": {
            BEAM_SEARCH: costs_bs,
            SIMULATED_ANNEALING: costs_sa,
            EA_DEAP: logbook_deap.select("min"),
            EA_OWN: costs_ea
        },
        "diversity": {
            EA_DEAP: logbook_deap.select("diversity"),
            EA_OWN: [log["diversity"] for log in logbook_own]
        },
        "execution_time": {
            BEAM_SEARCH: time_bs,
            SIMULATED_ANNEALING: time_sa,
            EA_DEAP: time_ea_deap,
            EA_OWN: time_ea_own
        },
        "statistics": {
            BEAM_SEARCH: {"students": students, "classes": classes_bs},
            SIMULATED_ANNEALING: {"students": students, "classes": classes_sa},
            EA_DEAP: {"students": students, "classes": classes_ea},
            EA_OWN: {"students": students, "classes": classes_ea_own}
        },
    }


if __name__ == "__main__":
    datasets = [
        "basic",
        "skewed",
        "large",
    ]

    students_basic = [
        "students_basic_0.xlsx",
        "students_basic_1.xlsx",
        "students_basic_2.xlsx",
        "students_basic_3.xlsx",
        "students_basic_4.xlsx",
        "students_basic_5.xlsx",
        "students_basic_6.xlsx",
        "students_basic_7.xlsx",
        "students_basic_8.xlsx",
        "students_basic_9.xlsx"
    ]

    students_skewed = [
        "students_skewed_0.xlsx",
        "students_skewed_1.xlsx",
        "students_skewed_2.xlsx",
        "students_skewed_3.xlsx",
        "students_skewed_4.xlsx",
        "students_skewed_5.xlsx",
        "students_skewed_6.xlsx",
        "students_skewed_7.xlsx",
        "students_skewed_8.xlsx",
        "students_skewed_9.xlsx"
    ]

    students_large = [
        "students_large_0.xlsx",
        "students_large_1.xlsx",
        "students_large_2.xlsx",
        "students_large_3.xlsx",
        "students_large_4.xlsx",
        "students_large_5.xlsx",
        "students_large_6.xlsx",
        "students_large_7.xlsx",
        "students_large_8.xlsx",
        "students_large_9.xlsx"
    ]

    iterations_list = [
        100,
        500,
        1000
    ]
    runs = 1
    for i in range(len(iterations_list)):
        iterations = iterations_list[i]
        logging.info(f"Running experiments for iterations: {iterations}")
        BEAM_ITERATIONS = iterations
        MAX_ITERATIONS = iterations
        GENERATIONS = iterations
        for dataset in datasets:
            file_path = f"{iterations}/{dataset}"
            if dataset == "basic":
                students = students_basic
                NUM_CLASSES = 4
            elif dataset == "skewed":
                students = students_skewed
                NUM_CLASSES = 4
            elif dataset == "large":
                students = students_large
                NUM_CLASSES = 8
            else:
                logging.error(f"Unknown dataset: {dataset}")
                continue

            logging.info(f"Running experiments for dataset: {dataset}")
            all_costs = {
                BEAM_SEARCH: [],
                SIMULATED_ANNEALING: [],
                EA_DEAP: [],
                EA_OWN: []
            }
            all_diversity = {
                EA_DEAP: [],
                EA_OWN: []
            }
            all_statistics = {
                BEAM_SEARCH: [],
                SIMULATED_ANNEALING: [],
                EA_DEAP: [],
                EA_OWN: [],
                "students": []
            }
            all_exec_times = {algo: [] for algo in all_costs}

            # Run all algorithms for each student file
            for i, student_file in enumerate(students):
                for j in range(runs):
                    logging.info(f"Running iteration {j + 1} for student file {student_file} (run {i + 1})")
                    results = run_all(student_file, file_path)
                    for key in all_costs:
                        all_costs[key].append(results["fitness"][key])
                        all_exec_times[key].append(results["execution_time"][key].total_seconds())
                        all_statistics[key].append(results["statistics"][key])

                    for key in all_diversity:
                        all_diversity[key].append(results["diversity"][key])

            timestamp = datetime.now().timestamp()
            compare_algorithms_graph(all_costs, f"compare_algorithms_{timestamp}.png", file_path)
            compare_algorithms_separate_graph(all_costs, f"separate_compare_algorithms_{timestamp}.png", file_path)
            compare_algorithms_sd(all_costs, f"SD_compare_algorithms{timestamp}.png", file_path)
            compare_diversity_progress(all_diversity, f"compare_diversity_{timestamp}.png", file_path)

            export_fitness_summary_to_csv(all_costs, all_exec_times, f"{file_path}/fitness_summary_{timestamp}.csv")
            save_all_statistics(all_statistics, f"{file_path}/statistics_summary_{timestamp}")
