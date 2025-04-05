from datetime import datetime

from beam_search import beam_search
from constants import NUM_CLASSES, INITIAL_TEMP, COOLING_RATE, MAX_ITERATIONS, BEAM_WIDTH
from evolution import evolution
from simulated_annealing import simulated_annealing
from student_loader import load_students
from visualisation import visualize_all


def run_and_plot_all():
    students = load_students("input_data/students_03.xlsx")
    classes_sa, costs_sa = simulated_annealing(
        students=students,
        num_classes=NUM_CLASSES,
        initial_temp=INITIAL_TEMP,
        cooling_rate=COOLING_RATE,
        max_iterations=MAX_ITERATIONS
    )

    classes_bs, costs_bs = beam_search(
        students=students,
        num_classes=NUM_CLASSES,
        beam_width=BEAM_WIDTH,
        max_iterations=100,
    )

    classes_ea, logbook = evolution(students)

    visualize_all(
        costs_sa=costs_sa,
        costs_bs=costs_bs,
        logbook=logbook,
        max_iterations_sa=MAX_ITERATIONS,
        max_iterations_bs=100,
        filename=f"all_{datetime.now().timestamp()}.png"
    )


if __name__ == "__main__":
    run_and_plot_all()
