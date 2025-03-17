import logging


def fitness(classes: list[list], print_progress: bool = True) -> tuple[float, float, float, float]:
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
    if print_progress:
        logging.info(
            f"Fitness evaluation -> Size deviation: {size_std_dev}, Boys deviation: {boys_std_dev}, "
            f"Girls deviation: {girls_std_dev}, Total cost: {total_cost}"
        )

    return total_cost, size_std_dev, boys_std_dev, girls_std_dev
