import logging

from constants import GENDER, MALE, FEMALE, DEFERRAL, LEARNING_DISABILITIES, TALENT, DIFF_MOTHER_LANG


def fitness(classes: list[list], print_progress: bool = True) -> dict[str, float]:
    class_sizes = [len(cls) for cls in classes]  # Number of students in each class
    boys_balance = [sum(1 for s in cls if s[GENDER] == MALE) for cls in classes]  # Number of boys in each class
    girls_balance = [sum(1 for s in cls if s[GENDER] == FEMALE) for cls in classes]

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
    fitness_dict = {
        "total_cost": total_cost,
        "size_dev": size_std_dev,
        "boys_dev": boys_std_dev,
        "girls_dev": girls_std_dev
    }
    return fitness_dict

def fitness2(classes: list[list], print_progress: bool = True) -> dict[str, float]:
    class_sizes = [len(cls) for cls in classes]  # Number of students in each class
    boys_balance = [sum(1 for s in cls if s[GENDER] == MALE) for cls in classes]
    girls_balance = [sum(1 for s in cls if s[GENDER] == FEMALE) for cls in classes]
    deferred_balance = [sum(1 for s in cls if s[DEFERRAL] == 1) for cls in classes]
    disabilities_balance = [sum(1 for s in cls if s[LEARNING_DISABILITIES] == 1) for cls in classes]
    talent_balance = [sum(1 for s in cls if s[TALENT] == 1) for cls in classes]
    diff_lang_balance = [sum(1 for s in cls if s[DIFF_MOTHER_LANG] == 1) for cls in classes]

    # Deviation from average class size
    size_std_dev = sum(abs(size - sum(class_sizes) / len(class_sizes)) for size in class_sizes)

    # Deviation from average gender distribution
    boys_std_dev = sum(abs(boys - sum(boys_balance) / len(boys_balance)) for boys in boys_balance)
    girls_std_dev = sum(abs(girls - sum(girls_balance) / len(girls_balance)) for girls in girls_balance)
    gender_std_dev = boys_std_dev + girls_std_dev

    # Deviation from average deferred student distribution
    deferred_std_dev = sum(
        abs(deferred - sum(deferred_balance) / len(deferred_balance)) for deferred in deferred_balance)

    # Deviation from average learning disabilities distribution
    disabilities_std_dev = sum(
        abs(disabilities - sum(disabilities_balance) / len(disabilities_balance)) for disabilities in
        disabilities_balance)

    # Deviation from average talent distribution
    talent_std_dev = sum(abs(talent - sum(talent_balance) / len(talent_balance)) for talent in talent_balance)

    # Deviation from average different mother language distribution
    diff_lang_std_dev = sum(
        abs(diff_lang - sum(diff_lang_balance) / len(diff_lang_balance)) for diff_lang in diff_lang_balance)

    # Total cost calculation
    total_cost = (size_std_dev + gender_std_dev + deferred_std_dev +
                  disabilities_std_dev + talent_std_dev + diff_lang_std_dev)

    if print_progress:
        logging.info(
            f"Fitness evaluation -> Size deviation: {size_std_dev}, Boys deviation: {boys_std_dev}, "
            f"Girls deviation: {girls_std_dev}, Deferred deviation: {deferred_std_dev}, "
            f"Disabilities deviation: {disabilities_std_dev}, Talent deviation: {talent_std_dev}, "
            f"Different language deviation: {diff_lang_std_dev}, Total cost: {total_cost}"
        )
    fitness_dict = {
        "total_cost": total_cost,
        "size_dev": size_std_dev,
        "boys_dev": boys_std_dev,
        "girls_dev": girls_std_dev,
        "deferred_dev": deferred_std_dev,
        "disabilities_dev": disabilities_std_dev,
        "talent_dev": talent_std_dev,
        "diff_lang_dev": diff_lang_std_dev
    }
    return fitness_dict
