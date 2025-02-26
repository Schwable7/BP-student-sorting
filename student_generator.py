import numpy as np
import pandas as pd
import random

from datetime import datetime, timedelta
from constants import STUDENTS_COUNT


def generate_student_age():
    # Assuming students are between 6 and 7 years old (1st year primary school)
    start_date = datetime(datetime.today().year - 7, 1, 1)  # Oldest student born at the start of the year
    end_date = datetime(datetime.today().year - 6, 12, 31)  # Youngest student born at the end of the year
    return (start_date + timedelta(days=random.randint(0, (end_date - start_date).days))).date()


def generate_students(students_count: int):
    pd.set_option('display.width', 1000)

    # Jmena sloupcu tabulky
    col_names_list = ["student_uid", "trida_id", "jmeno", "prijmeni", "pohlavi", "datum_narozeni", "odklad", "spolu",
                      "ne_spolu_01", "ne_spolu_02", "ne_spolu_03", "ne_spolu_04", "ne_spolu_05", "ne_spolu_06",
                      "ne_spolu_07"]

    name_list = ["Kvido", "Jarmila", "Borek", "Kazimira", "Radim", "Ludmila", "Blažena", "Horomír"]
    surname_list = ["Kido", "Jila", "Brek", "Kazra", "Kvak", "Ldila", "Blana", "Hrom"]
    class_id_list = ["A", "B", "C", "D"]
    divka_kluk_list = ["D", "K"]

    rand_bin_025 = [0, 0, 0, 1]
    rand_bin_01 = [1 if i in range(0, 1) else 0 for i in range(10)]
    rand_bin_005 = [1 if i in range(0, 5) else 0 for i in range(100)]

    df = pd.DataFrame()
    df[col_names_list[0]] = range(students_count)
    df[col_names_list[1]] = np.random.choice(class_id_list, students_count)
    df[col_names_list[2]] = np.random.choice(name_list, students_count)
    df[col_names_list[3]] = np.random.choice(surname_list, students_count)
    df[col_names_list[4]] = np.random.choice(divka_kluk_list, students_count)
    df[col_names_list[5]] = [generate_student_age() for _ in range(students_count)]  # Generating birth dates
    df[col_names_list[6]] = np.random.choice(rand_bin_025, students_count)
    df[col_names_list[7]] = np.random.choice(rand_bin_025, students_count)
    df[col_names_list[8]] = np.random.choice(rand_bin_005, students_count)
    df[col_names_list[9]] = np.random.choice(rand_bin_005, students_count)

    for i in range(10, len(col_names_list)):
        df[col_names_list[i]] = np.random.choice(rand_bin_01, students_count)

    print(df)

    df.to_excel("input_data/students_03.xlsx", sheet_name='studenti', index=False)


if __name__ == "__main__":
    generate_students(STUDENTS_COUNT)
