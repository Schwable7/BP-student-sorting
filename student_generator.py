import numpy as np
import pandas as pd
import random

from datetime import datetime, timedelta
from constants import STUDENTS_COUNT, FEMALE, MALE, ID, CLASS_ID, FIRST_NAME, LAST_NAME, GENDER, BIRTH_DATE, DEFERRAL, \
    TOGETHER, LEARNING_DISABILITIES, TALENT, DIFF_MOTHER_LANG, NOT_TOGETHER_01, NOT_TOGETHER_02, NOT_TOGETHER_03, \
    NOT_TOGETHER_04, NOT_TOGETHER_05, NOT_TOGETHER_06, NOT_TOGETHER_07, STUDENTS_FILENAME, TOGETHER_01, TOGETHER_02, \
    TOGETHER_03, TOGETHER_04, TOGETHER_05, TOGETHER_06, TOGETHER_07


def generate_student_age():
    # Assuming students are between 6 and 7 years old (1st year primary school)
    start_date = datetime(datetime.today().year - 7, 1, 1)  # Oldest student born at the start of the year
    end_date = datetime(datetime.today().year - 6, 12, 31)  # Youngest student born at the end of the year
    return (start_date + timedelta(days=random.randint(0, (end_date - start_date).days))).date()


def generate_students(students_count: int, iteration: int):
    pd.set_option('display.width', 1000)

    # Jmena sloupcu tabulky
    col_names_list = [ID, CLASS_ID, FIRST_NAME, LAST_NAME, GENDER, BIRTH_DATE, DEFERRAL, TOGETHER,
                      TOGETHER_01, TOGETHER_02, TOGETHER_03, TOGETHER_04, TOGETHER_05, TOGETHER_06, TOGETHER_07,
                      LEARNING_DISABILITIES, TALENT, DIFF_MOTHER_LANG, NOT_TOGETHER_01, NOT_TOGETHER_02,
                      NOT_TOGETHER_03, NOT_TOGETHER_04, NOT_TOGETHER_05, NOT_TOGETHER_06, NOT_TOGETHER_07]

    name_list = ["Kvido", "Jarmila", "Borek", "Kazimira", "Radim", "Ludmila", "Blažena", "Horomír"]
    surname_list = ["Kido", "Jila", "Brek", "Kazra", "Kvak", "Ldila", "Blana", "Hrom"]
    class_id_list = ["A", "B", "C", "D"]
    gender_list = [FEMALE, MALE]

    rand_bin_025 = [0, 0, 0, 1]
    rand_bin_002 = [1 if i in range(0, 2) else 0 for i in range(100)]
    rand_bin_005 = [1 if i in range(0, 5) else 0 for i in range(100)]

    df = pd.DataFrame()
    df[ID] = range(students_count)
    df[CLASS_ID] = np.random.choice(class_id_list, students_count)
    df[FIRST_NAME] = np.random.choice(name_list, students_count)
    df[LAST_NAME] = np.random.choice(surname_list, students_count)

    df[GENDER] = np.random.choice(gender_list, students_count)
    df[BIRTH_DATE] = [generate_student_age() for _ in range(students_count)]
    df[DEFERRAL] = np.random.choice(rand_bin_025, students_count)
    # df[TOGETHER] = np.random.choice(rand_bin_025, students_count)

    # New rare features
    df[LEARNING_DISABILITIES] = np.random.choice(rand_bin_005, students_count)
    df[TALENT] = np.random.choice(rand_bin_005, students_count)
    df[DIFF_MOTHER_LANG] = np.random.choice(rand_bin_005, students_count)

    # Not together flags (1% chance)
    for col in [NOT_TOGETHER_01, NOT_TOGETHER_02, NOT_TOGETHER_03,
                NOT_TOGETHER_04, NOT_TOGETHER_05, NOT_TOGETHER_06, NOT_TOGETHER_07]:
        df[col] = np.random.choice(rand_bin_005, students_count)

    # together flags (1% chance)
    for col in [TOGETHER_01, TOGETHER_02,
                TOGETHER_03, TOGETHER_04, TOGETHER_05, TOGETHER_06, TOGETHER_07]:
        df[col] = np.random.choice(rand_bin_005, students_count)

    print(df)

    df.to_excel(f"input_data/students_555_{iteration}.xlsx", sheet_name='studenti', index=False)


if __name__ == "__main__":
    for i in range(1):
        print(f"Generating students {i + 1}/10")
        generate_students(STUDENTS_COUNT, i)
