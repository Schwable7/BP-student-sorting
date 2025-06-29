import pandas as pd
from datetime import datetime

from constants import ID, CLASS_ID, FIRST_NAME, LAST_NAME, GENDER, BIRTH_DATE, AGE, DEFERRAL, TOGETHER, \
    LEARNING_DISABILITIES, TALENT, DIFF_MOTHER_LANG


def load_students(filename: str) -> list[dict]:
    # Read Excel file
    df = pd.read_excel(filename, dtype=str)

    # Identify dynamic "ne_spolu_xx" columns
    ne_spolu_cols = [col for col in df.columns if col.startswith("not_together_")]

    # Identify dynamic "spolu_xx" columns
    spolu_cols = [col for col in df.columns if col.startswith("together_")]

    # Convert dataframe to structured list
    students = []
    for _, row in df.iterrows():
        birth_date = datetime.strptime(row[BIRTH_DATE].strip(), "%Y-%m-%d %H:%M:%S").date() if BIRTH_DATE in df.columns else None
        age = datetime.today().year - birth_date.year if birth_date else None

        student = {
            ID: int(row[ID]),
            CLASS_ID: row[CLASS_ID],
            FIRST_NAME: row[FIRST_NAME],
            LAST_NAME: row[LAST_NAME],
            GENDER: row[GENDER],
            AGE: age,
            DEFERRAL: int(row[DEFERRAL]),
            # TOGETHER: int(row[TOGETHER]),
            LEARNING_DISABILITIES: int(row[LEARNING_DISABILITIES]),
            TALENT: int(row[TALENT]),
            DIFF_MOTHER_LANG: int(row[DIFF_MOTHER_LANG])
        }
        # Add "ne_spolu_xx" fields separately
        for col in ne_spolu_cols:
            student[col] = row[col] if pd.notna(row[col]) else None
        # Add "spolu_xx" fields separately
        for col in spolu_cols:
            student[col] = row[col] if pd.notna(row[col]) else None

        students.append(student)

    return students


if __name__ == "__main__":
    students = load_students("input_data/students_03.xlsx")
    for student in students:
        print(student)
