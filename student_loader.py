import pandas as pd
from datetime import datetime


def load_students(filename: str) -> list[dict]:
    # Read Excel file
    df = pd.read_excel(filename, dtype=str)

    # Identify dynamic "ne_spolu_xx" columns
    ne_spolu_cols = [col for col in df.columns if col.startswith("ne_spolu_")]

    # Convert dataframe to structured list
    students = []
    for _, row in df.iterrows():
        birth_date = datetime.strptime(row["vek"].strip(), "%Y-%m-%d %H:%M:%S").date() if "vek" in df.columns else None
        age = datetime.today().year - birth_date.year if birth_date else None

        student = {
            "student_uid": row["student_uid"],
            "trida_id": row["trida_id"],
            "jmeno": row["jmeno"],
            "prijmeni": row["prijmeni"],
            "pohlavi": row["pohlavi"],
            "vek": age,
            "odklad": row["odklad"],
            "spolu": row["spolu"],
        }
        # Add "ne_spolu_xx" fields separately
        for col in ne_spolu_cols:
            student[col] = row[col] if pd.notna(row[col]) else None

        students.append(student)

    return students


if __name__ == "__main__":
    students = load_students("students_01.xlsx")
    for student in students:
        print(student)
