import pandas as pd


def export_hall_of_fame(hall_of_fame, filename):
    """
    Export the Hall of Fame solutions to an Excel file.
    Each Hall of Fame solution will be in a separate sheet.
    """
    with pd.ExcelWriter(f"output_data/class_assignments/{filename}") as writer:
        for i, (classes, score) in enumerate(hall_of_fame):
            # Prepare data for DataFrame
            data = []
            for class_idx, cls in enumerate(classes):
                for student in cls:
                    data.append([class_idx + 1, int(student["student_uid"]), student["jmeno"], student["prijmeni"],
                                 student["pohlavi"]])

            # Create DataFrame
            df = pd.DataFrame(data, columns=["Class", "Student ID", "First name", "Last name", "Gender"])

            # Write each solution to a separate sheet
            df.to_excel(writer, sheet_name=f"Solution_{i + 1}_Score_{score}", index=False)

    print(f"Hall of Fame exported to {filename}")
