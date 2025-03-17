from matplotlib import pyplot as plt
import pandas as pd


def visualize_sa(costs, size_devs, boys_devs, girls_devs, probabilities, temperatures, max_iterations, filename):
    plt.figure(figsize=(15, 12))

    # Fitness Cost
    plt.subplot(3, 2, 1)
    plt.plot(range(max_iterations), costs, label="Fitness Cost", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Cost")
    plt.title("Fitness Cost vs Iterations")
    plt.legend()

    # Size Deviation
    plt.subplot(3, 2, 2)
    plt.plot(range(max_iterations), size_devs, label="Size Deviation", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Size Deviation")
    plt.title("Class Size Deviation vs Iterations")
    plt.legend()

    # Boys Deviation
    plt.subplot(3, 2, 3)
    plt.plot(range(max_iterations), boys_devs, label="Boys Deviation", color="green")
    plt.xlabel("Iterations")
    plt.ylabel("Boys Deviation")
    plt.title("Boys Deviation vs Iterations")
    plt.legend()

    # Girls Deviation
    plt.subplot(3, 2, 4)
    plt.plot(range(max_iterations), girls_devs, label="Girls Deviation", color="purple")
    plt.xlabel("Iterations")
    plt.ylabel("Girls Deviation")
    plt.title("Girls Deviation vs Iterations")
    plt.legend()

    # Acceptance Probability
    plt.subplot(3, 2, 5)
    plt.plot(range(len(probabilities)), probabilities, label="Acceptance Probability", color="brown")
    plt.xlabel("Iterations")
    plt.ylabel("Acceptance Probability")
    plt.title("Decrease in Acceptance Probability Over Time")
    plt.legend()

    # Temperature
    plt.subplot(3, 2, 6)
    plt.plot(range(max_iterations), temperatures, label="Temperature", color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Temperature")
    plt.title("Temperature vs Iterations")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"output_data/{filename}")


def visualize_bs(costs, size_devs, boys_devs, girls_devs, max_iterations, filename):
    plt.figure(figsize=(12, 8))

    # Fitness Cost
    plt.subplot(2, 2, 1)
    plt.plot(range(max_iterations), costs, label="Fitness Cost", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Cost")
    plt.title("Fitness Cost vs Iterations")
    plt.legend()

    # Size Deviation
    plt.subplot(2, 2, 2)
    plt.plot(range(max_iterations), size_devs, label="Size Deviation", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Size Deviation")
    plt.title("Class Size Deviation vs Iterations")
    plt.legend()

    # Boys Deviation
    plt.subplot(2, 2, 3)
    plt.plot(range(max_iterations), boys_devs, label="Boys Deviation", color="green")
    plt.xlabel("Iterations")
    plt.ylabel("Boys Deviation")
    plt.title("Boys Deviation vs Iterations")
    plt.legend()

    # Girls Deviation
    plt.subplot(2, 2, 4)
    plt.plot(range(max_iterations), girls_devs, label="Girls Deviation", color="purple")
    plt.xlabel("Iterations")
    plt.ylabel("Girls Deviation")
    plt.title("Girls Deviation vs Iterations")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"output_data/visualisation/{filename}")


def visualize_hall_of_fame(hall_of_fame):
    for i, (classes, score) in enumerate(hall_of_fame):
        print(f"\n--- Hall of Fame Solution {i + 1} --- (Fitness Score: {score})")

        # Create a DataFrame to show class assignments
        data = []
        for class_idx, cls in enumerate(classes):
            for student in cls:
                data.append([class_idx + 1, student["student_uid"], student["pohlavi"]])  # Add more attributes if needed

        df = pd.DataFrame(data, columns=["Class", "Student ID", "Gender"])

        # Display using Pandas formatting
        print(df.to_string(index=False))  # Print without row index for clarity
