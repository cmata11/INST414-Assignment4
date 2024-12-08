import pandas as pd

# Load the dataset from the CSV file
file_name = "tracking_data.csv"

# Read the CSV and specify the columns to display
columns_to_display = [
    "Calories_Burned",
    "Workout_Type",
    "Session_Duration (hours)",
    "Fat_Percentage",
    "Max_BPM",
    "Avg_BPM",
    "Workout_Frequency (days/week)"
]

try:
    data = pd.read_csv(file_name, usecols=columns_to_display)
    print("First 50 Rows of Selected Features from the Dataset:")
    print(data.head(50))  # Display the first 50 rows of the dataset
except ValueError as e:
    print(f"Error loading file or columns: {e}")
except FileNotFoundError:
    print(f"File '{file_name}' not found.")
