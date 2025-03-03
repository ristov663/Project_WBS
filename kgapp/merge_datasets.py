import pandas as pd
import glob

# Read all CSV files in the current directory
csv_files = glob.glob("kgapp/datasets/*.csv")  # Change the path if files are not in the same directory

dfs = []  # List to store all DataFrames

for file in csv_files:
    df = pd.read_csv(file)  # Read CSV
    df = df.drop(columns=["No."], errors="ignore")  # Drop "No." column if it exists
    dfs.append(df)

# Merge based on common columns
merged_df = pd.concat(dfs, join="inner", ignore_index=True)

# Save the final file
merged_df.to_csv("kgapp/datasets/datasets.csv", index=False)

print("Merge completed successfully!")
