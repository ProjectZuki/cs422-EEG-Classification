import pandas as pd

# file to analyze
parquet_file_path = input("Enter file path to analyze: ")

# Read the Parquet file into a DataFrame
df = pd.read_parquet(parquet_file_path)

# sample first rows
print("Sample Data:")
print(df.head())

# output column labels
print("\nColumn Labels:")
print(df.columns.tolist())

# output dataframe information
print("\nDataFrame Information:")
df.info()
