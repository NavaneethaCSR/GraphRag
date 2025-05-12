import pandas as pd

# Read the Parquet file
parquet_file = r'C:\Users\LENOVO\Downloads\create_summarized_entities (3).parquet'

# Load the Parquet file into a DataFrame
df = pd.read_parquet(parquet_file)

# Save the DataFrame as a CSV file
csv_file = r'C:\Users\LENOVO\Downloads\create_summarized_entities (3).csv'
df.to_csv(csv_file, index=False)

print(f"CSV file has been saved as {csv_file}")
