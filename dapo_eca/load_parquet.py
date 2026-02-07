# load /home/tiger/jason/eca/dapo_eca/80-20-data/math__combined_54.4k.parquet and print first few rows
# print all columns
import pandas as pd

df = pd.read_parquet("/home/tiger/verl/data/math__aime2025_repeated_32x_960.parquet")
df = df.sample(frac=1, random_state=42)
# print first element
print(df.iloc[0]["prompt"][0]["content"])
print(df.keys())
print(df.head())
