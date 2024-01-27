import pandas as pd

# Read the CSV file
df = pd.read_csv('dataset/reviews.csv')

# Slice the DataFrame to keep only the first 1000 lines
df = df.iloc[:1000]

# Save the sliced DataFrame to a new CSV file
df.to_csv('dataset/reviews_compressed.csv', index=False)