import pandas as pd


df = pd.read_csv('dataset/reviews.csv')
df = df.iloc[:1000]
df.to_csv('dataset/reviews_compressed.csv', index=False)