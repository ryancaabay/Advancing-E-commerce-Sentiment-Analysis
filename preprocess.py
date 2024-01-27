import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

df = pd.read_csv('reviews_compressed.csv')
cols_to_drop = ['Id', 'ProductId', 'UserId', 'ProfileName', 'Time', 'Summary']
df = df.drop(columns=cols_to_drop).rename(columns={'Text': 'review', 'Score': 'rating', 'HelpfulnessNumerator': 'upvotes', 'HelpfulnessDenominator': 'total_votes'})

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
results = classifier(df['review'].tolist(), truncation='longest_first', max_length=512)

sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

labels = []
scores = []

for result in classifier(df['review'].tolist(), truncation='longest_first', max_length=512):
    labels.append(sentiment_mapping[result['label']])
    scores.append(result['score'])

df = df.assign(sentiment=labels, confidence=scores)
df.to_csv('reviews_preprocessed.csv', index=False)