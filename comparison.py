import pandas as pd
from tkinter import *
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model import saved_model


# load unseen data from row 1000 to 2000
data = pd.read_csv('dataset/reviews.csv')
data = data.iloc[1000:2000]
cols_to_drop = ['Id', 'ProductId', 'UserId', 'ProfileName', 'Time', 'Summary']
data = data.drop(columns=cols_to_drop).rename(columns={'Text': 'review', 'Score': 'rating', 'HelpfulnessNumerator': 'upvotes', 'HelpfulnessDenominator': 'total_votes'})


# perform pre-processing
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
sentiment = []
confidence = []

for result in classifier(data['review'].tolist(), truncation='longest_first', max_length=512):
    sentiment.append(sentiment_mapping[result['label']])
    confidence.append(result['score'])

data['sentiment'] = sentiment
data['confidence'] = confidence
data.to_csv('dataset/first.csv', index=False)


# use pre-trained model to predict unseen data
new_data = pd.read_csv('dataset/first.csv')
reviews = new_data['review'].copy()  
new_data = new_data.drop(['review', 'sentiment'], axis='columns')

X = new_data
y_pred = saved_model.predict(X)

# save new dataframe to csv for comparison
new_data['review'] = reviews  
new_data['comparison_sentiment'] = y_pred
new_data.to_csv('dataset/second.csv', index=False)

