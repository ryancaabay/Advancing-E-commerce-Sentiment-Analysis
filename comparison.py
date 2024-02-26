import joblib
import pandas as pd
from tkinter import *
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


data = pd.read_csv('reviews.csv')
data = data.iloc[0:1000]
cols_to_drop = ['Id', 'ProductId', 'UserId', 'ProfileName', 'Time', 'Summary']
data = data.drop(columns=cols_to_drop).rename(columns={'Text': 'review', 'Score': 'rating', 'HelpfulnessNumerator': 'upvotes', 'HelpfulnessDenominator': 'total_votes'})

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
current_system_model = joblib.load('model/main/xgb_model.pkl')

confidence = []
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
sentiment = []


for result in classifier(data['review'].tolist(), truncation='longest_first', max_length=512):
    confidence.append(result['score'])
    sentiment.append(sentiment_mapping[result['label']])
    
data['confidence'] = confidence
data['sentiment'] = sentiment
data.to_csv('dataset/roberta_only.csv', index=False)


# use pre-trained model to predict unseen data
new_data = pd.read_csv('dataset/roberta_only.csv')
reviews = new_data['review'].copy()  
new_data = new_data.drop(['review', 'sentiment'], axis='columns')

X = new_data
y_pred = current_system_model.predict(X)


# save new dataframe to csv for comparison
new_data['review'] = reviews  
new_data['comparison_sentiment'] = y_pred
new_data.to_csv('dataset/roberta_xgb.csv', index=False)