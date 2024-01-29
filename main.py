import pandas as pd
from tkinter import *
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model import saved_model
from sklearn.model_selection import cross_val_score


model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def process_user_review():
    user_review_value = user_review_text.get("1.0",'end-1c')

    for result in classifier(user_review_value, truncation='longest_first', max_length=512):
        sentiment = [result['label']]
        confidence = int((result['score']))
    
    return sentiment, confidence

def process_features():
    upvotes_value = int(upvotes_entry.get())
    total_votes_value = int(total_votes_entry.get())
    rating_value = int(rating_entry.get())

    return upvotes_value, total_votes_value, rating_value

def predict():
    sentiment, confidence = process_user_review()
    upvotes_value, total_votes_value, rating_value = process_features()

    data = pd.DataFrame({'upvotes': [upvotes_value], 
                         'total_votes': [total_votes_value], 
                         'rating': [rating_value], 
                         'sentiment': [sentiment], 
                         'confidence': [confidence]})
    
    sentiment_mapping = {0 : 'Negative', 1 : 'Neutral', 2 : 'Positive'}
    
    X = data.drop(columns=['sentiment'])
    y = data['sentiment']
    
    single_row = pd.DataFrame(X.iloc[0]).transpose()
    y_pred = saved_model.predict(single_row)

    prediction_label = Label(window, text="Prediction: " + sentiment_mapping[y_pred[0]])
    prediction_label.grid(row=5, column=0, columnspan=10)

window = Tk()
window.geometry("450x400")
window.title("Sentiment Predictor")
window.resizable(False, False)

user_review_text = Text(window, width=53, height=10)
user_review_text.grid(row=0, column=0, columnspan=10, padx=11, pady=10)

upvotes_label = Label(window, text="Upvotes: ")
upvotes_label.grid(row=1, column=0, sticky=E)
upvotes_entry = Entry(window, width=10)
upvotes_entry.grid(row=1, column=1, sticky=W)

total_votes_label = Label(window, text="Total Votes: ")
total_votes_label.grid(row=2, column=0, sticky=E)
total_votes_entry = Entry(window, width=10)
total_votes_entry.grid(row=2, column=1, sticky=W)

rating_label = Label(window, text="Rating: ")
rating_label.grid(row=3, column=0, sticky=E)
rating_entry = Entry(window, width=10)
rating_entry.grid(row=3, column=1, sticky=W)

predict_button = Button(window, text="Predict", command=predict)
predict_button.grid(row=4, column=2, padx=10, pady=10)

window.mainloop()