print("\n\nLoading system...\n\n")

import joblib
import pandas as pd
from tkinter import *
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class System:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        self.current_system_model = joblib.load('model/main/xgb_model.pkl')

    def process_user_review(self):
        self.user_review_value = user_review_text.get("1.0",'end-1c')

        for result in self.classifier(self.user_review_value, truncation='longest_first', max_length=512):
            self.confidence = result['score']
            self.sentiment = result['label']
        
        return self.sentiment, self.confidence
    
    def process_features(self):
        self.upvotes_value = int(upvotes_entry.get())
        self.total_votes_value = int(total_votes_entry.get())
        self.rating_value = int(rating_entry.get())

        return self.upvotes_value, self.total_votes_value, self.rating_value
    
    def predict(self):
        self.sentiment, self.confidence = self.process_user_review()
        self.upvotes_value, self.total_votes_value, self.rating_value = self.process_features()

        self.data = pd.DataFrame({'upvotes': [self.upvotes_value], 
                            'total_votes': [self.total_votes_value], 
                            'rating': [self.rating_value], 
                            'sentiment': [self.sentiment], 
                            'confidence': [self.confidence]})
        #print('/n' + self.data + '/n')
        self.sentiment_mapping = {0 : 'Negative', 1 : 'Neutral', 2 : 'Positive'}
        
        self.X = self.data.drop(columns=['sentiment'])
        self.y = self.data['sentiment']
        
        self.single_row = pd.DataFrame(self.X.iloc[0]).transpose()
        self.y_pred = self.current_system_model.predict(self.single_row)

        self.prediction_label = Label(window, text="Prediction: " + self.sentiment_mapping[self.y_pred[0]])
        self.prediction_label.grid(row=5, column=0, columnspan=10)

if __name__ == "__main__":
    system = System()

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

    predict_button = Button(window, text="Predict", command=system.predict)
    predict_button.grid(row=4, column=2, padx=10, pady=10)

    window.mainloop()

    print("\n\nSystem exited...\n\n")