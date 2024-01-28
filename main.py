import pandas as pd
from tkinter import *

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model import saved_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 


data = pd.DataFrame(columns=['upvotes', 'total_votes', 'rating', 'sentiment', 'confidence'])
    
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def process_user_review():
    global data
    user_review_value = user_review_text.get("1.0",'end-1c')

    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

    labels = []
    scores = []

    for result in classifier(user_review_value, truncation='longest_first', max_length=512):
        labels.append(sentiment_mapping[result['label']])
        scores.append(result['score'])

    data = data.assign(sentiment=labels, confidence=scores)
    

def process_features():
    global data
    upvotes_value = upvotes_entry.get("1.0",'end-1c')
    total_votes_value = total_votes_entry.get("1.0",'end-1c')
    rating_value = rating_entry.get("1.0",'end-1c')
    upvotes_value.append(int(data['upvotes']))
    total_votes_value.append(int(data['total_votes']))
    rating_value.append(int(data['rating']))


def predict():
    global data

    process_user_review()
    process_features()

    X = data.drop(columns=['sentiment'])
    y = data['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    y_pred = saved_model.predict(X_test)
    y_train_pred = saved_model.predict(X_train)

    print('\nTrain Accuracy Score: ', accuracy_score(y_train, y_train_pred))
    print('Test Accuracy Score: ', accuracy_score(y_test, y_pred,), '\n')

    print('Train Precision Score: ', precision_score(y_train, y_train_pred, average='weighted'))
    print('Test Precision Score: ', precision_score(y_test, y_pred, average='weighted'), '\n')

    print('Train Recall Score: ', recall_score(y_train, y_train_pred, average='weighted'))
    print('Test Recall Score: ', recall_score(y_test, y_pred, average='weighted'), '\n')

    print('Train F1 Score: ', f1_score(y_train, y_train_pred, average='weighted'))
    print('Test F1 Score: ', f1_score(y_test, y_pred, average='weighted'), '\n')


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


# Add a button to trigger the prediction
predict_button = Button(window, text="Predict", command=predict)
predict_button.grid(row=4, column=2, padx=10, pady=10)

# Run the tkinter event loop
window.mainloop()