import pandas as pd

data = pd.read_csv('dataset/reviews_preprocessed.csv')
data = data.drop(['review'], axis='columns')

from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


window = Tk()
window.geometry("500x400")
window.title("Sentiment Predictor")
#window.resizable(False, False)

user_review = Text(window, width=50, height=10)
user_review.pack(padx=(5, 5), pady=(5, 5))

"""
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
"""
upvotes_value = 0
total_votes_value = 0
rating_value = 0


label = Label(window, text="attribute")
label.pack(padx=(5, 5), pady=(5, 5))
text = Text(window, state="normal")
text.pack(padx=(5, 5), pady=(5, 5))


# Add a button to trigger the prediction
predict_button = Button(window, text="Predict")
predict_button.pack(padx=(5, 5), pady=(5, 5))

# Run the tkinter event loop
window.mainloop()