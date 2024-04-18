import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
import numpy as np

# Load the preprocessed dataset
df = pd.read_csv('dataset/reviews_preprocessed.csv')

# Split the dataset into training and testing sets
X = df['review'].values
y = df['sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Build the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Predict the classes
y_pred = model.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')