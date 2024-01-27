import pandas as pd
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from model import saved_model


data = pd.read_csv('dataset/reviews_preprocessed.csv')
data = data.drop(['review'], axis='columns')

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

score = cross_val_score(saved_model, X, y, cv=10)
#print(score)
print('Cross-Validation Score: ', score.mean(), '\n')

plot_importance(saved_model)
plt.show()