import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv('reviews_preprocessed.csv')
data = data.drop(['review'], axis='columns')

X = data.drop(columns=['sentiment'])
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='multi:softmax')

bst.fit(X_train, y_train)

y_pred = bst.predict(X_test)
y_train_pred = bst.predict(X_train)


from sklearn.metrics import accuracy_score
print('Train Accuracy: ', accuracy_score(y_train, y_train_pred))
print('Test Accuracy: ', accuracy_score(y_test, y_pred))

