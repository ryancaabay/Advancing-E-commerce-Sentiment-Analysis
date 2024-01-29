import pandas as pd
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib


params={
    "objective": ['multi:softmax'],
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2 , 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5 , 0.7],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9]
}

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now()-start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\nTime taken: %i hours %i minutes and %s seconds.\n' % (thour, tmin, round(tsec, 2)))

data = pd.read_csv('dataset/reviews_preprocessed.csv')
data = data.drop(['review'], axis='columns')

X = data.drop(columns=['sentiment'])
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

"""
model = XGBClassifier()

grid_search = GridSearchCV(model, param_grid=params, refit=True, scoring='roc_auc_ovr', n_jobs=-1, cv=5, verbose=3)

start_time = timer(None)
grid_search.fit(X_train, y_train)
timer(start_time)

print(grid_search.best_estimator_)

best_params = grid_search.best_params_

# Print the best parameters
print(best_params, '\n')
"""

best_params = {
    "colsample_bytree": 0.5, 
    "gamma": 0.4, 
    "learning_rate": 0.05, 
    "max_depth": 4, 
    "min_child_weight": 7, 
    "objective": 'multi:softmax',
    "subsample": 0.8
}

classifier = XGBClassifier(**best_params)
classifier.fit(X_train, y_train)

joblib.dump(classifier, 'model/xgb_model.pkl')
saved_model = joblib.load('model/xgb_model.pkl')

