print("\n\nLoading system...\n\n")

import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 


class Dataset:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def drop_columns(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
        self.dataframe = self.dataframe.drop(columns = self.columns_to_drop)
    
    def rename_columns(self, columns_to_rename):
        self.columns_to_rename = columns_to_rename
        self.dataframe = self.dataframe.rename(columns = self.columns_to_rename)

    def define_row_count(self, initial_row, final_row):  
        self.initial_row = initial_row
        self.final_row = final_row
        self.dataframe = self.dataframe.iloc[self.initial_row:self.final_row]
    
    def save_csv(self):
        self.set_dataset_name = input('\n\nEnter the name of the preprocessed dataset to be saved: ')
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        self.dataframe.to_csv(f'dataset/{self.set_dataset_name}.csv', index=False)


class BERT:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    def append_bert_embeddings(self, dataframe, sentiment_mapping, sentiment, confidence):
        self.dataframe = dataframe
        self.sentiment_mapping = sentiment_mapping
        self.confidence = confidence
        self.sentiment = sentiment

        for result in self.classifier(self.dataframe['review'].tolist(), truncation='longest_first', max_length=512):
            self.confidence.append(result['score'])
            self.sentiment.append(self.sentiment_mapping[result['label']])

        self.dataframe['confidence'] = self.confidence
        self.dataframe['sentiment'] = self.sentiment


class XGBoost:
    def __init__(self, model):
        self.model = model
    
    def split_data(self, data, X, y):
        self.data = data
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=.2)

    def search_best_params(self, grid_search):
        self.grid_search = grid_search
        self.grid_search.fit(self.X_train, self.y_train)
        self.best_params = self.grid_search.best_params_

    def set_best_params(self, best_params):
        self.best_params = best_params
    
    def classify_sentiment(self):
        self.classifier = self.model
        self.classifier.set_params(**self.best_params)
        self.classifier.fit(self.X_train, self.y_train)

    def save_model(self):
        self.set_model_name = input('\nEnter the name of the trained model to be saved: ')
        if not os.path.exists('model'):
            os.makedirs('model')
        self.set_folder_name = input('\nEnter the name of the folder where the saved model and variables are to be stored: ')
        if not os.path.exists(f'model/{self.set_folder_name}'):
            os.makedirs(f'model/{self.set_folder_name}')
        
        joblib.dump(self.classifier, f'model/{self.set_folder_name}/{self.set_model_name}.pkl')
        joblib.dump(self.X, (f'model/{self.set_folder_name}/X.pkl'))
        joblib.dump(self.y, (f'model/{self.set_folder_name}/y.pkl'))
        joblib.dump(self.X_train, (f'model/{self.set_folder_name}/X_train.pkl'))
        joblib.dump(self.X_test, (f'model/{self.set_folder_name}/X_test.pkl'))
        joblib.dump(self.y_train, (f'model/{self.set_folder_name}/y_train.pkl'))
        joblib.dump(self.y_test, (f'model/{self.set_folder_name}/y_test.pkl'))


def preprocess_dataset():
    dataframe = pd.read_csv('reviews.csv')
    print('\n\tDefine the range of rows to be used in the dataset\n')
    time.sleep(1)
    initial_row = int(input('\t\tEnter the initial row: '))
    final_row = int(input('\t\tEnter the final row: '))
    columns_to_drop = ['Id', 'ProductId', 'UserId', 'ProfileName', 'Time', 'Summary']
    columns_to_rename = {'Text': 'review', 'Score': 'rating', 'HelpfulnessNumerator': 'upvotes', 'HelpfulnessDenominator': 'total_votes'}
    print(' ')

    df = Dataset(dataframe)
    df.drop_columns(columns_to_drop)
    df.rename_columns(columns_to_rename)
    df.define_row_count(initial_row, final_row)

    return df


def generate_bert_embeddings(df):
    bert_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    sentiment = []
    confidence = []

    roberta = BERT(bert_model_name)
    roberta.append_bert_embeddings(df.dataframe, sentiment_mapping, sentiment, confidence)
    df.save_csv()


def train_xgboost_model():
    data_name = input('\n\tEnter the file name of the dataset to train XGBoost with (Hint: file_name.csv): ')
    print(' ')
    data = pd.read_csv(f'dataset/{data_name}').drop(['review'], axis='columns')
    X = data.drop(columns=['sentiment'])
    y = data['sentiment']

    params = {
                "objective": ['multi:softmax'],
                "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
                "min_child_weight": [1, 3, 5, 7],
                "gamma": [0.0, 0.1, 0.2 , 0.3, 0.4],
                "colsample_bytree": [0.3, 0.4, 0.5 , 0.7],
                "subsample": [0.5, 0.6, 0.7, 0.8, 0.9]
             } 
    
    params_preset = {
                "colsample_bytree": 0.5, 
                "gamma": 0.4, 
                "learning_rate": 0.05, 
                "max_depth": 4, 
                "min_child_weight": 7, 
                "objective": 'multi:softmax',
                "subsample": 0.8
             }
    
    xgb_model = XGBClassifier()

    grid_search = GridSearchCV(xgb_model, param_grid = params, refit=True, scoring='roc_auc_ovr', n_jobs=-1, cv=5, verbose=3)

    xgb = XGBoost(xgb_model)
    xgb.split_data(data, X, y)
    xgb.search_best_params(grid_search)
    print(' ')
    print(xgb.best_params)
    #xgb.set_best_params(params_preset)
    print(' ')
    xgb.classify_sentiment()
    xgb.save_model()

    return xgb


def display_evaluation_metrics():
    model_folder = input('\n\tEnter the name of the folder where model and variables are to be retrieved: ')
    model_name = input('\n\tEnter the file name of the model to be evaluated (Hint: file_name.pkl): ')

    current_model = joblib.load(f'model/{model_folder}/{model_name}')
    X = joblib.load(f'model/{model_folder}/X.pkl')
    y = joblib.load(f'model/{model_folder}/y.pkl')
    X_train = joblib.load(f'model/{model_folder}/X_train.pkl')
    X_test = joblib.load(f'model/{model_folder}/X_test.pkl')
    y_train = joblib.load(f'model/{model_folder}/y_train.pkl')
    y_test = joblib.load(f'model/{model_folder}/y_test.pkl')

    y_train_pred = current_model.predict(X_train)
    y_pred = current_model.predict(X_test)

    print('\n\nHere are the evaluation metrics for the current model\n')
    print('\tTrain Accuracy Score: ', accuracy_score(y_train, y_train_pred))
    print('\tTest Accuracy Score: ', accuracy_score(y_test, y_pred,), '\n')

    print('\tTrain Precision Score: ', precision_score(y_train, y_train_pred, average='weighted'))
    print('\tTest Precision Score: ', precision_score(y_test,y_pred, average='weighted'), '\n')

    print('\tTrain Recall Score: ', recall_score(y_train, y_train_pred, average='weighted'))
    print('\tTest Recall Score: ', recall_score(y_test,y_pred, average='weighted'), '\n')

    print('\tTrain F1 Score: ', f1_score(y_train, y_train_pred, average='weighted'))
    print('\tTest F1 Score: ', f1_score(y_test, y_pred, average='weighted'), '\n')

    #cv_score = cross_val_score(current_model, X, y, cv=10)
    print('\tCross-Validation Score: ', cross_val_score(current_model, X, y, cv=10).mean())

    plot_importance(current_model)
    plt.show()


if __name__ == "__main__":
    print("\n\nSystem loaded successfully...")
    time.sleep(1)

    start_time = time.time()
    
    get_user_confirmation = input(f"\n\nDo you want to preprocess_dataset() and generate_bert_embeddings()? (yes/no): ")
    if get_user_confirmation.lower() == 'yes':
        df = preprocess_dataset()
        generate_bert_embeddings(df)

    get_user_confirmation = input("\n\nDo you want to train_xgboost_model()? (yes/no): ")
    if get_user_confirmation.lower() == 'yes':
        train_xgboost_model()

    get_user_confirmation = input(f"\n\nDo you want to display_evaluation_metrics()? (yes/no): ")
    if get_user_confirmation.lower() == 'yes':
        display_evaluation_metrics()

    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n\nTime elapsed: {str(int(hours)).zfill(2)}:{str(int(minutes)).zfill(2)}:{str(int(seconds)).zfill(2)}")

    print("\n\nExiting system...\n\n")
    time.sleep(1)