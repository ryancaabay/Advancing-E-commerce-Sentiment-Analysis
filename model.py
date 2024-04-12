print("\n\nLoading system...")


import os
import re
import time
import joblib
import pandas as pd
from textblob import TextBlob
from xgboost import XGBClassifier
from unicodedata import normalize
from scipy.stats import uniform, randint
from sklearn.experimental import enable_halving_search_cv  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split, HalvingGridSearchCV, HalvingRandomSearchCV, cross_val_score


class Dataset:
    def __init__(self, dataframe):
        self.dataframe = dataframe


    def drop_columns(self, columns_to_drop):
        self.dataframe = self.dataframe.drop(columns = columns_to_drop)


    def rename_columns(self, columns_to_rename):
        self.dataframe = self.dataframe.rename(columns = columns_to_rename)


    def define_row_count(self, initial_row, final_row):  
        self.dataframe = self.dataframe.iloc[initial_row:final_row]
    

    def remove_html_tags(self):
        self.dataframe['review'] = self.dataframe['review'].apply(lambda x: re.sub('<.*?>|<br />', '', x))


    def remove_accented(self):
        self.dataframe['review'] = self.dataframe['review'].apply(lambda x: normalize('NFKD', x).encode('ASCII', 'ignore').decode())


    def remove_special_characters(self):
        self.dataframe['review'] = self.dataframe['review'].replace('[^A-Za-z0-9 !&-,.?]+', '', regex=True)


    def remove_white_space(self):
        self.dataframe['review'] = self.dataframe['review'].str.replace('\s+', ' ', regex=True)
    

    def append_polarity(self):
        self.dataframe['polarity'] = self.dataframe['review'].apply(lambda x: TextBlob(x).sentiment.polarity)


    def append_subjectivity(self):
        self.dataframe['subjectivity'] = self.dataframe['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)


    def save_csv(self, dataset_name):
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        self.dataframe.to_csv(os.path.join('dataset', f'{dataset_name}.csv'), index=False)


class BERT:
    def __init__(self, model_name):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)


    def append_bert_embeddings(self, dataframe, sentiment_mapping):
        sentiment = []
        confidence = []

        for result in self.classifier(dataframe['review'].tolist(), truncation='longest_first', max_length=512):
            confidence.append(result['score'])
            sentiment.append(sentiment_mapping[result['label']])

        dataframe['confidence'] = confidence
        dataframe['sentiment'] = sentiment


class XGBoost:
    def __init__(self, classifier):
        self.classifier = classifier
        self.best_params = None


    def split_data(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=.2)


    def search_best_params(self, param_search):
        param_search.fit(self.X_train, self.y_train)
        self.best_params = param_search.best_params_


    def set_best_params(self, best_params):
        self.best_params = best_params


    def classify_sentiment(self):
        self.classifier.set_params(**self.best_params)
        self.classifier.fit(self.X_train, self.y_train)


    def save_model(self, model_name, folder_name):
        if not os.path.exists('model'):
            os.makedirs('model')
        if not os.path.exists(f'model/{folder_name}'):
            os.makedirs(f'model/{folder_name}')
        
        joblib.dump(self.classifier, f'model/{folder_name}/{model_name}.pkl')
        joblib.dump(self.X, (f'model/{folder_name}/X.pkl'))
        joblib.dump(self.y, (f'model/{folder_name}/y.pkl'))
        joblib.dump(self.X_train, (f'model/{folder_name}/X_train.pkl'))
        joblib.dump(self.X_test, (f'model/{folder_name}/X_test.pkl'))
        joblib.dump(self.y_train, (f'model/{folder_name}/y_train.pkl'))
        joblib.dump(self.y_test, (f'model/{folder_name}/y_test.pkl'))


def preprocess_dataset():
    dataframe = pd.read_csv('reviews.csv')
    columns_to_drop = ['Id', 'ProductId', 'UserId', 'ProfileName', 'Time', 'Summary']
    columns_to_rename = {'Text': 'review', 'Score': 'rating', 'HelpfulnessNumerator': 'upvotes', 'HelpfulnessDenominator': 'total_votes'}

    print('\n\tDefine the range of rows to be used in the dataset\n')

    time.sleep(1)

    initial_row = int(input('\t\tEnter the initial row: '))
    final_row = int(input('\t\tEnter the final row: '))

    df = Dataset(dataframe)
    df.drop_columns(columns_to_drop)
    df.rename_columns(columns_to_rename)
    df.define_row_count(initial_row, final_row)
    df.remove_html_tags()
    df.remove_accented()
    df.remove_special_characters()
    df.remove_white_space()
    df.append_polarity()
    df.append_subjectivity()

    return df


def generate_bert_embeddings(df):
    #bert_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    bert_model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    #bert_model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    set_dataset_name = input('\nEnter the name of the preprocessed dataset to be saved: ')

    print('\nGenerating BERT embeddings, please wait...\n\n')

    roberta = BERT(bert_model_name)
    roberta.append_bert_embeddings(df.dataframe, sentiment_mapping)
    df.save_csv(set_dataset_name)


def train_xgboost_model():
    dataset_name = input('\n\tEnter the file name of the dataset to train XGBoost with (Hint: file_name.csv): ')

    print(' ')

    dataframe = pd.read_csv(f'dataset/{dataset_name}').drop(['review'], axis='columns')
    X = dataframe.drop(columns=['sentiment'])
    y = dataframe['sentiment']

    '''
    grid_params =   {
                    'max_depth': [3, 5, 7, 9],
                    'min_child_weight': [3, 5, 7, 9],
                    'gamma': [0, 0.1, 0.2, 0.3],
                    'colsample_bytree': [0.3, 0.5, 0.7, 0.9],
                    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9]
                    'learning_rate': [0.001, 0.01, 0.05, 0.1],
                    }'''
    
    halving_grid_params =   {
                            'max_depth': [3, 5, 7, 9],
                            'min_child_weight': [3, 5, 7, 9],
                            'gamma': [0, 0.1, 0.2, 0.3],
                            'subsample': [0.3, 0.5, 0.7, 0.9],
                            'colsample_bytree': [0.3, 0.5, 0.7, 0.9],
                            'learning_rate': [0.001, 0.01, 0.05, 0.1],
                            'n_estimators': [200, 300, 400, 500],
                            'reg_lambda': [1.0, 25.0, 50.0, 100.0],
                            'reg_alpha': [0, 0.2, 0.4, 0.6],
                            }

    halving_random_params = {
                    'max_depth': randint(3, 9),  
                    'min_child_weight': randint(3, 9),
                    'gamma': uniform(0, 0.3),    
                    'subsample': uniform(0.3, 0.9),  
                    'colsample_bytree': uniform(0.3, 0.9),
                    'learning_rate': uniform(0.001, 0.1),
                    'n_estimators': randint(200, 500),      
                    'reg_lambda': uniform(1.0, 100.0),  
                    'reg_alpha': uniform(0, 0.6),  
                    }

    best_params_preset = {
                        'max_depth': 7,
                        'min_child_weight': 4,
                        'gamma': 0.01,
                        'subsample': 0.34,
                        'colsample_bytree': 0.85,
                        'learning_rate': 0.02,
                        'n_estimators': 315,
                        'reg_lambda': 34.73,
                        'reg_alpha': 0.39,
                        }
    
    xgb_model = XGBClassifier(objective='multi:softmax')

    #halving_grid_search = HalvingGridSearchCV(xgb_model, param_grid=halving_grid_params, scoring='roc_auc_ovr', n_jobs=-1, refit=True, cv=10, verbose=3)
    halving_random_search = HalvingRandomSearchCV(xgb_model, param_distributions=halving_random_params, n_candidates='exhaust', factor=2, 
                                                  scoring='roc_auc_ovr', n_jobs=-1, refit=True, cv=10, verbose=3)

    xgb = XGBoost(xgb_model)
    xgb.split_data(X, y)
    #xgb.search_best_params(halving_grid_search)
    xgb.search_best_params(halving_random_search)
    #xgb.set_best_params(best_params_preset)

    print(' ')
    print(xgb.best_params)
    print(' ')

    xgb.classify_sentiment()

    set_model_name = input('\nEnter the name of the trained model to be saved: ')
    set_folder_name = input('\nEnter the name of the folder where the saved model and variables are to be stored: ')

    xgb.save_model(set_model_name, set_folder_name)


def display_evaluation_metrics():
    model_folder = input('\n\tEnter the name of the folder where model and variables are to be retrieved: ')
    model_name = input('\n\tEnter the file name of the model to be evaluated (Hint: file_name.pkl): ')

    current_model, X, y, X_train, X_test, y_train, y_test = (joblib.load(f'model/{model_folder}/{name}') for name 
                                                             in [model_name, 'X.pkl', 'y.pkl', 'X_train.pkl', 'X_test.pkl', 'y_train.pkl', 'y_test.pkl'])

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

    #cross_val_score = cross_val_score(current_model, X, y, cv=10)
    print('\tCross-Validation Score: ', cross_val_score(current_model, X, y, cv=10).mean())


if __name__ == "__main__":
    print("\n\nSystem loaded successfully...")
    
    time.sleep(1)

    start_time = time.time()
    
    get_user_confirmation = input("\n\nDo you want to preprocess a dataset derived from the main dataset and generate its BERT embeddings? (y/n): ")
    if get_user_confirmation.lower() == 'y':
        df = preprocess_dataset()
        generate_bert_embeddings(df)

    get_user_confirmation = input("\n\nDo you want to train an XGBoost model on a specific dataset? (y/n): ")
    if get_user_confirmation.lower() == 'y':
        train_xgboost_model()

    get_user_confirmation = input("\n\nDo you want to display the evaluation metrics of a specific system model? (y/n): ")
    if get_user_confirmation.lower() == 'y':
        display_evaluation_metrics()

    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = seconds % 1
    seconds = seconds - milliseconds

    print(f"\n\nTime elapsed: {str(int(hours)).zfill(2)}:{str(int(minutes)).zfill(2)}:{str(int(seconds)).zfill(2)}.{str(int(milliseconds*100)).zfill(2)}")

    print("\n\nExiting system...\n\n")
    
    time.sleep(1)