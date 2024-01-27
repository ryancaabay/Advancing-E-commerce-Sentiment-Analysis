import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
import numpy as np
from xgboost import plot_importance

df = pd.read_csv('reviews_preprocessed.csv')

cols_to_drop = ['review']

df = df.drop(columns=cols_to_drop)

X = df.drop(columns=['sentiment'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=8)

estimators = [('encoder', TargetEncoder()), 
              ('classifier', XGBClassifier(random_state=8))]

pipe = Pipeline(estimators)

Pipeline(steps=[('encoder', TargetEncoder()), 
                ('classifier', 
                 XGBClassifier(base_score=None, booster=None,
                               colsample_bylevel=None, colsample_bynode=None,
                               colsample_bytree=None, enable_categorical=False,
                               gamma=None, gpu_id=None, importance_type=None,
                               interaction_constraints=None, learning_rate=None,
                               max_delta_step=None, max_depth=None,
                               min_child_weight=None, missing=np.nan,
                               monotone_constraints=None, n_estimators=100,
                               n_jobs=None, num_parallel_tree=None,
                               predictor=None, random_state=8, reg_alpha=None,
                               reg_lambda=None, scale_pos_weight=None,
                               subsample=None, tree_method=None,
                               validate_parameters=None, verbosity=None))])

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

search_space = {
    'classifier__max_depth': Integer(2, 8),
    'classifier__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'classifier_subsample': Real(0.5, 1.0),
    'classifier_colsample_bytree': Real(0.5, 1.0),
    'classifier_colsample_bylevel': Real(0.5, 1.0),
    'classifier_colsample_bynode': Real(0.5, 1.0),
    'classifier_reg_alpha': Real(0.0, 10.0),
    'classifier_reg_lambda': Real(0.0, 10.0),
    'classifier__gamma': Real(0.0, 10.0)
}

opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=10, scoring='roc_auc', random_state=8)

opt.fit(X_train, y_train)

opt.best_estimator_
opt.best_score_
opt.score(X_test, y_test)
opt.predict(X_test)
opt.predict_proba(X_test)

opt.best_estimator_.steps

xgboost_step = opt.best_estimator_.steps[1]
xgboost_model = xgboost_step[1]
plot_importance(xgboost_model)