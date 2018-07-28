from feature_selection import *
from library.datasets import Datasets
from transformations import df_cols_for_tree

datasets = Datasets(
    path="input/*.csv",
    target="TARGET",
    id_matcher="SK.*ID.*",
    exclude_paths=['input/application_test.csv', 'input/HomeCredit_columns_description.csv'],
)

with timer("load tree df and cols"):  # full load takes 32 minutes, 1591s = 26'
    df_full, cols_full = df_cols_for_tree(datasets.tree, nrows=32000)

col_effect = dict_from_csv('meta/col_effect_full_comma.csv')
best_cols = [c for c, r in col_effect.items() if r > 0.5 and c in df_full.columns]
best_variants = get_best_variants(best_cols)

X_train = df_full[best_variants]
y_train = df_full['TARGET']

import lightgbm
from sklearn.model_selection import GridSearchCV
import numpy as np

all_params = {
    # 1126s {'feature_fraction': 0.8, 'learning_rate': 0.015, 'min_data_in_leaf': 30, 'num_iterations': 300}
    'feature_fraction': np.linspace(0.7, 0.9, 3),
    'min_data_in_leaf': np.linspace(10, 30, 3, dtype=int),
    'learning_rate': np.linspace(0.005, 0.015, 3),  # 0.01

    'num_iterations': [300],  # This value was chosen by using early stopping since GridSearchCV can't be configured with early stopping

    # Using GridSearchCV, I established that the following parameters are best with the LGBMClassifier's defaults
    # 'num_leaves': np.linspace(28,34,3, dtype=int),
    # 'max_depth': np.linspace(3, 8, 3, dtype=int), # Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
    #     'lambda_l1': np.linspace(0,0.1,3),
    #     'lambda_l2': np.linspace(0,0.1,3),
    #     'min_gain_to_split': np.linspace(0,0.01,3),
    #     'bagging_fraction': np.linspace(0.9,1,3),
    #     'bagging_freq': np.linspace(0,4,3, dtype=int),
    #     'max_bin': np.linspace(200,300,3, dtype=int) #255
}

with timer("Hyperparameter tuning with GridSearchCV"):
    estimator = lightgbm.LGBMClassifier()
    print('Parameter space ', all_params)
    gbm = GridSearchCV(estimator, all_params, scoring='roc_auc')
    gbm.fit(X_train, y_train)
    print('Best Parameters found by GridSearchCV', gbm.best_params_)
