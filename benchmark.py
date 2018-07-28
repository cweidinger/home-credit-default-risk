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
    df_full, cols_full = df_cols_for_tree(datasets.tree, nrows=None)

# Benchmark Model

df_full['DEBT_TO_INCOME'] = (df_full['bureau.csv|AMT_CREDIT_SUM_DEBT,sum'] + df_full['credit_card_balance.csv|AMT_BALANCE,sum']) / df_full['application_train.csv|AMT_INCOME_TOTAL']

benchmark_cols = [
    'DEBT_TO_INCOME',
    'application_train.csv|DAYS_EMPLOYED,max_flag',
    'application_train.csv|DAYS_EMPLOYED',
    'application_train.csv|EXT_SOURCE_,AVG',
    'application_train.csv|EXT_SOURCE_1',
    'application_train.csv|EXT_SOURCE_2',
    'application_train.csv|EXT_SOURCE_3',
    'application_train.csv|AMT_INCOME_TOTAL',
    'bureau.csv|AMT_CREDIT_SUM_DEBT,sum',
    'credit_card_balance.csv|AMT_BALANCE,sum'
]

with timer('Train benchmark model'):
    target, predictions = kaggle_train_to_submit(df_full, benchmark_cols, get_lightgbm_with_default_parameters, to_files=False)