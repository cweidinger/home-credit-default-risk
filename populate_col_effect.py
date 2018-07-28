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
    df_full, cols_full = df_cols_for_tree(datasets.tree, nrows=ROWS)

with timer('seperate training from test'):  # 20s
    df_train = df_full[df_full['TARGET'].notnull()]

col_effect = {}

with timer('gen_col_effect'):  # 6388/3600 = 1.77 h with full records using
    gen_col_effect(df_train, cols_full[:10], col_effect)

dict_to_csv(col_effect, 'meta/col_effect_full_comma_again.csv')
