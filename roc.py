import gc
import time
from contextlib import contextmanager
from time import strftime

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict


@contextmanager
def timer(note):
    start = time.time()
    yield
    print("{} took {:.0f}s".format(note, time.time() - start))


def roc_auc_for_sklearn(df, cols, note=None, cv=2, rows=None, debug=False, model=None):
    start = time.time()
    if not note:
        note = ', '.join(cols)
    if rows:
        df = df.sample(rows)
    if not model:
        model = GradientBoostingRegressor()
    y = cross_val_predict(model, df[cols], df['TARGET'], cv=cv)
    end = time.time()
    roc_auc = roc_auc_score(df['TARGET'], y)
    if debug:
        print("{}-fold cross-validated roc_auc {:.6f} for {:d} cols, over {:d} samples  in {:.0f} sec - {}".format(
            cv,
            roc_auc,
            len(cols),
            len(df.index),
            end - start,
            note
        ))
    return roc_auc


def get_lightgbm_with_default_parameters(quick_using_defaults=False):
    return LGBMClassifier(
        num_iterations=100 if quick_using_defaults else 5000,  # num_iteration, num_tree, num_trees, num_round, num_rounds, num_boost_round, n_estimators
        learning_rate=0.1 if quick_using_defaults else 0.02,  # defaults to 0.1
    )


def get_lightgbm_with_tuned_hyperparameters(quick_using_defaults=False):
    return LGBMClassifier(
        # over fitting knobs
        lambda_l1=0.05,  # Default 0, reg_alpha
        lambda_l2=0.05,  # Defaults 0, reg_lambda
        min_gain_to_split=0.02,  # defaults to 0, min_split_gain
        feature_fraction=1 if quick_using_defaults else 0.95,  # Default 1, % of features used in random tree mode, aliases: colsample_bytree

        # model efficiency
        min_data_in_leaf=20,  # default 20,  min_data=min_data_per_leaf=min_child_samples, implicitly limit tree growth by data in leaves, 100s-1000s for large dataset
        max_depth=4,  # default -1 (none), first thing to drop with overfitting
        num_leaves=31,  # default 31 should be <= 2^max_depth, may cause overfitting

        # speed
        bagging_fraction=0.95,  # default 1, % rows to use each iteration (to speed up and reduce overfitting), subsample, sub_row, bagging
        bagging_freq=1,  # default 0, 0 disables it, perform bagging at every k iteration. aliases: subsample_freq,

        # better score
        num_iterations=100 if quick_using_defaults else 5000,  # num_iteration, num_tree, num_trees, num_round, num_rounds, num_boost_round, n_estimators
        learning_rate=0.1 if quick_using_defaults else 0.02,  # defaults to 0.1

        # deal with overfitting
        max_bin=255,  # default 255, max number of bins that feature values will be bucketed in,

        # basics
        boosting='gdbt',  # default gbdt, rf, random_forest, dart, goss - random forest, Dropouts meet Multiple Additive Regression Trees, Gradient-based One-Side Sampling
        tree_learner='serial',  # default serial. parallel: feature, data, voting -  https://lightgbm.readthedocs.io/en/latest/Parallel-Learning-Guide.html
        verbose=-1,  # verbosity produces lots more problems
        num_threads=4,  # real # of cores on my computer
        # categorical_feature=[0], # This produces auc of 0.5... https://github.com/Microsoft/LightGBM/issues/1096
    )


def roc_auc_for_lightgbm(df, cols, get_lightgbm, cv=2, target='TARGET', folds=None, rows=None, debug=False, note=None):
    start = time.time()
    if not note:
        note = ', '.join(cols)
    if rows:
        df = df.sample(rows)
    if not folds:
        folds = KFold(n_splits=cv, shuffle=True, random_state=0)

    predictions = np.zeros(df.shape[0])
    df_cols = df[cols]
    df_target = df[target]

    for _, (train_indexes, validity_indexes) in enumerate(folds.split(df_cols, df_target)):
        train_X = df_cols.iloc[train_indexes]
        train_y = df_target.iloc[train_indexes]
        validity_X = df_cols.iloc[validity_indexes]
        validity_y = df_target.iloc[validity_indexes]

        model = get_lightgbm(quick_using_defaults=True)

        model.fit(
            train_X,
            train_y,
            eval_metric='auc',
            verbose=False,
        )

        # predictions[validity_indexes] = model.predict(validity_X, num_iteration=100)[:, 1]
        predictions[validity_indexes] = model.predict_proba(validity_X, num_iteration=100)[:, 1]

        del model, train_X, train_y, validity_X, validity_y, train_indexes, validity_indexes
        gc.collect()

    roc_auc = roc_auc_score(df_target, predictions)
    if debug:
        print("{}-fold cross-validated roc_auc {:.6f} for {:d} cols, over {:d} samples in {:.0f} sec - {}".format(
            folds.n_splits,
            roc_auc,
            len(cols),
            df.shape[0],
            time.time() - start,
            note
        ))
    return roc_auc


def kaggle_train_to_submit(df, cols, get_lightgbm, cv=5, target='TARGET', folds=None, to_files=True):
    start = time.time()
    if not folds:
        folds = KFold(n_splits=cv, shuffle=True, random_state=0)

    df_train = df[df[target].notnull()]
    df_test = df[df[target].isnull()]

    predictions = np.zeros(df_train.shape[0])
    submission_predictions = np.zeros(df_test.shape[0])

    df_cols = df_train[cols]
    df_target = df_train[target]

    for fold_idx, (train_indexes, validity_indexes) in enumerate(folds.split(df_cols, df_target)):
        train_X = df_cols.iloc[train_indexes]
        train_y = df_target.iloc[train_indexes]
        validity_X = df_cols.iloc[validity_indexes]
        validity_y = df_target.iloc[validity_indexes]

        model = get_lightgbm()

        model.fit(
            train_X,
            train_y,
            eval_set=[(train_X, train_y), (validity_X, validity_y)],
            eval_metric='auc',
            verbose=100,
            early_stopping_rounds=200
        )

        # print(model.predict(validity_X, num_iteration=model.best_iteration_))
        # predictions[validity_indexes] = model.predict(validity_X, num_iteration=model.best_iteration_)
        predictions[validity_indexes] = model.predict_proba(validity_X, num_iteration=model.best_iteration_)[:, 1]

        if to_files:
            submission_predictions += model.predict_proba(df_test[cols], num_iteration=model.best_iteration_)[:, 1] / folds.n_splits

        del model, train_X, train_y, validity_X, validity_y, train_indexes, validity_indexes
        gc.collect()

    roc_auc = roc_auc_score(df_target, predictions)
    print("{:d}-fold cross-validated roc_auc {:.6f} for {:d} cols, over {:d} samples in {:.0f} sec".format(
        folds.n_splits,
        roc_auc,
        len(cols),
        df_train.shape[0],
        time.time() - start
    ))

    now = strftime("%Y-%m-%dT%H-%M-%S")
    submission_filename = 'submissions/{}-submission.csv'.format(now)
    if to_files:
        print('submissions/{}-output.txt'.format(now))
        print("https://www.kaggle.com/c/home-credit-default-risk/submit")
        print('kaggle competitions submit -c home-credit-default-risk -f {} -m "Try Try Again"'.format(submission_filename))
        df_test['TARGET'] = submission_predictions
        df_test[['SK_ID_CURR', 'TARGET']].to_csv(submission_filename, index=False)

    return df_train['TARGET'], predictions
