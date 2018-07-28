from os.path import basename
import pandas as pd

from feature_selection import *


def str_partially_in_list(string, strings):
    for s in strings:
        if string in s:
            return True
    return False


def should_make(col):
    try:
        return len(prod_cols) == 0 or str_partially_in_list(col, prod_cols)
    except:
        return True


prod_cols = []
assert should_make('a')
prod_cols = ['a', 'b', 'abd']
assert should_make('a')
assert should_make('b')
assert should_make('d')
assert should_make('c') == False
del prod_cols
assert should_make('a')


def tx(df):
    cols = []
    one_hot_cols = []
    df.replace([np.inf, -np.inf], np.nan)
    og_cols = list(df.columns)
    for c in df.columns:
        if c in ['TARGET', 'index'] or not should_make(c):
            continue
        elif df[c].dtype == 'int64':
            #             df[c].fillna(0)
            cols.append(c)
        elif df[c].dtype == 'float32':
            cols.append(c)
        elif df[c].dtype == 'float64':
            df[c] = df[c].astype(np.float32)  # .fillna(0)
            cols.append(c)
        elif df[c].dtype == 'object':
            one_hot_cols.append(c)
            # df[c + ',factorized'] = pd.factorize(df[c])[0]
            # cols.append(c + ',factorized')
        else:
            raise Exception('Unknown dtype for {}: {}'.format(c, df[c].dtype))
    out = pd.get_dummies(df, columns=one_hot_cols, dummy_na=True)
    dummy_columns = [c for c in out.columns if c not in og_cols]
    cols = cols + dummy_columns
    return out, cols


def min_max_flags_and_replacements(df, cols):
    sz = len(df.index)
    min_irregularity = 11  # 11
    max_irregularity = 5000000  # 98
    for col in list(cols):
        if df[col].dtype == 'object':
            continue
        uniques = df[col].unique()
        n_uniques = len(uniques)
        pct_at_boundary_if_uniform = 1 / n_uniques
        if n_uniques > 14:
            minimum = df[col].min()
            mins = sum(df[col] == minimum)
            pct_at_min = mins / sz
            irregularity = pct_at_min / pct_at_boundary_if_uniform
            if min_irregularity < irregularity < max_irregularity:
                df[col + ',min_flag'] = df[col].apply(lambda x: 1 if x == minimum else 0)
                cols.append(col + ',min_flag')
                df[col + ',0_instead_of_min'] = df[col].apply(lambda x: 0 if x == minimum else x)
                cols.append(col + ',0_instead_of_min')
                df[col + ',nan_instead_of_min'] = df[col].apply(lambda x: np.nan if x == minimum else x)
                cols.append(col + ',nan_instead_of_min')
                # print(col, 'min', minimum, pct_at_min, irregularity)
                # df.hist(column=col, bins=100)
                # plt.xlabel(col)
                # plt.ylabel('count')
                # plt.show()
            maximum = df[col].max()
            maxs = sum(df[col] == maximum)
            pct_at_max = maxs / sz
            irregularity = pct_at_max / pct_at_boundary_if_uniform
            if min_irregularity < irregularity < max_irregularity:
                df[col + ',max_flag'] = df[col].apply(lambda x: 1 if x == maximum else 0)
                cols.append(col + ',max_flag')
                df[col + ',0_instead_of_max'] = df[col].apply(lambda x: 0 if x == maximum else x)
                cols.append(col + ',0_instead_of_max')
                df[col + ',nan_instead_of_max'] = df[col].apply(lambda x: np.nan if x == maximum else x)
                cols.append(col + ',nan_instead_of_max')
                # print(col, 'max', maximum, pct_at_max, irregularity)
                # df.hist(column=col, bins=100)
                # plt.xlabel(col)
                # plt.ylabel('count')
                # plt.show()
    # looks for abnormally large amounts of points in between the min and max
    #         for u in uniques:
    #             us = sum(df[col] == u)
    #             if us / ct > 20* 1 / n_uniques and u != maximum and u != minimum:
    #                 pass
    #                 print(col, 'val skew ', u, us / ct / (1 / n_uniques))
    return df, cols


def get_top_cols(d, n=10):
    i = 0
    cols = []
    for c, v in list(d.items()):
        try:
            float(v)
        except:
            del d[c]
    for c in sorted(d, key=d.get, reverse=True):
        if i == n:
            break
        if not str_partially_in_list(c, cols) and ',instead_of' not in c and '_per_' not in c and '_minus' not in c:
            i += 1
            cols.append(c)
    return cols


def gen_pairs(cols):
    cols = list(cols)
    pairs = []
    for index, c in enumerate(cols):
        for j in range(index + 1, len(cols)):
            pairs.append((c, cols[j]))
    return pairs


# top_cols = get_top_cols(col_effect,n=50)
# for c in top_cols:
#     print(c, col_effect[c])
# gen_pairs(top_cols)


def strip_digits(s):
    return ''.join([i for i in s if not i.isdigit()])


def aggregate_column_families(df, cols):
    # for all cols differing by a number do some numerical aggregations
    cc = {}
    for c in list(cols):
        if not should_make(c):
            continue
        csd = strip_digits(c)
        if csd in cc:
            cc[csd].append(c)
        else:
            cc[csd] = [c]
    for csd in cc.keys():
        cls = cc[csd]
        if len(cls) > 1:
            # df[csd + ',PRODUCT'] = df[cls].product(axis=1)
            # cols.append(csd + ',PRODUCT')
            # df[csd + ',SUM'] = df[cls].sum(axis=1)
            # cols.append(csd + ',SUM')
            df[csd + ',AVG'] = df[cls].mean(axis=1)
            cols.append(csd + ',AVG')
            df[csd + ',STD'] = df[cls].std(axis=1)
            cols.append(csd + ',STD')
            df[csd + ',KURT'] = df[cls].kurtosis(axis=1)
            cols.append(csd + ',KURT')
            df[csd + ',DIST'] = np.sqrt(df[cls].apply(lambda x: x ** 2).sum(axis=1, skipna=False))
            cols.append(csd + ',DIST')
    return df, cols


def divide_things_that_are_important(df, cols):
    col_effect = dict_from_csv('meta/col_effect_full.csv')
    for col, c in gen_pairs(get_top_cols(col_effect, n=20)):
        if col not in cols or c not in cols:
            #             print(col, 'or', c, 'not in')
            continue
        nn = "-".join([c, 'per', col])
        #      col_effect_full_comma   print('should make', nn, should_make(nn))
        if should_make(nn):
            df[nn] = df[c] / df[col]
            cols.append(nn)
        # nn = "_".join([c, 'minus', col])
        #         print('should make', nn, should_make(nn))
        # if should_make(nn):
        #     df[nn] = df[c] - df[col]
        #     cols.append(nn)
    return df, cols


def all_tx(df):
    df, cols = tx(df)
    df, cols = aggregate_column_families(df, cols)
    df, cols = min_max_flags_and_replacements(df, cols)
    df, cols = divide_things_that_are_important(df, cols)
    return df, cols


def df_cols_for_apps(nrows=None):
    df = pd.read_csv('input/application_train.csv', nrows=nrows)
    df, cols = all_tx(df)
    return df, cols


def join_df_cols(og_df, og_cols, df, cols, foreign_key, prefix, nrows=None):
    aggregations = {}
    possible_aggs = ['mean', 'median', 'prod', 'sum', 'std', 'var']
    for col in cols:
        if col != foreign_key:
            aggs = [agg_type for agg_type in possible_aggs if should_make(','.join([col, agg_type]))]
            if len(aggs) != 0:
                aggregations[col] = aggs
    if len(aggregations) == 0:
        return og_df, og_cols
    #     print(prefix, foreign_key)
    #     print(df.columns)
    #     print(aggregations)
    df_agg = df.groupby(foreign_key).agg(aggregations)
    df_agg.columns = pd.Index([','.join([agg[0], agg[1]]) for agg in df_agg.columns.tolist()])
    df_agg[prefix + '|,count'] = df.groupby(foreign_key).size()
    og_df = og_df.join(df_agg, how='left', on=foreign_key)
    og_cols = og_cols + list(df_agg.columns)
    for c in og_df.columns:
        if og_df[c].dtype == 'float32' and og_df[c].isin([np.inf, -np.inf]).any():  # np.nan, LightGBM doesn't need these
            og_df[c] = og_df[c].replace([np.inf, -np.inf], 0.0)  # , np.nan
        elif og_df[c].dtype == 'float64' and og_df[c].isin([np.inf, -np.inf]).any():  # np.nan,
            og_df[c] = og_df[c].astype(np.float32).replace([np.inf, -np.inf], 0.0)  # , np.nan
    del df, df_agg
    gc.collect()
    return og_df, og_cols


def df_cols_for_tree(tree, submission=False, nrows=2000):
    df = pd.read_csv(tree['filename'], nrows=nrows)
    if 'train' in tree['filename'] and submission == True:
        test_filename = tree['filename'].replace('train', 'test')
        df_test = pd.read_csv(test_filename, nrows=nrows)
        df = df.append(df_test).reset_index()
        del df_test
        gc.collect()
    filename = basename(tree['filename'])
    # rename columns so they are uniquely identifiable after joins
    df.columns = pd.Index(['|'.join([filename, col]) if 'target' not in tree or tree['target'] != col else col for col in df.columns.tolist()])
    df, cols = all_tx(df)
    # assign right here if functions provided below.. for now -> otherwise use assign_tx in client code
    if filename in filename_to_tx:
        tree['tx'] = filename_to_tx[filename]
    if 'tx' in tree:
        df, cols = tree['tx'](df, cols)

    for child in tree['children']:
        df_child, cs_child = df_cols_for_tree(child, submission, nrows)
        child_filename = basename(child['filename'])
        df_child[child['id']] = df_child[child_filename + '|' + child['id']]
        df[child['id']] = df[filename + '|' + child['id']]
        df, cols = join_df_cols(df, cols, df_child, cs_child, child['id'], child_filename)
    return df, cols


# These functions allow the user of my library to create features manually using their domain knowledge
def app_tx(df, cols):
    return df, cols


def pos_tx(df, cols):
    return df, cols


def bureau_balance_tx(df, cols):
    return df, cols


def bureau_tx(df, cols):
    return df, cols


def cc_tx(df, cols):
    return df, cols


def installments_tx(df, cols):
    return df, cols


def previous_app_tx(df, cols):
    return df, cols


filename_to_tx = {
    'application_train.csv': app_tx,
    'bureau.csv': bureau_tx,
    'bureau_balance.csv': bureau_balance_tx,
    'credit_card_balance.csv': cc_tx,
    'installments_payments.csv': installments_tx,
    'POS_CASH_balance.csv': pos_tx,
    'previous_application.csv': previous_app_tx
}


def assign_tx(tree):
    filename = basename(tree['filename'])
    if filename in filename_to_tx:
        tree['tx'] = filename_to_tx[filename]
    for child in tree['children']:
        assign_tx(child)
