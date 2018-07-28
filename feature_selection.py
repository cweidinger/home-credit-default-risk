import csv
import os

from roc import *

ROWS = None # this means do all the datasets
# ROWS = 2000 # this means do just a tiny fraction of the datasets to test the execution of the files for errors


def gen_col_effect(df, cols, col_effect):
    for c in cols:
        if c not in col_effect:
            try:
                # print(c)
                col_effect[c] = roc_auc_for_lightgbm(df, [c], get_lightgbm_with_default_parameters, debug=True)
            except Exception as ex:
                if type(ex).__name__ == 'LightGBMError':
                    col_effect[c] = 0.4999
                else:
                    df[c] = df[c].astype(np.float32).replace([np.inf, -np.inf], 0)
                    col_effect[c] = roc_auc_for_lightgbm(df, [c], get_lightgbm_with_default_parameters, debug=False)


def build_up_columns_incrementally(df, col_effect, cols_so_far=None, best_score=0.5, cols_to_try=None, cols_to_skip=None):
    if not cols_so_far:
        cols_so_far = []
    if not cols_to_skip:
        cols_to_skip = []
    if not cols_to_try:
        cols_to_try = [k for k in sorted(col_effect, key=col_effect.get, reverse=True) if k not in cols_so_far and k not in cols_to_skip]
    not_good_enough_in_a_row = 0
    for c in cols_to_try:
        cols_test = list(cols_so_far)
        cols_test.append(c)
        print('trying', c)
        score = roc_auc_for_lightgbm(df, cols_test, debug=True)
        if score > best_score:
            not_good_enough_in_a_row = 0
            best_score = score
            print('added', c)
            cols_so_far.append(c)
        else:
            cols_to_skip.append(c)
            not_good_enough_in_a_row += 1
        print('cols_to_skip = ["{}"]'.format('", "'.join(cols_to_skip)))
        print('cols_so_far = ["{}"]'.format('", "'.join(cols_so_far)))
        if not_good_enough_in_a_row > 15:
            break
    return cols_so_far, best_score


def dict_to_csv(d, filename):
    now = strftime("%Y-%m-%dT%H-%M-%S")
    try:
        os.rename(filename, filename + "_" + now + '.csv')
    except:
        pass
    df = pd.DataFrame(data={"key": list(d.keys()), "value": list(d.values())})
    df = df.sort_values('value', ascending=False)
    df.to_csv(filename, index=False)
    return df


def dict_from_csv(filename):
    reader = csv.DictReader(open(filename))
    d = {}
    for row in reader:
        try:
            d[row['key']] = float(row['value'])
        except:
            pass
    return d


def print_composites_better_than_individuals(col_effect, operator):
    df_col_effect = pd.DataFrame(data={"key": list(col_effect.keys()), "value": list(col_effect.values())})
    contains_per = df_col_effect['key'].str.contains(operator)
    for index, row in df_col_effect[contains_per].iterrows():
        try:
            first, second = row['key'].split(operator)
            #         print(row['key'], first,second)
            first_value = df_col_effect[df_col_effect['key'] == first]['value'].values[0]
            second_value = df_col_effect[df_col_effect['key'] == second]['value'].values[0]
            if row['value'] > first_value and row['value'] > second_value:
                print(row['key'], row['value'], first_value, second_value)
        except:
            print(row['key'])


def print_dict(d):
    s = [(k, d[k]) for k in sorted(d, key=d.get, reverse=True)]
    for k, v in s:
        print(k, v)


def find_base(c):
    extras = [',AVG', ',STD', ',KURT', ',SUM', ',PRODUCT', ',DIST']
    possible_aggs = [',mean', ',median', ',prod', ',sum', ',std', ',var']
    min_max = [',nan_instead_of_min', ',0_instead_of_min', ',min_flag', ',nan_instead_of_max', ',0_instead_of_max', ',max_flag']
    for s in min_max + possible_aggs + extras:
        c = c.replace(s, "")
    return c


def get_best_variants(cols):
    bases_already_added = []
    out = []
    for c in cols:
        base = find_base(c)
        if base in bases_already_added:
            continue
        else:
            bases_already_added.append(base)
            out.append(c)
    print(len(out), 'chosen out of', len(cols))
    return out
