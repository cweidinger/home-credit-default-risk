import random

import matplotlib.pyplot as plt

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

with timer('train and submit'):
    target_benchmark, benchmark_predictions = kaggle_train_to_submit(df_full, benchmark_cols, get_lightgbm_with_default_parameters, to_files=False)

# My Best Model

col_effect = dict_from_csv('meta/col_effect_full_comma.csv')
best_cols = [c for c, r in col_effect.items() if r > 0.5 and c in df_full.columns]
best_variants = get_best_variants(best_cols)

with timer('train and submit'):
    target, predictions = kaggle_train_to_submit(df_full, best_variants, get_lightgbm_with_tuned_hyperparameters, to_files=False)

# Calculating Net Dollars per application over cut points

loss_given_default = 0.53  # Historical Loss Given Default is 53% http://www.spcapitaliq-credit.com/how-safe-is-your-bank/


def calc_net_on_good_loan(loan_amount):
    opportunity_cost_rate = 0.017  # I made this up
    inflation_rate = 0.029
    yearly_interest_rate = 0.05 - inflation_rate - opportunity_cost_rate
    monthly_interest_rate = yearly_interest_rate / 12
    n_months_in_term = 12 * 30
    factor = (1 + monthly_interest_rate) ** n_months_in_term
    monthly_payment = loan_amount * monthly_interest_rate * factor / (factor - 1)
    total_payment = monthly_payment * n_months_in_term
    interest_collected = total_payment - loan_amount
    return interest_collected


opportunity_cost_rate = 0.017  # I made this up
inflation_rate = 0.029
yearly_interest_rate = 0.05 - inflation_rate - opportunity_cost_rate
monthly_interest_rate = yearly_interest_rate / 12
n_months_in_term = 12 * 30
factor = (1 + monthly_interest_rate) ** n_months_in_term
monthly_payment_per_loan_amount = monthly_interest_rate * factor / (factor - 1)
interest_collected_per_loan_amount = monthly_payment_per_loan_amount * n_months_in_term - 1


def calc_net_on_good_loan_faster(loan_amount):
    return loan_amount * interest_collected_per_loan_amount


def nets_for_cut_points(target, pred, points=30, maximum=0.6):
    nets = []
    cut_points = np.linspace(0, maximum, points)
    credit = df_full['application_train.csv|AMT_CREDIT']
    for cut_point in cut_points:
        net_for_cut_point = 0
        for i in range(len(target)):
            loan_amount = credit[i]
            if pred[i] < cut_point:  # predicted no default
                if target[i] == 1:  # loan defaulted
                    net_for_cut_point -= loss_given_default * loan_amount
                else:
                    net_for_cut_point += interest_collected_per_loan_amount * loan_amount
        nets.append(net_for_cut_point)
    return cut_points, nets


random_predictions = target.apply(lambda x: random.uniform(0, 1))

with timer('bench'):
    cut_points_benchmark, nets_benchmark = nets_for_cut_points(target, benchmark_predictions)

with timer('mine'):
    cut_points, nets = nets_for_cut_points(target, predictions)

with timer('perfect'):
    cut_points_perfect, nets_perfect = nets_for_cut_points(target, target)

with timer('mine'):
    cut_points_random, nets_random = nets_for_cut_points(target, random_predictions, points=10, maximum=1)

plt.close('all')
plt.clf()
benchmark = plt.scatter(cut_points_benchmark, np.asarray(nets_benchmark) / len(target), color='b')
mine = plt.scatter(cut_points, np.asarray(nets) / len(target), color='g')
perfect = plt.scatter(cut_points_perfect, np.asarray(nets_perfect) / len(target), color='y')
random_plt = plt.scatter(cut_points_random, np.asarray(nets_random) / len(target), color='y')
x = 'Default Risk Prediction Cut Point'
y = 'Net Dollars / Application'
plt.title('{} vs {}'.format(y, x))
plt.xlabel(x)
plt.ylabel(y)
plt.legend([benchmark, mine, perfect, random_plt], ['Benchmark', 'Best Variants', 'Perfect Prediction', 'Random Guessing'])
plt.show()

print("${:,.0f} per application is earned by using my best variants model over the benchmark".format(
    (max(nets) - max(nets_benchmark)) / len(target)))
print("${:,.0f} per application can be earned by making perfect predictions over using my best model".format(
    (max(nets_perfect) - max(nets)) / len(target)))
