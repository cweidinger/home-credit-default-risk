import matplotlib.pyplot as plt

from feature_selection import *
from library.datasets import Datasets


def analyze_target(df):
    target_counts = df.TARGET.value_counts()
    print("Of the {:,d} samples in the training set, {:,d} had a target of 0 and {:,d} had a target of 1.".format(df.TARGET.count(), target_counts[0], target_counts[1]))
    print("About {:.2%} of the time, the target was 0.".format(target_counts[0] / df.TARGET.count()))


def print_some_statistics_on_nan_columns(df):
    sz = len(df.index)
    nan_percents = df.isna().sum() / sz * 100
    nan_percents = nan_percents.sort_values(ascending=False)
    nan_percents.plot()
    print(nan_percents)
    nan = pd.concat([pd.Series(df.dtypes).apply(lambda x: str(x)), nan_percents], axis=1)
    print(nan.groupby(0).agg({1: ['mean', 'median', 'count']}))


def prepare_column_definition_file_for_annotation():
    cs = pd.read_csv('meta/HomeCredit_columns_description.csv')
    cs = cs.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
    cs['Row'] = cs['Row'].apply(lambda x: 'SK_ID_BUREAU' if x == 'SK_BUREAU_ID' else x)
    cs['Table'] = cs['Table'].apply(lambda x: 'application_train.csv' if x == 'application_{train|test}.csv' else x)
    cs.drop(columns=['index'])
    cs.to_csv(annotated_column_definitions_result_filename, index=False)


def annotate_column_definition_file():
    datasets = Datasets(
        path="input/*.csv",
        target="TARGET",
        id_matcher="SK.*ID.*",
        exclude_paths=['input/application_test.csv', 'input/HomeCredit_columns_description.csv'],
    )
    datasets.accumulate_column_statistics(annotated_column_definitions_result_filename)


def show_missing_percentages_broken_out_by_number_of_columns_with_different_data_types(cs):
    x = 'Column Count'
    y = 'Missing Value Percentage'
    plt.title('{} vs {}'.format(y, x))
    plt.xlabel(x)
    cs[cs['dtype'] == 'object'].sort_values('nan_percent', ascending=False)['nan_percent'].rename('Categorical').plot(kind='line', use_index=False, legend=True)
    cs[cs['dtype'] == 'float64'].sort_values('nan_percent', ascending=False)['nan_percent'].rename('Float').plot(kind='line', use_index=False, legend=True)
    cs[cs['dtype'] == 'int64'].sort_values('nan_percent', ascending=False)['nan_percent'].rename('Ints').plot(kind='line', use_index=False, legend=True)
    plt.ylabel(y)
    plt.show()


def show_missing_percentages_broken_out_by_number_of_columns_in_different_files(cs):
    x = 'Column Count'
    y = 'Missing Value Percentage'
    plt.title('{} vs {}'.format(y, x))
    plt.xlabel(x)
    for file in cs['Table'].unique():
        cs[cs['Table'] == file].sort_values('nan_percent', ascending=False)['nan_percent'].rename(file).plot(kind='line', use_index=False, legend=True)
    plt.ylabel(y)
    plt.show()


def show_auc_by_missing_value_percentages_for_int_and_floats(cs):
    x = 'Nan Percentage'
    y = 'AUC of the ROC'
    cs[cs['dtype'] == 'int64'].sort_values('auc', ascending=False).plot(x='nan_percent', y='auc', kind='line', use_index=False, legend=True)
    plt.title('{} vs {} for int64 variables'.format(y, x))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    cs[cs['dtype'] == 'float64'].sort_values('auc', ascending=False).plot(x='nan_percent', y='auc', kind='line', use_index=False, legend=True)
    plt.title('{} vs {} for float64 variables'.format(y, x))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


df = pd.read_csv('input/application_train.csv')
analyze_target(df)

annotated_column_definitions_result_filename = 'annotated_column_definitions_result.csv'

if not os.path.isfile(annotated_column_definitions_result_filename):
    prepare_column_definition_file_for_annotation()
    annotate_column_definition_file()

cs = pd.read_csv(annotated_column_definitions_result_filename)
show_missing_percentages_broken_out_by_number_of_columns_in_different_files(cs)

