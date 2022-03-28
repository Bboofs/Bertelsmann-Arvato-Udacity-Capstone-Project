import os
import time
# import pickle
import numpy as np
import pandas as pd
# import seaborn as sns
# import xgboost as xgb
# import matplotlib.pyplot as plt

from itertools import islice
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
#
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
#
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import GradientBoostingClassifier

# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split


# load in the data
azdias = pd.read_csv('../../data/Term2/capstone/arvato_data/Udacity_AZDIAS_052018.csv', sep=';')
customers = pd.read_csv('../../data/Term2/capstone/arvato_data/Udacity_CUSTOMERS_052018.csv', sep=';')

# load data descriptions
attributes_values = pd.read_excel('./data/DIAS Attributes - Values 2017.xlsx')
attributes_info = pd.read_excel('./data/DIAS Information Levels - Attributes 2017.xlsx')

cols_to_drop = ['ALTER_KIND4', 'ALTER_KIND3', 'TITEL_KZ', 'ALTER_KIND2', 'ALTER_KIND1', 'CAMEO_DEUG_2015',
                'KK_KUNDENTYP', 'KBA05_BAUMAX', 'EXTSEL992', 'KKK', 'REGIOTYP']


def get_common_cols():
    common_azdias = set(azdias.columns).intersection(set(attributes_values['Attribute'].unique()))
    len(common_azdias)

    common_customers = set(customers.columns).intersection(set(attributes_values['Attribute'].unique()))
    len(common_customers)

    common_cols = set(common_customers).intersection(set(common_azdias))
    len(common_cols)

    return common_cols


def take(n, iterable):
    """
    Return first n items of the iterable as a list
    """
    return list(islice(iterable, n))


def get_col_len(df):
    """
    input: df - a dataframe
    ouput: recommended col size

    the output is used to determine the value of column when aligning columns in the next cells
    """
    # an empty dictionary to store length of columns
    col_len = {}

    for col in df.columns:
        col_len[col] = len(col)

    # sort the columns by length
    sorted_col_len = {k: v for k, v in sorted(col_len.items(), key=lambda item: item[1], reverse=True)}

    # print top 5
    n_items = take(1, sorted_col_len.items())
    return int(round(n_items[0][1], -1))


def print_nulls(df, descending=True, top=15):
    """
    Loads a dataframe and lists the column names, null count and cardilaity

    Args:
        df (DataFrame): dataframe to sift through it's columns
        descending (bool): whether to sort the columns in descending/ascending order. default sort is descending

    Returns:
        None
        :param top:
    """

    if descending:
        print('Showing top {} columns:\n'.format(top))
    else:
        print('Showing bottom {} columns:\n'.format(top))

    counter = 0
    h_card = ''  # high cardinality columns

    # sort the columns
    null_counts = {k: v for k, v in sorted(df.isnull().sum().items(), key=lambda item: item[1], reverse=descending)}

    print('{0:>30}{1:>15}{2:>20}{3:>20}{4:>12}'.format('Column Name', 'Null Count', 'Null Count (%)', 'Cardinality',
                                                       'Cardinality'))
    print('{0:>30}{1:>15}{2:>20}{3:>20}{4:>12}'.format('_' * 30, '_' * 20, '_' * 20, '_' * 20, '_' * 12))

    for k, v in null_counts.items():
        v_perc = v * 100 / df.shape[0]
        cardinality = df[k].nunique() * 100 / df.shape[0]
        if cardinality > 50:
            h_card = 'High'

        print('{0:>30}{1:>15}{2:>18.5f} %{3:>18.5f} %{4:>12}'.format(k, v, v_perc, cardinality, h_card))
        h_card = ''
        counter += 1
        if counter <= top:
            continue
        else:
            break


def remove_invalid_values(df, possible_values):
    """
    Function to replace non-valid values with nan values

    input:
        - df (The full dataframe with all the data)
        - common_col (A list of common columns in all the three dataframes)
    """

    for col, val in possible_values.items():
        print('.', end='')
        df[col] = df[col].apply(lambda x: x if x in val else np.nan)

    return df


def remove_unknowns(df_with_unknowns, unknown_attr_df):
    """
    Function to replace unkown values with nans

    input:
        - df_with_unknowns (The full dataframe with all the data)
        - unknown_attr_df (A smaller dataframe with values of unknown entries)
    """

    for col in df_with_unknowns.columns:
        if col in unknown_attr_df['Attribute'].values:
            unknown_val = unknown_attr_df[unknown_attr_df['Attribute'] == col]['Value'].values[0]
            # print('unknown_val:', unknown_val)
            df_with_unknowns[col] = df_with_unknowns[col].apply(lambda x: np.nan if str(x) in str(unknown_val) else x)
            # print('{0:<20}{1:<30}'.format('CONVERTED', col))
            print('.', end='')
        else:
            # print('{0:<20}{1:<30}'.format('NOT CONVERTED', col))
            pass

    return df_with_unknowns


def drop_cols(df, cols=cols_to_drop):
    return df.drop(columns=cols, inplace=True)


def create_col_maps(df):
    """
    Function to replace 0s with nan values columns,
    ['LP_FAMILIE_FEIN', 'LP_FAMILIE_GROB', 'LP_LEBENSPHASE_FEIN',
           'LP_LEBENSPHASE_GROB', 'LP_STATUS_FEIN', 'LP_STATUS_GROB']
    """
    # LP_FAMILIE_GROB
    convert_1 = {1: 'single', 2: 'couple', 3: 'singleparent', 4: 'singleparent', 5: 'singleparent',
                 6: 'family', 7: 'family', 8: 'family', 9: 'multihousehold', 10: 'multihousehold', 11: 'multihousehold'}
    convert_2 = {'single': 0, 'couple': 1, 'singleparent': 2, 'family': 3, 'multihousehold': 4}

    df['LP_FAMILIE_GROB'] = df['LP_FAMILIE_GROB'].replace(convert_1)
    df['LP_FAMILIE_GROB'] = df['LP_FAMILIE_GROB'].replace(convert_2)

    # LP_STATUS_GROB
    convert_1 = {1: 'lowincome', 2: 'lowincome', 3: 'avgincome', 4: 'avgincome', 5: 'avgincome',
                 6: 'independant', 7: 'independant', 8: 'houseowner', 9: 'houseowner', 10: 'topearner'}
    convert_2 = {'lowincome': 0, 'avgincome': 1, 'independant': 2, 'houseowner': 3, 'topearner': 4}

    df['LP_STATUS_GROB'] = df['LP_STATUS_GROB'].replace(convert_1)
    df['LP_STATUS_GROB'] = df['LP_STATUS_GROB'].replace(convert_2)

    # LP_LEBENSPHASE_FEIN
    life_stages = {1: 'younger_age', 2: 'middle_age', 3: 'younger_age',
                   4: 'middle_age', 5: 'advanced_age', 6: 'retirement_age',
                   7: 'advanced_age', 8: 'retirement_age', 9: 'middle_age',
                   10: 'middle_age', 11: 'advanced_age', 12: 'retirement_age',
                   13: 'advanced_age', 14: 'younger_age', 15: 'advanced_age',
                   16: 'advanced_age', 17: 'middle_age', 18: 'younger_age',
                   19: 'advanced_age', 20: 'advanced_age', 21: 'middle_age',
                   22: 'middle_age', 23: 'middle_age', 24: 'middle_age',
                   25: 'middle_age', 26: 'middle_age', 27: 'middle_age',
                   28: 'middle_age', 29: 'younger_age', 30: 'younger_age',
                   31: 'advanced_age', 32: 'advanced_age', 33: 'younger_age',
                   34: 'younger_age', 35: 'younger_age', 36: 'advanced_age',
                   37: 'advanced_age', 38: 'retirement_age', 39: 'middle_age',
                   40: 'retirement_age'}

    wealth_scale = {1: 'low', 2: 'low', 3: 'average', 4: 'average', 5: 'low', 6: 'low',
                    7: 'average', 8: 'average', 9: 'average', 10: 'wealthy', 11: 'average',
                    12: 'average', 13: 'top', 14: 'average', 15: 'low', 16: 'average',
                    17: 'average', 18: 'wealthy', 19: 'wealthy', 20: 'top', 21: 'low',
                    22: 'average', 23: 'wealthy', 24: 'low', 25: 'average', 26: 'average',
                    27: 'average', 28: 'top', 29: 'low', 30: 'average', 31: 'low',
                    32: 'average', 33: 'average', 34: 'average', 35: 'top', 36: 'average',
                    37: 'average', 38: 'average', 39: 'top', 40: 'top'}

    df['Temp'] = df['LP_LEBENSPHASE_FEIN']

    df['LP_LEBENSPHASE_FEIN'] = df['LP_LEBENSPHASE_FEIN'].replace(life_stages)
    df['LP_LEBENSPHASE_GROB'] = df['Temp'].replace(wealth_scale)

    life_stages = {'younger_age': 1, 'middle_age': 2, 'advanced_age': 3,
                   'retirement_age': 4}
    wealth_scale = {'low': 1, 'average': 2, 'wealthy': 3, 'top': 4}

    df['LP_LEBENSPHASE_FEIN'] = df['LP_LEBENSPHASE_FEIN'].replace(life_stages)
    df['LP_LEBENSPHASE_GROB'] = df['LP_LEBENSPHASE_GROB'].replace(wealth_scale)

    return df


def convert_date_cols(df):
    df['EINGEFUEGT_AM'] = pd.to_datetime(df['EINGEFUEGT_AM']).dt.year
    # df['EINGEFUEGT_AM'] = df['EINGEFUEGT_AM'].map(lambda x: x.year)

    return df


def remove_missing_rows(df, threshold, name=''):
    """
    Takes in a dataframe and drops rows with number of missing features
    as per given threshold.
    """
    total_rows = df.shape[0]

    tmp_df = df.dropna(thresh=df.shape[1] - threshold)

    df = tmp_df.copy()
    tmp_df = pd.DataFrame()

    removed_rows = total_rows - df.shape[0]

    print(f'\tRemoved {removed_rows} rows from {name} dataframe')

    # Reset index
    df = df.reset_index()
    del df['index']

    return df


def encode_ost_west_col(df):
    """
    Function to label encode the feature "OST_WEST_KZ"
    """

    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].replace({'W': 0, 'O': 1})

    return df


def encode_anrede_col(df):
    """
    Function to label encode the feature "ANREDE_KZ"
    """

    df['ANREDE_KZ'] = df['ANREDE_KZ'].replace({1: 0, 2: 1})

    return df


def encode_cameo_intl_col(df):
    df['CAMEO_INTL_2015_WEALTH'] = df['CAMEO_INTL_2015'].apply(
        lambda x: np.floor_divide(float(x), 10) if float(x) else np.nan)
    df['CAMEO_INTL_2015_FAMILY'] = df['CAMEO_INTL_2015'].apply(lambda x: np.mod(float(x), 10) if float(x) else np.nan)

    tmp_df = df.drop('CAMEO_INTL_2015', axis=1, inplace=False)
    return tmp_df


def clean_wohnlage_col(df):
    """
    Function to remove '0' from "WOHNLAGE"
    """

    df['WOHNLAGE'] = df['WOHNLAGE'].replace({0: np.nan})

    return df


def load_all_data(clean=False, scaled=False):
    """
    Loads azdias, customer, attribute values and attribute information data

    Args:
        clean (bool): loads cleaned data, if set to True and cleaned data exists.
                Default: False

    Returns:
        azdias : Azdias DataFrame
        customers : Customers DataFrame
        attribute_values : Attribute values DataFrame
        attribute_info : Attribute information DataFrame
        :param scaled:
    """

    azdias_src = '../../data/Term2/capstone/arvato_data/Udacity_AZDIAS_052018.csv'
    customers_src = '../../data/Term2/capstone/arvato_data/Udacity_CUSTOMERS_052018.csv'

    # Ignore any empty columns while loading data
    no_unamed = [lambda x: 'Unnamed' not in x]

    # Check if cleaned data exists
    clean_exists = False
    if clean:
        clean_exists = os.path.isfile("./data/Customers_cleaned.pickle.bz2") & \
                       os.path.isfile("./data/Azdias_cleaned.pickle.bz2")
        if clean_exists:
            print("Cleaned Data Exists")
        else:
            print("Cleaned Data does not Exist!")

    # Check if scaled data exists
    scaled_exists = False
    if scaled:
        scaled_exists = clean_exists = os.path.isfile("./data/scaled_customer.pickle.bz2") & os.path.isfile(
            "./data/scaled_azdias.pickle.bz2")
        if scaled_exists:
            print("Scaled Data Exists")
        else:
            print("Scaled Data does not Exist!")

    # azdias = None
    # customers = None
    customers_additional_dup = None
    # attributes_values = None
    # attributes_info = None

    if scaled and scaled_exists:
        print("Loading Scaled Azdias and Customers Data")
        azdias_dup = pd.read_pickle('./data/02scaled_azdias.pickle.bz2')
        customers_dup = pd.read_pickle('./data/02scaled_customer.pickle.bz2')
        customers_additional_dup = pd.read_pickle('./data/02Customer_Additional_cleaned.pickle.bz2')

    else:
        if clean_exists:
            print("Loading Cleaned Azdias and Customers Data")
            azdias_dup = pd.read_pickle('./data/02Azdias_cleaned.pickle.bz2')
            customers_dup = pd.read_pickle('./data/02Customers_cleaned.pickle.bz2')
            customers_additional_dup = pd.read_pickle('./data/02Customer_Additional_cleaned.pickle.bz2')
        else:
            print("Loading Raw Azdias and Customers Data")
            azdias_dup = pd.read_csv(azdias_src, sep=';')
            customers_dup = pd.read_csv(customers_src, sep=';')

    attributes_values_dup = pd.read_excel('./data/DIAS Attributes - Values 2017.xlsx')
    attributes_info_dup = pd.read_excel('./data/DIAS Information Levels - Attributes 2017.xlsx')

    attributes_values_dup.drop(columns=['Unnamed: 0'], inplace=True)
    attributes_info_dup.drop(columns=['Unnamed: 0'], inplace=True)

    ### Save up on memory
    # azdias_dup = pd.DataFrame()
    # customers_dup = pd.DataFrame()
    # attributes_values_dup = pd.DataFrame()
    # attributes_info_dup = pd.DataFrame()

    return azdias_dup, customers_dup, customers_additional_dup, attributes_values_dup, attributes_info_dup


def clean_data(azdias, customers, attributes_values, column_miss_perc=30, row_miss_count=50):
    print("\n\nCleaning Given Dataframes")

    start = time.time()

    cleaning_info = {}

    # forward fill all the rowname values in the atributes table
    attributes_values['Attribute'] = attributes_values['Attribute'].ffill()

    # create a dictionary of possible/allowed values from the attributes table
    possible_values = {}
    col_cnt = 0

    print('\n\nPrinting raw values (all columns):')
    common_cols = get_common_cols()
    for col in common_cols:
        col_cnt += 1
        possible_values[col] = list(attributes_values[attributes_values['Attribute'] == col]['Value'].values)
        print('.', end='')  # just to show some activity is going on...
        # print('{0:>3}. {1}: {2}'.format(col_cnt, col, possible_values[col]))
        # if col_cnt % 50 == 0:
        #    print('-'*80)

    # Remove possible values from the attributes with obvious invalid values (in this case, with only one element)
    print(
        '\n\nRemove possible values from the attributes with obvious invalid values (in this case, with only one element)')
    for key in list(possible_values):
        if len(possible_values[key]) < 2:
            # print(key, possible_values[key])
            del possible_values[key]

    # print(len(possible_values))

    # rectify the column values and list them
    col_cnt = 0

    print('\n\nPrinting columns with error values only:')
    for col, v in possible_values.items():
        if '-1, 9' in possible_values[col]:
            possible_values[col].remove('-1, 9')
            if -1 not in possible_values[col]:
                possible_values[col].insert(0, -1)
            if 9 not in possible_values[col]:
                possible_values[col].insert(1, 9)
            col_cnt += 1
            # print('{0:>3}. {1}: {2}'.format(col_cnt, col, possible_values[col]))
            print('.', end='')  # just to show some activity is going on...
            # if col_cnt % 50 == 0:
            #    print('-'*80)

    print('\n\nRemoving invalid values from customers df')
    customers = remove_invalid_values(customers, possible_values)

    print('\n\nRemoving invalid values from azdias df')
    azdias = remove_invalid_values(azdias, possible_values)

    print('\n\nRemoving unknown values from azdias df')
    unknown = attributes_values[attributes_values['Meaning'] == 'unknown'][['Attribute', 'Value']]

    azdias = remove_unknowns(azdias, unknown)

    print('\n\nRemoving unknown values from customers df')
    customers = remove_unknowns(customers, unknown)

    cols_to_drop = ['ALTER_KIND4', 'ALTER_KIND3', 'TITEL_KZ', 'ALTER_KIND2', 'ALTER_KIND1', 'CAMEO_DEUG_2015',
                    'KK_KUNDENTYP', 'KBA05_BAUMAX', 'EXTSEL992', 'KKK', 'REGIOTYP']

    # current shapes of data
    print('\n\nCurrent shapes of data\n\tazdias:{}\n\tcustomers:{}\n\n'.format(azdias.shape, customers.shape))

    # drop_cols(azdias)
    # drop_cols(customers)

    ###below 4 lines replaces above 2
    good_cols_azdias = [col for col in azdias.columns if col not in cols_to_drop]
    tmp_df = azdias[good_cols_azdias]
    azdias = tmp_df.copy()

    good_cols_customers = [col for col in customers.columns if col not in cols_to_drop]
    tmp_df = customers[good_cols_customers]
    customers = tmp_df.copy()

    tmp_df = pd.DataFrame()

    # new data shapes
    print('\n\nnew data shapes\n\tazdias:{}\n\tcustomers:{}'.format(azdias.shape, customers.shape))

    print('\n\nConverting date columns')
    customers = convert_date_cols(customers)
    azdias = convert_date_cols(azdias)

    print('\n\nNumber of rows before dropping')
    print(f'Azdias - {len(azdias)}')
    print(f'Customers - {len(customers)}')

    customers = remove_missing_rows(customers, threshold=50)
    azdias = remove_missing_rows(azdias, threshold=50)

    print('\n\nNumber of rows after dropping')
    print(f'Azdias - {len(azdias)}')
    print(f'Customers - {len(customers)}')

    remove_extra_cols = ['D19_LETZTER_KAUF_BRANCHE', 'CAMEO_DEU_2015', 'LP_FAMILIE_FEIN', 'LP_STATUS_FEIN',
                         'CAMEO_INTL_2015']

    print('\n\nDeleting more cols...', remove_extra_cols)

    azdias_cols = [col for col in azdias.columns if col not in remove_extra_cols]
    customers_cols = [col for col in customers.columns if col not in remove_extra_cols]

    # current shapes of data
    print('\n\nCurrent shapes of data\n\tazdias:{}\n\tcustomers:{}\n\n'.format(azdias.shape, customers.shape))

    # drop_cols(azdias, cols=remove_extra_cols)
    # drop_cols(customers, cols=remove_extra_cols)
    print('Working on azdias df')
    tmp_df = azdias[azdias_cols]
    azdias = tmp_df.copy()

    print('Working on customers')
    tmp_df = customers[customers_cols]
    customers = tmp_df.copy()

    print('cleaning up tmp df')
    tmp_df = pd.DataFrame()

    # new data shapes
    print('\n\nNew data shapes\n\tazdias:{}\n\tcustomers:{}'.format(azdias.shape, customers.shape))

    print('Encode azdias columns')
    azdias = encode_ost_west_col(azdias)
    azdias = encode_anrede_col(azdias)
    # azdias = encode_cameo_intl_col(azdias)

    print('Encode customers columns')
    customers = encode_ost_west_col(customers)
    customers = encode_anrede_col(customers)
    # customers = encode_cameo_intl_col(customers)

    print('cleaning WOHNLAGE')
    azdias = clean_wohnlage_col(azdias)
    customers = clean_wohnlage_col(customers)

    print('Take out the extra columns present in customers dataset as a separate dataframe')
    extra_cols_in_customers = list(set(customers.columns) - set(azdias.columns))
    customer_extra_cols = customers[extra_cols_in_customers]
    tmp_df = customers.drop(columns=extra_cols_in_customers, inplace=False)
    customers = tmp_df.copy()
    tmp_df = pd.DataFrame()

    print(
        f'azdias dataframe shape:\n\t{azdias.shape}\ncustomers dataframe shape:\n\t{customers.shape}\ncustomer_extra_cols dataframe shape:\n\t{customer_extra_cols.shape}')

    print('Imputing data')
    imputer = SimpleImputer(strategy="most_frequent")

    azdias = pd.DataFrame(imputer.fit_transform(azdias), columns=azdias.columns)
    customers = pd.DataFrame(imputer.transform(customers), columns=customers.columns)

    end = time.time()

    print(f"Completed Cleaning in {end - start} seconds")

    return azdias, customers, customer_extra_cols, cleaning_info


def clean_data_for_supervised(cleaned_azdias, df, attributes_values):
    print("\n\nCleaning Given Dataframes")

    start = time.time()

    cleaning_info = {}

    # forward fill all the rowname values in the atributes table
    attributes_values['Attribute'] = attributes_values['Attribute'].ffill()

    tmp_df = df[cleaned_azdias.columns].copy()

    print('cleaned_azdias column length:', len(cleaned_azdias.columns))
    print('Orginal test column length:', len(df.columns))
    print('Current test column length:', len(tmp_df.columns))

    common_cols = set(tmp_df.columns).intersection(set(attributes_values['Attribute'].unique()))

    # create a dictionary of possible/allowed values from the attributes table
    possible_values = {}
    col_cnt = 0

    print('\n\nPrinting raw values (all columns):')
    for col in common_cols:
        col_cnt += 1
        possible_values[col] = list(attributes_values[attributes_values['Attribute'] == col]['Value'].values)
        print('.', end='')  # just to show some activity is going on...
        # print('{0:>3}. {1}: {2}'.format(col_cnt, col, possible_values[col]))
        # if col_cnt % 50 == 0:
        #    print('-'*80)

    # Remove possible values from the attributes with obvious invalid values (in this case, with only one element)
    print(
        '\n\nRemove possible values from the attributes with obvious invalid values (in this case, with only one element)')
    for key in list(possible_values):
        if len(possible_values[key]) < 2:
            # print(key, possible_values[key])
            del possible_values[key]

    # print(len(possible_values))

    # rectify the column values and list them
    col_cnt = 0

    print('\n\nPrinting columns with error values only:')
    for col, v in possible_values.items():
        if '-1, 9' in possible_values[col]:
            possible_values[col].remove('-1, 9')
            if -1 not in possible_values[col]:
                possible_values[col].insert(0, -1)
            if 9 not in possible_values[col]:
                possible_values[col].insert(1, 9)
            col_cnt += 1
            # print('{0:>3}. {1}: {2}'.format(col_cnt, col, possible_values[col]))
            print('.', end='')  # just to show some activity is going on...
            # if col_cnt % 50 == 0:
            #    print('-'*80)

    print('\n\nRemoving invalid values from current test df')
    tmp_df = remove_invalid_values(tmp_df, possible_values)

    print('\n\nRemoving unknown values from current test df')
    unknown = attributes_values[attributes_values['Meaning'] == 'unknown'][['Attribute', 'Value']]

    tmp_df = remove_unknowns(tmp_df, unknown)

    print('\n\nConverting date columns')
    tmp_df = convert_date_cols(tmp_df)

    print('Encode current test columns')
    tmp_df = encode_ost_west_col(tmp_df)
    tmp_df = encode_anrede_col(tmp_df)
    # tmp_df = encode_cameo_intl_col(tmp_df)

    print('cleaning WOHNLAGE')
    tmp_df = clean_wohnlage_col(tmp_df)

    print(f'current test dataframe shape:\n\t{tmp_df.shape}')

    # Imputing Missing data
    print("\nImputing missing values with most frequent ones")
    imputer = SimpleImputer(strategy="most_frequent")

    imputer.fit(cleaned_azdias)
    tmp_df = pd.DataFrame(imputer.transform(tmp_df), columns=tmp_df.columns)

    end = time.time()

    print(f"Completed Cleaning in {end - start} seconds")

    return tmp_df
