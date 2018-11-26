import sys
import pickle

from sklearn import preprocessing
import pandas as pd


def read_data():
    types_dict_train = {'item_condition_id': 'int8'}
    types_dict_test = {'item_condition_id': 'int8'}
    input_train = pd.read_csv(
        'mercari/train.tsv', sep='\t', dtype=types_dict_train)
    input_test = pd.read_csv(
        'mercari/test_stg2.tsv', sep='\t', dtype=types_dict_test)
    print('READ INPUT...')
    return input_train, input_test


def label_encoder(data):
    le = preprocessing.LabelEncoder()

    return le.fit(data)


def one_hot_encoder(data):
    ohe = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)

    # Reshape to 2D array
    data = data.reshape(len(data), 1)

    # return ohe.fit_transform(data)
    return ohe.fit(data)


if __name__ == '__main__':
    args = sys.argv
    column = int(args[1])
    column_name = args[2]

    input_train, input_test = read_data()

    # Concat data
    data = pd.concat([input_train, input_test], ignore_index=True, sort=False)
    # Fill missing data with nothing
    data = data.fillna('')
    data = data[column_name].unique().astype(str)

    print('Data stringified!')

    # Label encode data
    le = label_encoder(data)

    # One-hot encode data
    one_hot_result = one_hot_encoder(le.transform(data))

    print('Finished')

    # Put each data in pickle file
    with open(f'processed/ohe_{column_name}.pickle', 'wb') as f:
        pickle.dump(one_hot_result, f)

    with open(f'processed/le_{column_name}.pickle', 'wb') as f:
        pickle.dump(le, f)
