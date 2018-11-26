import re
import pickle

import pandas as pd
from gensim.models import word2vec

EMBEDDING_SIZE = 100
COLUMN_NAMES = [
    # 'name',
    'category_name',
    # 'brand_name',
    # 'item_description',
]


def read_data():
    input_train = pd.read_csv('mercari/train.tsv', sep='\t')
    input_test = pd.read_csv('mercari/test_stg2.tsv', sep='\t')
    print('READ INPUT...')
    return input_train, input_test


def extract_word_features(column_name, input_train, input_test):
    print('Start Reading...')
    # Strip sentences into words
    # If category_name, divide it into 3 words
    train_sentences_desc = [s for s in [s.strip().lower().replace('/', ' ').split(' ')
                                        for i in input_train[column_name].values for s in re.split('\.', str(i))]]
    test_sentences_desc = [s for s in [s.strip().lower().replace('/', ' ').split(' ')
                                       for i in input_test[column_name].values for s in re.split('\.', str(i))]]
    sentences_desc = train_sentences_desc + test_sentences_desc + ['']

    print('Start Extracting...')
    model = word2vec.Word2Vec(
        sentences_desc, size=EMBEDDING_SIZE, window=5, min_count=1, workers=50)

    with open('processed/word2vec_%s.pickle' % column_name, 'wb') as f:
        pickle.dump(model, f)

    print('Done')


if __name__ == '__main__':
    input_train, input_test = read_data()
    for column in COLUMN_NAMES:
        extract_word_features(column, input_train, input_test)
