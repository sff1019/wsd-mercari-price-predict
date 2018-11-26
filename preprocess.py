import csv
import sys
import re
import time
import pickle
from itertools import chain

import numpy as np


if __name__ == '__main__':

    path = 'mercari/test_stg2.tsv'
    # path = 'mercari/train.tsv'
    f = open(path, 'r', encoding='utf-8')
    tsv = csv.reader(f, delimiter='\t')
    next(tsv)

    with open('processed/ohe_item_condition_id.pickle', 'rb') as f:
        ohe_condition = pickle.load(f)

    with open('processed/le_item_condition_id.pickle', 'rb') as f:
        le_condition = pickle.load(f)

    with open('processed/word2vec_name.pickle', 'rb') as f:
        word2vec_name = pickle.load(f)

    with open('processed/word2vec_item_description.pickle', 'rb') as f:
        word2vec_item_description = pickle.load(f)

    with open('processed/word2vec_brand_name.pickle', 'rb') as f:
        word2vec_brand_name = pickle.load(f)

    with open('processed/word2vec_category_name.pickle', 'rb') as f:
        word2vec_category_name = pickle.load(f)

    print('Reading...')
    sys.stdout.flush()
    rows = []
    start_time = time.time()
    count = 0
    for index, line in enumerate(tsv):
        condition = le_condition.transform([line[2]]).reshape(1, 1)
        np_condition = ohe_condition.transform(condition).flatten()

        name = [s for s in [s.strip().lower().replace('/', ' ').split(' ')
                            for s in re.split('\.', line[1])]]
        name = list(chain.from_iterable(name))
        np_name = word2vec_name.wv[name].mean(axis=0).flatten()

        # desc = [s for s in [s.strip().lower().replace('/', ' ').split(' ')
        #                     for s in re.split('\.', line[7])]]
        desc = [s for s in [s.strip().lower().replace('/', ' ').split(' ')
                            for s in re.split('\.', line[6])]]
        desc = list(chain.from_iterable(desc))
        np_desc = word2vec_item_description.wv[desc].mean(axis=0).flatten()

        category = [s for s in [s.strip().lower().replace('/', ' ').split(' ')
                                for s in re.split('\.', line[3])]][0]

        if category[0] != '':
            category1 = category[0]
            category2 = category[1]
            category3 = category[2]
            np_category1 = word2vec_category_name.wv[category1].flatten()
            np_category2 = word2vec_category_name.wv[category2].flatten()
            np_category3 = word2vec_category_name.wv[category3].flatten()
        else:
            np_category1 = np.zeros(100)
            np_category2 = np.zeros(100)
            np_category3 = np.zeros(100)

        brand = [s for s in [s.strip().lower().replace('/', ' ').split(' ')
                             for s in re.split('\.', line[4])]]
        brand = list(chain.from_iterable(brand))
        np_brand = word2vec_brand_name.wv[brand].mean(axis=0).flatten()

        # np_price = np.array(float(line[5])).flatten()
        # np_shipping = np.array(int(line[6])).flatten()
        np_shipping = np.array(int(line[5])).flatten()

        row = np.array([
            # np_price,
            np_name,
            np_condition,
            np_category1,
            np_category2,
            np_category3,
            np_brand,
            np_shipping,
            np_desc,
        ])

        processed = np.hstack(row)
        rows.append(processed)

        if index % 10000 == 0:
            progress = 100 * index / 3500000 + 0.1
            # progress = 100 * index / 1400000 + 0.1
            elapsed_time = time.time() - start_time
            etc = 100 / progress * elapsed_time / 60
            print('Progress: %f [%%]' % progress)
            print('Elapsed Time: %d [s]' % elapsed_time)
            print('Estimated time to complete: %d [min]' % etc)
            sys.stdout.flush()

        if (index != 0 and index % 500000 == 0) or index == 3460724:
            # if (index != 0 and index % 500000 == 0) or index == 1482534:
            rows = np.array(rows)
            with open('models/preprocessed_test_%d.pickle' % count, 'wb') as f:
                print('DUMP TEST %d START' % count)
                pickle.dump(rows, f)
                print('Done!')
                sys.stdout.flush()

            # with open('models/preprocessed_%d.pickle' % count, 'wb') as f:
            #     print('DUMP TRAIN %d START' % count)
            #     pickle.dump(rows[:450000, :], f)
            #     print('Done!')
            #     sys.stdout.flush()
            #
            # with open('models/preprocessed_val_%d.pickle' % count, 'wb') as f:
            #     print('DUMP VAL %d START' % count)
            #     pickle.dump(rows[450000:, :], f)
            #     print('Done!')
            #     sys.stdout.flush()

            count += 1
            rows = []
