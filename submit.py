import csv

from pickle_load import pickle_load


if __name__ == '__main__':
    model = pickle_load('model/test_rfr.pickle')

    for i in range(0, 7):
        dataset = pickle_load(f'processed/preprocessed_test_{i}.pickle')

        print('Predicting...')
        pred = model.predict(dataset)
        print('Predicted!!')

        with open('submit.csv', 'a') as f:
            print('Writing....')
            output = csv.writer(f, lineterminator='\n')
            csv.writerows(pred, f)
            print('Done!')
