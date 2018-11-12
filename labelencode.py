import csv

from sklearn import preprocessing


def extrtact_brand(data):
    result = []
    for line in data:
        result.append(line[4])

    return result


def label_encode(data):
    labelencoder = preprocessing.LabelEncoder()

    return labelencoder.fit_transform(data)


def one_hot_encoder(data):
    ohe = preprocessing.OneHotEncoder(sparse=False)

    # Reshape to 2D array
    data = data.reshape(len(data), 1)

    return ohe.fit_transform(data)


if __name__ == '__main__':
    tsv_file = open('mercari/train.tsv', 'r')
    tsv = csv.reader(tsv_file, delimiter='\t')

    brand_names = extrtact_brand(tsv)
    brand_lables = label_encode(brand_names)
    print(one_hot_encoder(brand_lables))
