import pickle

import numpy as np
# from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


if __name__ == '__main__':

    with open('processed/preprocessed_0.pickle', 'rb') as f:
        train = pickle.load(f)

    with open('processed/preprocessed_val_0.pickle', 'rb') as f:
        val = pickle.load(f)

    # clf = linear_model.LinearRegression()
    rfr = RandomForestRegressor(
        n_jobs=-1, min_samples_leaf=5, n_estimators=200)

    # X = train[:2000, 1:]
    # Y = train[:2000, :1]
    X = train[:10000, 1:]
    Y = train[:10000, :1]

    print('Gonna FIT!!')

    # clf.fit(X, Y)
    rfr.fit(X, Y)

    print('Created!')

    # X_val = val[:100, 1:]
    # Y_val = val[:100, :1]
    X_val = val[:1000, 1:]
    Y_val = val[:1000, :1]

    pred = rfr.predict(X_val)
    # pred = clf.predict(X_val)

    with open('model/test_rfr.pickle', 'wb') as f:
        # pickle.dump(clf, f)
        pickle.dump(rfr, f)

    print('Dumped!')

    print(np.sqrt(mean_squared_error(Y_val, pred)))

