from time import time
start_time = time()
t0 = time()
print "Importing magical pythons..."
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn import cross_validation, ensemble, tree, preprocessing, neighbors, naive_bayes
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
le = preprocessing.LabelEncoder()
import xgboost as xgb
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
print "Imported magical pythons:",round(time()-t0,3),"s"

t0 = time()
print "Mapk function..."
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
print "Mapking done:",round(time()-t0,3),"s"

ids = []
predictions = []

t0 = time()
print "Dtypes..."
train_dtypes = {'date_time': pd.np.object,
'site_name': pd.np.int64,
'posa_continent': pd.np.int64,
'user_location_country': pd.np.int64,
'user_location_region': pd.np.int64,
'user_location_city': pd.np.int64,
'orig_destination_distance': pd.np.float64,
'user_id': pd.np.int64,
'is_mobile': pd.np.int64,
'is_package': pd.np.int64,
'channel': pd.np.int64,
'srch_ci': pd.np.object,
'srch_co': pd.np.object,
'srch_adults_cnt': pd.np.int64,
'srch_children_cnt': pd.np.int64,
'srch_rm_cnt': pd.np.int64,
'srch_destination_id': pd.np.int64,
'srch_destination_type_id': pd.np.int64,
'is_booking': pd.np.int64,
'cnt': pd.np.int64,
'hotel_continent': pd.np.int64,
'hotel_country': pd.np.int64,
'hotel_market': pd.np.int64,
'hotel_cluster': pd.np.int64}

test_dtypes = {'id': pd.np.int64,
'date_time': pd.np.object,
'site_name': pd.np.int64,
'posa_continent': pd.np.int64,
'user_location_country': pd.np.int64,
'user_location_region': pd.np.int64,
'user_location_city': pd.np.int64,
'orig_destination_distance': pd.np.float64,
'user_id': pd.np.int64,
'is_mobile': pd.np.int64,
'is_package': pd.np.int64,
'channel': pd.np.int64,
'srch_ci': pd.np.object,
'srch_co': pd.np.object,
'srch_adults_cnt': pd.np.int64,
'srch_children_cnt': pd.np.int64,
'srch_rm_cnt': pd.np.int64,
'srch_destination_id': pd.np.int64,
'srch_destination_type_id': pd.np.int64,
'hotel_continent': pd.np.int64,
'hotel_country': pd.np.int64,
'hotel_market': pd.np.int64}
print "Dtyping complete:",round(time()-t0,3),"s"

t0 = time()
print "Importing test, train, and destinations..."
test = pd.read_csv('test.csv', dtype=test_dtypes)
train = pd.read_csv('train.csv', dtype=train_dtypes, iterator=True, chunksize=100000)
# train = pd.concat([chunk[chunk['user_location_city'].isin(np.unique(test.user_location_city))] for chunk in train], ignore_index=True)
# train = pd.concat([chunk[chunk['is_booking'] == 1] for chunk in train], ignore_index=True)
train = pd.concat([chunk for chunk in train], ignore_index=True)
# print "Train length before user removal:", len(train)
# train = pd.concat([chunk[(chunk['user_id'].isin(np.unique(test.user_id))) & (chunk['is_booking'] == 1)] for chunk in train], ignore_index=True)
# train = train[train['user_id'].isin(np.unique(test.user_id))]
destinations = pd.read_csv('destinations.csv')
print "Importing data complete:",round((time()-t0)/60,2),"m"

print "Train Length:", len(train),  "Test Length:", len(test)

t0 = time()
print "PCA for destinations..."
destination_ids = destinations['srch_destination_id']
destinations = destinations.drop(['srch_destination_id'], 1)
pca = PCA(n_components=11, whiten=True)
destinations = pca.fit_transform(destinations)
destinations = pd.DataFrame(destinations)
destinations['srch_destination_id'] = destination_ids
print "PCA complete:",round(time()-t0,3),"s"

t0 = time()
print "Creating train features..."
# train['id'] = [i for i in range(0, len(train))]
train['orig_destination_distance'] = train['orig_destination_distance'].fillna(-1)
train['date_time'] = pd.to_datetime(train['date_time'], errors='coerce')
train['srch_ci'] = pd.to_datetime(train['srch_ci'], errors='coerce')
train['srch_co'] = pd.to_datetime(train['srch_co'], errors='coerce')
train['activity_month'] = train['date_time'].fillna(-1).dt.month.astype(int)
train['activity_year'] = train['date_time'].fillna(-1).dt.year.astype(int)
train['activity_dow'] = train['date_time'].fillna(-1).dt.dayofweek.astype(int)
train['activity_day'] = train['date_time'].fillna(-1).dt.day.astype(int)
train['activity_quarter'] = train['date_time'].fillna(-1).dt.quarter.astype(int)
train['checkin_month'] = train['srch_ci'].fillna(-1).dt.month.astype(int)
train['checkin_year'] = train['srch_ci'].fillna(-1).dt.year.astype(int)
train['checkin_dow'] = train['srch_ci'].fillna(-1).dt.dayofweek.astype(int)
train['checkin_day'] = train['srch_ci'].fillna(-1).dt.day.astype(int)
train['checkin_quarter'] = train['srch_ci'].fillna(-1).dt.quarter.astype(int)
train['checkout_month'] = train['srch_co'].fillna(-1).dt.month.astype(int)
train['checkout_year'] = train['srch_co'].fillna(-1).dt.year.astype(int)
train['checkout_dow'] = train['srch_co'].fillna(-1).dt.dayofweek.astype(int)
train['checkout_day'] = train['srch_co'].fillna(-1).dt.day.astype(int)
train['checkout_quarter'] = train['srch_co'].fillna(-1).dt.quarter.astype(int)
train['stay_length'] = (train['srch_co'] - train['srch_ci']).astype(int)
print "Train features complete:",round((time()-t0)/60,3),"m"

t0 = time()
print "Creating test features..."
test['orig_destination_distance'] = test['orig_destination_distance'].fillna(-1)
test['date_time'] = pd.to_datetime(test['date_time'], errors='coerce')
test['srch_ci'] = pd.to_datetime(test['srch_ci'], errors='coerce')
test['srch_co'] = pd.to_datetime(test['srch_co'], errors='coerce')
test['activity_month'] = test['date_time'].fillna(-1).dt.month.astype(int)
test['activity_year'] = test['date_time'].fillna(-1).dt.year.astype(int)
test['activity_dow'] = test['date_time'].fillna(-1).dt.dayofweek.astype(int)
test['activity_day'] = test['date_time'].fillna(-1).dt.day.astype(int)
test['activity_quarter'] = test['date_time'].fillna(-1).dt.quarter.astype(int)
test['checkin_month'] = test['srch_ci'].fillna(-1).dt.month.astype(int)
test['checkin_year'] = test['srch_ci'].fillna(-1).dt.year.astype(int)
test['checkin_dow'] = test['srch_ci'].fillna(-1).dt.dayofweek.astype(int)
test['checkin_day'] = test['srch_ci'].fillna(-1).dt.day.astype(int)
test['checkin_quarter'] = test['srch_ci'].fillna(-1).dt.quarter.astype(int)
test['checkout_month'] = test['srch_co'].fillna(-1).dt.month.astype(int)
test['checkout_year'] = test['srch_co'].fillna(-1).dt.year.astype(int)
test['checkout_dow'] = test['srch_co'].fillna(-1).dt.dayofweek.astype(int)
test['checkout_day'] = test['srch_co'].fillna(-1).dt.day.astype(int)
test['checkout_quarter'] = test['srch_co'].fillna(-1).dt.quarter.astype(int)
test['stay_length'] = (test['srch_co'] - test['srch_ci']).astype(int)
print "Test features complete:",round((time()-t0)/60,3),"m"

t0 = time()
print "Merging train with destinations..."
train = pd.merge(train, destinations, how='left')
train.fillna(-1, inplace=True)
print "Merge train with destinations complete:",round((time()-t0)/60,3),"m"

t0 = time()
print "Merging test with destinations..."
test = pd.merge(test, destinations, how='left')
test.fillna(-1, inplace=True)
print "Merge test with destinations complete:",round((time()-t0)/60,3),"m"

features = [c for c in train.columns if c not in ['id', 'is_booking', 'cnt', 'hotel_cluster', 'date_time', 'srch_ci', 'srch_co']]

from keras.models import Sequential

model = Sequential()

from keras.layers import Dense, Activation

model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(train[features], train['hotel_cluster'], nb_epoch=5, batch_size=32)

# print len(np.unique(train.user_id)), len(np.unique(test.user_id))

# features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(train[features], train['hotel_cluster'], test_size=0.50)
# features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features_train, labels_train, test_size=0.60)

# # neigh = neighbors.KNeighborsClassifier(weights='distance', n_jobs=-1).fit(train[features], train['hotel_cluster'])
# forest = ensemble.RandomForestClassifier(n_estimators=10, n_jobs=-1).fit(train[features], train['hotel_cluster'])
# # bayes = naive_bayes.GaussianNB().fit(train[features], train['hotel_cluster'])

t0 = time()
print "Predicting probabilities..."
# proba = model.predict_proba(X_test, batch_size=32)
probs = pd.DataFrame(model.predict_proba(test[features], batch_size=32))

# probs = pd.DataFrame(forest.predict_proba(test[features]))
probs.columns = np.unique(train['hotel_cluster'].sort_values().values)
# probs.columns = np.unique(labels_train.sort_values().values)
# probs.columns = np.unique(labels_train.values)
preds = pd.DataFrame([list([r.sort_values(ascending=False)[:5].index.values]) for i,r in probs.iterrows()])
# # print mapk([[l] for l in labels_test], preds[0], 5)
print "Probablity prediction complete:",round((time()-t0)/60,3),"m"


t0 = time()
print "Creating submission..."
submission = pd.DataFrame()
submission['id'] = test['id']
submission['hotel_cluster'] = [' '.join(str(x) for x in y) for y in preds.values]
submission.sort_values(by='id', inplace=True)
# submission.head()
submission.to_csv('submission.csv', index=False)
print "Submission creation complete:",round((time()-t0)/60,3),"m"

print "Script End:",round((time()-start_time)/60,2),"m"