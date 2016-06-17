from time import time
start_time = time()
t0 = time()
print "Importing magical pythons..."
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import xgboost as xgb
from datetime import datetime, date
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
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

t0 = time()
print "Importing test, train, and destinations..."
train = pd.read_csv('combined_data.csv', iterator=True, chunksize=100000)
train = pd.concat([chunk for chunk in train], ignore_index=True)
print "Importing data complete:",round((time()-t0)/60,2),"m"

# t0 = time()
# print "Building model..."
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import SGD

# model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
# model.add(Dense(64, input_dim=34, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(64, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(10, init='uniform'))
# model.add(Activation('softmax'))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
t0 = time()
print "Creating testing and training sets..."
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(train[features], train['hotel_cluster'], test_size=0.70)
print "Test and train sets complete:",round((time()-t0)/60,3),"m"

# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
t0 = time()
print "Building model..."
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
 
# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=34, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
 
# create model
model = KerasClassifier(build_fn=create_model, nb_epoch=20, batch_size=32)
# evaluate using 10-fold cross validation
# kfold = KFold(n=len(features_train), n_folds=10, shuffle=True, random_state=seed)
# results = cross_val_score(model, features_train.values, labels_train.values, cv=kfold)
# print "Cross validation results:", (results.mean()*100), (results.std()*100)
model.fit(features_train.values, labels_train.values)

print "Model building complete:",round((time()-t0)/60,3),"m"

# print len(np.unique(train.user_id)), len(np.unique(test.user_id))

# features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features_train, labels_train, test_size=0.60)

# # neigh = neighbors.KNeighborsClassifier(weights='distance', n_jobs=-1).fit(train[features], train['hotel_cluster'])
# forest = ensemble.RandomForestClassifier(n_estimators=10, n_jobs=-1).fit(train[features], train['hotel_cluster'])
# # bayes = naive_bayes.GaussianNB().fit(train[features], train['hotel_cluster'])

t0 = time()
print "Predicting probabilities..."
probs = pd.DataFrame(model.predict_proba(features_test.values, batch_size=32))

# probs = pd.DataFrame(forest.predict_proba(test[features]))
probs.columns = np.unique(labels_train.sort_values().values)
# probs.columns = np.unique(labels_train.sort_values().values)
# probs.columns = np.unique(labels_train.values)
preds = pd.DataFrame([list([r.sort_values(ascending=False)[:5].index.values]) for i,r in probs.iterrows()])
print "Mapk score for model:", mapk([[l] for l in labels_test], preds[0], 5)
print "Probablity prediction complete:",round((time()-t0)/60,3),"m"

# t0 = time()
# print "Creating submission..."
# submission = pd.DataFrame()
# submission['id'] = test['id']
# submission['hotel_cluster'] = [' '.join(str(x) for x in y) for y in preds.values]
# submission.sort_values(by='id', inplace=True)
# # submission.head()
# submission.to_csv('submission.csv', index=False)
# print "Submission creation complete:",round((time()-t0)/60,3),"m"

print "Script End:",round((time()-start_time)/60,2),"m"