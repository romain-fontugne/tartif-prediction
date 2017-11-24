import dataManipulation as dm
from sklearn import tree
from sklearn import svm as ssvm
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier


def decisionTree(X, Y, cv=5, verbose=0):
    """Decision tree classification."""

    clf = tree.DecisionTreeClassifier(class_weight="balanced")
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', clf))
    pipeline = Pipeline(estimators)
    results = cross_validate(
        pipeline, X, Y,
        scoring=["precision", "recall", "accuracy"],
        cv=cv, verbose=verbose, return_train_score=False)
    return results


def randomForest(X, Y, cv=5, verbose=0):
    """Random forest classification."""

    clf = ensemble.RandomForestClassifier(100, class_weight="balanced")
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', clf))
    pipeline = Pipeline(estimators)
    results = cross_validate(
        pipeline, X, Y, scoring=["precision", "recall", "accuracy"],
        cv=cv, verbose=verbose, return_train_score=False)
    return results


def svm(X, Y, cv=5, verbose=0):
    """Support vector machine classification."""

    clf = ssvm.SVC(max_iter=10000, class_weight="balanced")
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', clf))
    pipeline = Pipeline(estimators)
    results = cross_validate(
        pipeline, X, Y, scoring=["precision", "recall", "accuracy"], 
        cv=cv, verbose=verbose, return_train_score=False)
    return results


def createMlpModel():
    """Function to create mlp model, required for KerasClassifier."""
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=20, activation="sigmoid"))
    # model.add(Dense(150, activation="sigmoid"))
    # model.add(Dense(150, activation="sigmoid"))
    # model.add(Dense(150, activation="sigmoid"))
    # model.add(Dense(90, activation="sigmoid"))
    model.add(Dense(30, activation="sigmoid"))
    model.add(Dense(30, activation="sigmoid"))
    model.add(Dense(10, activation="sigmoid"))
    model.add(Dense(1, activation="sigmoid"))
    # Compile model
    model.compile(
        loss="binary_crossentropy", optimizer=SGD(lr=1.0),
        metrics=["accuracy"])

    return model


def mlp(X, Y, cv=5, verbose=0):

    # ratio of leaving vs. staying 1:467
    clf = KerasClassifier(build_fn=createMlpModel, epochs=100, batch_size=512)
    # class_weight={1:450.0, 0:1.0})

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', clf))
    pipeline = Pipeline(estimators)
    results = cross_validate(
        pipeline, X, Y, scoring=["precision", "recall", "accuracy"], 
        cv=cv, verbose=verbose, return_train_score=False)
    return results


def printScores(precision, recall):

    print("Precision: %0.2f (+/- %0.2f)" % (precision.mean(), precision.std()*2))
    print(precision)
    print("Recall: %0.2f (+/- %0.2f)" % (recall.mean(), recall.std()*2))
    print(recall)


def main():
    verbose = 0
    kfold = StratifiedKFold(n_splits=3, shuffle=True)
    md = dm.mioData()
    X, Y = md.loadData()

    print("MLP scores: ")
    results = mlp(X, Y, kfold, verbose)
    print results

    print("Decision tree scores: ")
    precision, recall = decisionTree(X, Y, kfold, verbose)
    print results

    print("Random forest scores: ")
    precision, recall = randomForest(X, Y, kfold, verbose)
    print results

    print("SVM scores: ")
    precision, recall = svm(X, Y, kfold, verbose)
    print results


if __name__ == "__main__":
    main()
