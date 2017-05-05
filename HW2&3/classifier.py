import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.svm import SVC


def form_train_test(df,selected_features,portion):
    '''
    Separate a portion of data as train and the other portion as test set to work with.
    Inputs:
        selected_features: list of features fields that you think could be used as useful features for prediction
        portion(float): the portion of test data
    '''
    X = df[selected_features]
    y = df[DEP_VAR]
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=portion, random_state=42)

    return (X_train, X_test, y_train, y_test)


def classifier(classifier_type,X_train, X_test, y_train):
    '''
    Build classifier model
    Return:
        yhat(array): the fitted/predicted  value
        probs(array): Predict probabilities
    '''
    if classifier_type == "KNN":
        clf = KNeighborsClassifier(n_neighbors=13,metric='minkowski', weights='distance')
    elif classifier_type == "Logit":
        clf = LogisticRegression()
    elif classifier_type == "Tree":
        clf = tree.DecisionTreeClassifier()
    elif classifier_type == "GB":
        clf =GradientBoostingClassifier()
    else:
        raise ValueError('{c_type} not avaliable'.format(c_type=classifier_type))
        
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    return (yhat, probs)

