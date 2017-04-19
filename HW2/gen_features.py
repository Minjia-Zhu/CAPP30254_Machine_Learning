import pandas as pd
import numpy as np
import math
import pylab as pl
from sklearn.ensemble import RandomForestClassifier


def find_features(df, features):
    '''
    Use scikit-learn lib to determine which variables are the best at predicting risk.
    Then, from the calculated importances, order them from most to least important
    and make a barplot to visualize what is/isn't important
    '''
    clf = RandomForestClassifier()
    clf.fit(df[features], df[DEP_VAR])
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)
    padding = np.arange(len(features)) + 0.5
    pl.barh(padding, importances[sorted_idx], align='center')
    pl.yticks(padding, features[sorted_idx])
    pl.xlabel("Relative Importance")
    pl.title("Variable Importance")


def generate_bins(df,var,size):
    '''
    Generate a list of bin boundary for categorical var
    
    '''
    lb = int(math.floor(df[var].describe()[3]))
    lb2 = int(math.floor(df[var].describe()[4]))
    ub = int(math.ceil(df[var].describe()[7]))
    ub2_temp = int(math.ceil(df[var].describe()[6]))
    bins = int(math.ceil((ub2_temp-lb2)/size))
    ub2 = int(lb2+size*(bins))
    bins = [lb] + range(lb2,ub2+size,size) + [ub]
    return bins

def build_category(df,var,bins):
    '''
    Discretize a continous variable
    '''
    new_name = var + '_bucket'
    df[new_name] = pd.cut(df[var], bins, labels=False,include_lowest=True)
    return df

def create_dummy(df,var):
    '''
    Take categorical var and create binary/dummy variables from it
    '''
    dummy_df = pd.get_dummies(df[var],prefix=var)
    return dummy_df