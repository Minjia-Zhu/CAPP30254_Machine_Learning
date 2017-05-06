
# coding: utf-8

# In[1]:

#Code for the pipeline was significantly inspired on:
# /rayidghani/magicloops/blob/master/magicloops.py
# /hectorsalvador/ML_for_Public_Policy


# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pylab as pl
pd.options.display.max_rows = 99
pd.options.display.max_columns = 99

# import pipline files:
import read
import explore
import clean
import features
import classify
import evaluate

#Global Variable:
DEP_VAR ='SeriousDlqin2yrs'
explore.DEP_VAR ='SeriousDlqin2yrs'
explore.LEAD_VAR = 2


# In[3]:

df = read.read("credit-data.csv")
df.head()


# In[4]:

d = explore.explore(df)
summary = d["summary"]
fts = d["features"]


# In[5]:

fts


# In[6]:

# Summary statistics for the whole dataset
summary


# In[7]:

explore.plots(df)


# In[8]:

age_graph = explore.explore_var(df,'age','line')["graph"]


# In[9]:

loan_graph = explore.explore_var(df,'NumberOfOpenCreditLinesAndLoans','line')["graph"]


# In[10]:

dependent_graph = explore.explore_var(df,'NumberOfDependents','line')["graph"]


# In[11]:

#check the null value
clean.check_missing_data(df)


# In[12]:

#Check null values again after filling in missing values
df = clean.clean(df,'NumberOfDependents','zero')
df = clean.clean(df,'MonthlyIncome','mean')
clean.check_missing_data(df)


# Generate categorical bin boundary for selected variables

# In[13]:

features.binning(df, 'MonthlyIncome', 'quantiles', [0, 0.25, 0.5, 0.75, 1])
fts.append(df.keys()[-1])


# In[14]:

df.describe()


# In[15]:

sel_fts = ['RevolvingUtilizationOfUnsecuredLines',
 'NumberOfTime30-59DaysPastDueNotWorse',
 'DebtRatio',
 'NumberOfOpenCreditLinesAndLoans',
 'NumberOfTimes90DaysLate',
 'NumberRealEstateLoansOrLines',
 'NumberOfTime60-89DaysPastDueNotWorse',
 'NumberOfDependents']


# In[16]:

grid_size = 'small'
classifiers, grid = classify.define_clfs_params(grid_size)
models=['RF','DT','KNN', 'ET', 'AB', 'GB', 'LR', 'NB']
metrics = ['precision', 'recall', 'f1', 'auc']


# In[ ]:

all_models = classify.classify(df[sel_fts], df[DEP_VAR], models, 3, 0.05, metrics,classifiers, grid)
#the number printing below is the time cost for each model


# In[ ]:

all_models


# In[ ]:

table_auc, best_models_auc, winner_auc = classify.select_best_models(all_models, models, 'auc')
table_prec, best_models_prec, winner_prec = classify.select_best_models(all_models, models, 'precision')
table_rec, best_models_rec, winner_rec = classify.select_best_models(all_models, models, 'recall')
table_f1, best_models_f1, winner_f1 = classify.select_best_models(all_models, models, 'f1')


# In[ ]:

table_auc


# In[ ]:

table_prec


# In[ ]:

table_rec


# In[ ]:

table_f1


# In[ ]:

best_models_auc


# In[ ]:

best_models_prec


# In[ ]:

best_models_rec


# In[ ]:

best_models_f1


# In[ ]:

winner_auc


# In[ ]:

winner_prec


# In[ ]:

winner_rec


# In[ ]:

winner_f1


# In[ ]:

classify.gen_precision_recall_plots(df[sel_fts], df[DEP_VAR], best_models_auc,classifiers)


# In[ ]:




# In[ ]:




# In[ ]:
