# -*- coding: utf-8 -*-
"""
Principal component analysis
ML implementation - Indonesia
------------------------------
@author: Mariza Cooray
version: July 2021
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
# ---------------------------------
# Importing libraries:
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components = 10) # <----- Change here number of PCs
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import os
os.chdir(r'C:/2021/07_July/Mariza/results/')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier # Stochastic gradient descending
import xgboost # Gradient boosting algorithm
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.ensemble import RandomForestClassifier # Random forest

#%% data source route
loading_data = 'C:/2021/07_July/Mariza/db/'

#%% Loading data:
df = pd.read_csv(loading_data + 'db_final_July2021.csv')

#%% Defining target (dependent) variable
df['y'] = df.lrpcexp # Continuous variable
df.drop(columns = 'lrpcexp', inplace = True)

#%% Parameters
threshold = 0.45  # Poverty threshold
train_per = 0.67  # Percentage of data in the train sample (67%), test: 33%

#%% Principal component analysis 

# PCA is applied in the loop at the end of the script

#%% Normalization
# sn: separate estimations by kabupayten; separate estimations by year

# No normalization is implemented

# See loop at the end of the script (this method estimates different models
# for different kabupatens and different years). PCAs are calculated 
# separateky for each kabu/year. No normalization is implemented

#%% Defining functions for models (now including household weights):
    
# Models that take as input the continuous target y:
def linear_reg(X_train, y_train, X_test, wert_train):
    reg = LinearRegression().fit(X_train, y_train, sample_weight=wert_train)
    preds_temp = reg.predict(X_test)
    preds = pd.Series(preds_temp).rank(pct = True).apply(lambda x: 1 if x < threshold else 0)
    return preds

def elastic_net(X_train, y_train, X_test, wert_train):
    # The problem was the default value of the penalty strength parameter that
    # multiplies the regularization parameters, so I reduced the value from 1 to .01
    reg = ElasticNet(random_state=0, alpha=0.01, max_iter = 1500).fit(X_train, y_train, sample_weight=wert_train)
    preds_temp = reg.predict(X_test)
    preds = pd.Series(preds_temp).rank(pct = True).apply(lambda x: 1 if x < threshold else 0)
    return preds

def sgdclf(X_train, y_train_cat, X_test, wert_train):
    # Stochastic gradient:
    sgd = SGDClassifier(random_state=0,loss='modified_huber',penalty='l1')
    # Fitting SGD on the train sample:
    sgd.fit(X_train, y_train_cat, sample_weight=wert_train)
    # Predicting y in the test sample:
    preds = sgd.predict(X_test)
    return preds

def xgb(X_train, y_train, X_test, wert_train):
    # Estimating gradient boosting model:        
    xgb_model = xgboost.XGBRegressor(objective='reg:squarederror')
    xgb_model.fit(X_train, y_train, sample_weight=wert_train)
    # Predicting y in the test sample:
    preds_temp = xgb_model.predict(X_test)
    preds = pd.Series(preds_temp).rank(pct = True).apply(lambda x: 1 if x < threshold else 0)
    preds.reset_index(drop = True, inplace = True)
    return preds

# Models (classifiers) that take as input the discrete target y:
def log_reg(X_train, y_train_cat, X_test, wert_train):
    reg = LogisticRegression(random_state=0).fit(X_train, y_train_cat, sample_weight=wert_train)
    preds = reg.predict(X_test)
    # If logistic regression fails due to imbalance, I introduced below a
    # class_weight = 'balanced' in order to modify the loss function
    # from cross entropy to weighted cross-entropy. I made this change because
    # if the model is being affected by class imbalance it will not produce
    # any results
    prop_preds = preds.sum()/len(preds)
    if prop_preds < .1:
        reg = LogisticRegression(random_state=0,class_weight = 'balanced').fit(X_train, y_train_cat, sample_weight=wert_train)
        preds = reg.predict(X_test)
    return preds

def nn(X_train, y_train_cat, X_test, y_train):
    # Note: MLPClassifier does not currently supports the use of sample weights
    clf = MLPClassifier(solver='adam',
                     hidden_layer_sizes=(10,10), max_iter = 500, random_state=0)
    clf.fit(X_train, y_train_cat)
    preds = clf.predict(X_test)
    # If MLPClassifier fails, I change to MLPRegressor:
    prop_preds = preds.sum()/len(preds)
    if prop_preds < .1:
        clf = MLPRegressor(solver='adam',
                         hidden_layer_sizes=(10,10), max_iter = 500, random_state=0)
        clf.fit(X_train, y_train)
        preds_temp = clf.predict(X_test)
        preds = pd.Series(preds_temp).rank(pct = True).apply(lambda x: 1 if x < threshold else 0)
    return preds

def nBayes(X_train, y_train_cat, X_test, wert_train):
    # Defining Naive Bayes model:
    naive_bayes = GaussianNB()
    # Fitting and predicting y in the test sample:
    preds = naive_bayes.fit(X_train, y_train_cat, sample_weight=wert_train).predict(X_test)
    return preds

def ranforest(X_train, y_train_cat, X_test, wert_train):
    # Defining random forest:
    rforest = RandomForestClassifier(random_state=0,max_features=3,n_estimators=10,n_jobs = -1)
    # Fitting random forest on the train sample:
    rforest.fit(X_train, y_train_cat, sample_weight=wert_train)
    # Predicting y in the test sample:
    preds = rforest.predict(X_test)
    return preds

#%% Metrics function
def metrics(y_test_cat, preds):
    r2 = r2_score(y_test_cat, preds)
    ytest = np.array(y_test_cat)
    preds = np.array(preds)
    tn, fp, fn, tp = confusion_matrix(ytest, preds).ravel()
    if (tn+fp) == 0:
        specificity = float('NaN')
        sensitivity = tp / (tp+fn)
        inclusion = fp / (fp+tp)
        exclusion = fn / (tp+fn)
    elif (tp+fn) == 0:
        specificity = tn / (tn+fp)
        sensitivity = float('NaN')
        inclusion = fp / (fp+tp)
        exclusion = float('NaN')
    elif (fp+tp) == 0:
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp+fn)
        inclusion = float('NaN')
        exclusion = fn / (tp+fn)
    else:
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp+fn)
        inclusion = fp / (fp+tp)
        exclusion = fn / (tp+fn)
    return (sensitivity, specificity, exclusion, inclusion, r2)

#%% Master function
def master(X, y, kabupaten, year):
    # This part below splits the sample:
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(1-train_per), random_state=0)
    
    # Here I categorize the continuous target (for classification models):
    y_cat = y
    # Below I changed the categorization from zero to 1 for poor, and from 1 to zero to non-poor:
    y_cat = y_cat.rank(pct = True).apply(lambda x: 1 if x < threshold else 0)
    y_train_cat = y_cat.loc[y_train.index]
    y_test_cat  = y_cat.loc[y_test.index]
    
    # Here I prepare the sample weights at household level:
    wert = X['wert']
    wert_train  = wert.loc[y_train.index]
    
    # Below I removed from the X matrix the variables that should not be
    # explanatory terms: household weights (wert) 
    X_train = X_train.drop(columns = 'wert')
    X_test = X_test.drop(columns = 'wert')
    
    # This part below estimates different models:
    print('Estimating linear regression...')
    y1_cat = linear_reg(X_train, y_train, X_test, wert_train)
    print('Estimating elastic nets...')
    y2_cat = elastic_net(X_train, y_train, X_test, wert_train)
    print('Estimating logistic regression...')
    y3_cat = log_reg(X_train, y_train_cat, X_test, wert_train)
    print('Estimating neural networks...')
    y4_cat = nn(X_train, y_train_cat, X_test, y_train)
    print('Estimating stochastic gradient classifier...')
    y5_cat = sgdclf(X_train, y_train_cat, X_test, wert_train)
    print('Estimating gradient boosting model...')
    y6_cat = xgb(X_train, y_train, X_test, wert_train)
    print('Estimating naive bayes classification...')
    y7_cat = nBayes(X_train, y_train_cat, X_test, wert_train)
    print('Estimating random forest...')
    y8_cat = ranforest(X_train, y_train_cat, X_test, wert_train)
    print('Models estimated')
 
    print('Calculating performance metrics for kabupaten: ',str(kabupaten))
    final = []
        
    y0_temp = y_test_cat

    final.append(metrics(y0_temp, y1_cat))
    final.append(metrics(y0_temp, y2_cat))
    final.append(metrics(y0_temp, y3_cat))
    final.append(metrics(y0_temp, y4_cat))
    final.append(metrics(y0_temp, y5_cat))
    final.append(metrics(y0_temp, y6_cat))
    final.append(metrics(y0_temp, y7_cat))
    final.append(metrics(y0_temp, y8_cat))  
    
    final = pd.DataFrame(final)
    final.columns = ['sensitivity', 'specificity', 'exclusion error', 'inclusion error', 'r2 score']
    final.index = ['Linear regression', 'Elastic net', 'Logistic classification', 'Neural network', 'Stochastic gradient','Gradient boosting', 'Naive Bayes','Random forest']
         
    final.to_csv('ss_method/' + str(yr) + '/' + str(kabu) + '.csv')  
    return


#%% Master function: this function implements the whole analysis pipeline:
# (1) Splits the data
# (2) Estimates the models
# (3) Calculates performance metrics in the test sample, for each kabupaten
# ss: No normalization, separate estimations for each kabupaten and by year 

# Separate estimations for each kabupaten and for each year in the double loop below:
df2 = df.copy()
for yr in np.unique(df2.year):
    df = df2[df2.year == yr]
    for kabu in np.unique(df.id_kabu):
        tp = df[df.id_kabu == kabu]
        X = tp.drop(columns = ['id_kabu','wert','y', 'year'])
        X = pd.DataFrame(pca.fit_transform(X)) # No normalization
        tp.reset_index(inplace = True)
        X['id'] = tp['id_kabu']
        X['wert'] = tp.wert
        y = tp.y
        master(X, y, kabu, yr)
        print('---- Models estimated for kabupaten',kabu,'and year',yr,'----') 
        
# ----------------------- end of script --------------------------------------

