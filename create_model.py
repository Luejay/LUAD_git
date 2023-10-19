from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import  f_classif, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import spearmanr

import pandas as pd
import random

from sklearn.model_selection import StratifiedKFold, KFold

import os

from datetime import datetime

from sklearn.base import clone

import joblib

from config import rand_var

from tqdm import tqdm


class DropCollinear(BaseEstimator, TransformerMixin):
    
    def __init__(self, thresh:float = 0.8):
        
        self.thresh = thresh
        
        self.columns_to_drop = [] #columns of features to drop based on a threshold
        
        
        
    def fit(self, X, y=None):

        # Find variables to remove
        X_corr = X.corr()
        large_corrs = X_corr>self.thresh #boolean matrix of all coorlations greater than threshold
        indices = np.argwhere(large_corrs.values)#all indicies where coorlation is greater
        
        indices_nodiag = np.array([[m,n] for [m,n] in indices if m!=n])#remove diagonal cases
        
        
        #end if there are no pairs
        if indices_nodiag.size ==0:
            return self
        
        

        indices_nodiag_lowfirst = np.sort(indices_nodiag, axis=1)#sort in ascending order
        correlated_pairs = np.unique(indices_nodiag_lowfirst, axis=0)#extract unique pairs
            
        #calculate coorlation between each of the pair of features to y
        resp_corrs = np.array([[np.abs(spearmanr(X.iloc[:,m], y).correlation), np.abs(spearmanr(X.iloc[:,n], y).correlation)] for [m,n] in correlated_pairs])
            
            
            
        element_to_drop = np.argmin(resp_corrs, axis=1) #drop feature with least coorlating to y
        list_to_drop = np.unique(correlated_pairs[range(element_to_drop.shape[0]),element_to_drop])#list of indicies to drop
            
            
            
        self.columns_to_drop = X.columns.values[list_to_drop]

        

        return self
    
    
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)
    
    
    def get_params(self, deep=False):
        return {'thresh': self.thresh}










class SelectAtMostKBest(SelectKBest):
    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            # set k to "all" (skip feature selection), if less than k features are available
            self.k = "all"




#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#make model functions

def make_linear_regression_model(X:pd.DataFrame, 
                          y: pd.DataFrame, 
                          cor_threshold:float, 
                          cv:list,
                          metric='neg_mean_squared_error',
                          model_iter = 500,
                          search_iter = 100 ):
    '''
    returns the search result of sklearn randomsearch method of logistic regression model
    
    input: 
    x - pd dataframe
    y - pd dataframe
    cor_threshold - coorlation threshold from 0-1.0 to drop a feature coorlating with another
    cv - folds
    
    '''
    
    linear_regression = ElasticNet(random_state = rand_var,  max_iter = model_iter)
    
    pipe = Pipeline(steps = [
        ('dropcoll', DropCollinear(cor_threshold)),
        ('scaler', StandardScaler()),
        ('kbest', SelectAtMostKBest(score_func=f_regression)),
        ('linres', linear_regression)
    ])
    
    param_grid = {
        'kbest__k': np.arange(2,X.shape[1],1),
        'linres__alpha': [0.01, 0.1, 1, 10],
        'linres__l1_ratio': np.arange(0.0, 1.1, 0.25)
    }
    
    search = RandomizedSearchCV(pipe, param_grid, cv = cv, scoring=metric, return_train_score=True, n_jobs = -1, verbose = 0, n_iter = search_iter, random_state= rand_var)
    search.fit(X,y)
    
    return search
    


#9/26 not used for this task
def make_logres_model(X:pd.DataFrame, y:pd.DataFrame, cor_threshold:float, cv:list, penalty = 'elasticnet',metric='roc_auc',max = 100):
    '''
    returns the search result of sklearn randomsearch method of logistic regression model
    
    input: 
    x - pd dataframe
    y - pd dataframe
    cor_threshold - coorlation threshold from 0-1.0 to drop a feature coorlating with another
    cv - folds
    penalty = 'elasticnet' for L1+L2, 'l1' for L1, 'l2' for L2
    
    
    
    '''
    
    logres = LogisticRegression(random_state=rand_var,penalty=penalty, solver='saga', max_iter=500, n_jobs=-1, class_weight=True)
    
    pipe = Pipeline(steps=[('dropcoll', DropCollinear(cor_threshold)), 
                           ('scaler', StandardScaler()), 
                           ('kbest', SelectAtMostKBest(score_func=f_classif)), 
                           ('logres', logres)])

    

    param_grid = {  'kbest__k': np.arange(2,X.shape[1],1),
                    'logres__C': np.logspace(-3,3,30),
                    'logres__l1_ratio': [0.0, 0.5, 1.0],
                    'logres__class_weight': ['balanced',{0:1,1:2},{0:2,1:1}]}
    
    
    

    search = RandomizedSearchCV(pipe, param_grid, cv=cv, scoring=metric, return_train_score=True, n_jobs=-1, verbose=0,n_iter=max,random_state=rand_var)
    search.fit(X,y)

    return search


def make_svc_model(X:pd.DataFrame,
                   y:pd.DataFrame, 
                   cor_threshold:float, 
                   cv:list,
                   metric='neg_mean_squared_error',
                   model_iter = 500,
                   search_iter = 100 ):
    # Pipeline components
    svc = SVC(random_state=rand_var, max_iter=model_iter, probability=True)
    pipe = Pipeline(steps=[('dropcoll',  DropCollinear(cor_threshold)), ('scaler', StandardScaler()), ('kbest', SelectAtMostKBest(score_func=f_classif)), ('svc', svc)])

    param_grid = { 'kbest__k': np.arange(2,X.shape[1],1),
                    'svc__kernel': ['rbf','sigmoid','linear'],
                    'svc__gamma': np.logspace(-9,-2,60),
                    'svc__C': np.logspace(-3,3,60)}

    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, cv=cv,scoring=metric,return_train_score=True, n_jobs=-1, verbose=0,n_iter=search_iter, random_state=rand_var)
    search.fit(X,y)

    return search

#for binary classification, not used for regression task
def make_gb_model(X:pd.DataFrame, 
                  y:pd.DataFrame, 
                  cor_threshold:float, 
                  cv:list,
                  metric='neg_mean_squared_error',
                  model_iter = 500, #not used
                  search_iter = 100 ):
    
    # Pipeline components

    gb = GradientBoostingClassifier(random_state=rand_var)
    pipe = Pipeline(steps=[('dropcoll', DropCollinear(cor_threshold)), 
                           ('scaler', StandardScaler()), 
                           ('kbest', SelectAtMostKBest(score_func=f_classif)), 
                           ('gb', gb)])
    
    
    # Parameter ranges
    param_grid = { 'kbest__k': np.arange(2,X.shape[1],1),
                  "gb__n_estimators": [5, 10, 25, 50, 100],
                  "gb__max_depth": [1,2,3,4,5,6, None],
                  "gb__max_features": [0.05, 0.1, 0.2, 0.5, 0.7,0.9],
                  "gb__min_samples_split": [2, 3, 6, 10, 12, 15]
                  
                  }
    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, cv=cv, scoring=metric,return_train_score=True, n_jobs=-1, verbose=0,n_iter=search_iter,random_state=rand_var)
    search.fit(X,y)

    return search

def make_gb_regression_model(X:pd.DataFrame, 
                  y:pd.DataFrame, 
                  cor_threshold:float, 
                  cv:list,
                  metric='neg_mean_squared_error',
                  model_iter = 500, #not used
                  search_iter = 100 ):
    gb = GradientBoostingRegressor(random_state=rand_var)
    pipe = Pipeline(steps=[('dropcoll', DropCollinear(cor_threshold)), 
                           ('scaler', StandardScaler()), 
                           ('kbest', SelectKBest(score_func=f_regression)),  # Use f_regression for regression tasks
                           ('gb', gb)])
    # Parameter ranges
    param_grid = {
        'kbest__k': np.arange(2, X.shape[1], 1),
        "gb__n_estimators": [5, 10, 25, 50, 100],
        "gb__max_depth": [1, 2, 3, 4, 5, 6, None],
        "gb__max_features": [0.05, 0.1, 0.2, 0.5, 0.7, 0.9],
        "gb__min_samples_split": [2, 3, 6, 10, 12, 15]
    }
    
    # Optimization
    search = RandomizedSearchCV(pipe, param_grid, cv=cv, scoring=metric, return_train_score=True, n_jobs=-1, verbose=0, n_iter=search_iter, random_state=rand_var)
    search.fit(X, y)
    
    return search
    
    
    

#splits for data for binary classification
def defineSplits(X,ycateg):
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    splits = []
    for (tr,ts) in cv.split(X, ycateg):
        splits.append((tr,ts))
    return splits


# for continuous y value
def define_splits_for_regression(X, ycateg):
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)

    splits = []
    for train_index, test_index in kf.split(X,ycateg):
        splits.append((train_index, test_index))
        
        
    return splits



def create_all_models(X:pd.DataFrame,
                   y:pd.DataFrame, 
                   y_featname: str,
                   cor_threshold:float, 
                   cv:list,
                   test = False,
                   model_iter = 100,
                   search_iter = 100,
                   unique_name_of_dir = "",
                   create_model_for_feature_importance = False,
                   make_linres = True,
                   make_svc = True,
                   make_gb = True
                   ):
    
    
    '''
    Function that makes all the models and models without a feature for the z score calculation
    
    '''
    
    tm = datetime.now()
    
    dir_save_all_model = f"./models_made/{unique_name_of_dir}_{tm.year}_{tm.month}_{tm.day}_{tm.strftime('%H_%M_%S')}_randvar_{rand_var}_cor_{cor_threshold}_yvar_{y_featname}"
    
    if test:
        dir_save_all_model = "./models_made/test"
        model_iter = 100
        search_iter = 30
        
        
    os.makedirs(dir_save_all_model,exist_ok=True)
    
    
    #get the search result of the model
    
    if make_linres:
    
        linres_best_model = make_linear_regression_model(X,y,cor_threshold,cv,model_iter=model_iter, search_iter=search_iter)
        dir_linres = f"{dir_save_all_model}/linres"
        os.makedirs(dir_linres,exist_ok=True)
        joblib.dump(linres_best_model,f"{dir_linres}/baseline.pkl")
        
        print("Finished Creating Linres model")
    
    

    
    if make_svc:
        svc_best_model = make_svc_model(X,y,cor_threshold,cv,model_iter=model_iter, search_iter=search_iter)
    
        dir_svc = f"{dir_save_all_model}/svc"
        os.makedirs(dir_svc,exist_ok=True)
        joblib.dump(svc_best_model,f"{dir_svc}/baseline.pkl")
        
        
    if make_gb:
        gb_best_model = make_gb_regression_model(X,y,cor_threshold,cv,model_iter=model_iter, search_iter=search_iter)
    
        dir_gb = f"{dir_save_all_model}/gb"
        
        os.makedirs(dir_gb,exist_ok=True)
        
        joblib.dump(gb_best_model,f"{dir_gb}/baseline.pkl")
        
        print("Finished Creating GB model")
    
    print("Finished Creating Baseline Models")
    
    #only make baseline models
    if not create_model_for_feature_importance:
        if test:
            return True
        else:
            return None
    
    list_of_features = X.columns.tolist()
    
    for feature_to_drop in tqdm(list_of_features):
        x_dropped = X.drop([feature_to_drop],axis = 1)
        model_name = f'without_{feature_to_drop}.pkl'
        
        if make_linres:
            linres_model_without_feat = clone(linres_best_model)
            linres_model_without_feat.fit(x_dropped,y)
            joblib.dump(linres_model_without_feat,f"{dir_linres}/{model_name}")
            
        if make_svc:
            svc_model_without_feat = clone(svc_best_model)
            svc_model_without_feat.fit(x_dropped,y)
            joblib.dump(svc_model_without_feat,f"{dir_svc}/{model_name}")
            
        if make_gb:
            gb_model_without_feat = clone(gb_best_model)
            gb_model_without_feat.fit(x_dropped,y)
            
            joblib.dump(gb_model_without_feat,f"{dir_gb}/{model_name}")
        
        
    if test:
        return True
        
    
    
    



if __name__=='__main__':
    random_data = pd.DataFrame({'x1':[random.random() for _ in range(30)],
                                'x2':[random.random() for _ in range(30)],
                                'x3':[random.random() for _ in range(30)],
                                'x4':[random.random() for _ in range(30)],
                                'x5':[random.random() for _ in range(30)],
                                'y':[0 if x < 0.5 else 1 for x in [random.random() for _ in range(30)]]})
    
    x_data = random_data.drop(['y'],axis = 1)
    
    y_data= random_data['y']
    
    cor_threshold_test = 0.9
    
    cv = defineSplits(x_data,y_data)
    
    
    
    print('Testing linres model')
    assert make_linear_regression_model(x_data,y_data,cor_threshold_test,cv,model_iter= 40, search_iter=20)
    
    print('Testing svc model')
    assert make_svc_model(x_data,y_data,cor_threshold_test,cv,model_iter= 40, search_iter=20)
    
    print('Testing gb model')
    assert make_gb_model(x_data,y_data,cor_threshold_test,cv,model_iter= 40, search_iter=20)
    
    list_of_features = x_data.columns.tolist()
    
    

    assert create_all_models(x_data,y_data,'y',cor_threshold_test,cv,list_of_features,test = True)
    print('Everything Worked As Expected')
    
    
    
    