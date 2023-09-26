import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

import joblib

from torchviz import make_dot

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import numpy as np

import shap


def calculate_feature_importance(X:pd.DataFrame,y:pd.DataFrame,dir:str):


    
    y_pred_base = joblib.load(f'{dir}/baseline.pkl').predict_proba(X)[:, 1]

    base_auc = roc_auc_score(y,y_pred_base)
    
    df_raw_roc_score = pd.DataFrame(index=X.columns, columns=['ROC_AUC'])
    df_z_score = pd.DataFrame(index=X.columns, columns=['z-score'])
    
    for feature in X.columns:
        
        x_without_feat = X.drop(columns=[feature])
        
        model_without_feat = joblib.load(f'{dir}/without_{feature}.pkl')
        
        y_pred_without_feat = model_without_feat.predict_proba(x_without_feat)[:, 1]
        
        auc_without_feat = roc_auc_score(y, y_pred_without_feat)
        
        df_raw_roc_score.loc[feature, 'ROC_AUC'] = auc_without_feat
        
        
        
    sd_auc_change = (base_auc-df_raw_roc_score['ROC_AUC']).std()
    
    for feature in df_raw_roc_score.index:
        auc_score = df_raw_roc_score.loc[feature,'ROC_AUC']
        z_score = (base_auc-auc_score)/(sd_auc_change)

        df_z_score.loc[feature,"z-score"] = z_score
        
        
    df_z_score['z-score'] = pd.to_numeric(df_z_score['z-score'], errors='coerce')
    
    return df_z_score

def calculate_dnn_feature_importance(X:pd.DataFrame, y: pd.DataFrame, model, 
                                     do_permutation = False, do_ablation = False, do_shap = False):
    
    df_feature_importance = pd.DataFrame(index=X.columns)
    
    x_tensor = torch.tensor(X.values).to(torch.float32)
    y_tensor = torch.tensor(y.values).to(torch.float32).unsqueeze(1)
    
    if do_permutation:
        
        df_feature_importance['permutation'] = [0 for _ in range(len(X.columns))]
        
        y_baseline_pred = model(x_tensor)
        
        baseline_loss = F.mse_loss(y_tensor, y_baseline_pred).item()
        
        
        for feature in X.columns:
            
            test_x = X.copy()
            np.random.shuffle(test_x[feature].values)

            y_pred = model(torch.tensor(test_x.values).to(torch.float32))
            
            mse_loss = F.mse_loss(y_tensor, y_pred).item()
            
            df_feature_importance.loc[feature, 'permutation'] = (mse_loss-baseline_loss)
            
            
    if do_ablation:
        df_feature_importance['ablation'] = [0 for _ in range(len(X.columns))]
        
        y_baseline_pred = model(x_tensor)
        
        baseline_loss = F.mse_loss(y_tensor, y_baseline_pred).item()
        
        for feature in X.columns:
            test_x = X.copy()
            
            test_x[feature] = test_x[feature].mean()
            y_pred = model(torch.tensor(test_x.values).to(torch.float32))
            
            mse_loss = F.mse_loss(y_tensor, y_pred).item()
            
            df_feature_importance.loc[feature, 'ablation'] = (mse_loss- baseline_loss)
            
    
            
    if do_shap:
        explainer = shap.DeepExplainer(model, x_tensor)
        
        shap_values = explainer.shap_values(x_tensor)

        mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
            
        print(mean_abs_shap_values)
            
        df_feature_importance['shap'] = mean_abs_shap_values
        
        
    return df_feature_importance
    

def graph_nn_model(model, sample_input, filename):
    sample_output = model(sample_input)
    graph = make_dot(sample_output, params=dict(model.named_parameters()))
    graph.render(filename, format="png")
    


if __name__ == '__main__':
    
    random_data = pd.DataFrame({'x1':[random.random() for _ in range(20)],
                                'x2':[random.random() for _ in range(20)],
                                'x3':[random.random() for _ in range(20)],
                                'x4':[random.random() for _ in range(20)],
                                'x5':[random.random() for _ in range(20)],
                                'y':[0 if x < 0.5 else 1 for x in [random.random() for _ in range(20)]]})
    
    x_data = random_data.drop(['y'],axis = 1)
    y_data= random_data['y']

    #assert(isinstance(calculate_feature_importance(x_data,y_data,'./models_made/test/gb'), pd.DataFrame))
    
    sample_model = nn.Sequential(
        nn.Linear(5,3),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(3),
        nn.Linear(3,1),
        
    )
    
    #x_tensor = torch.tensor(x_data.values).to(torch.float32)
    
    #graph_nn_model(sample_model, x_tensor, './test/test_model_image')
    
    feature = calculate_dnn_feature_importance(x_data, y_data, sample_model,
                                               do_permutation=True, do_ablation=True,do_shap=True )
    
    print(feature)