import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import random 

from tqdm import tqdm

import pandas as pd
import pickle
import os

import numpy as np


#kaiming he initialization on linear layer
def he_linear(in_dim, out_dim, nonlinearity = 'leaky_relu'):
    
    #nonlinearity: relu, leaky_relu, swish
    
    linear = nn.Linear(in_dim,out_dim)
    
    init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity=nonlinearity)
    init.constant_(linear.bias, 0.0)
    
    
    linear.weight = linear.weight.to(torch.float32)
    
    return linear



class linear_model(nn.Module):
    
    @staticmethod
    def linear_block(in_dim: int, 
                     out_dim:int,
                     activation:str,
                     dropout_perc:int
                     ):
        
        layer_linear = he_linear(in_dim,out_dim,activation)
        

        
        
        
        if activation == 'relu':
            layer_actf = nn.ReLU(inplace=True)
            
            
        elif activation == 'leaky_relu':
            layer_actf = nn.LeakyReLU(negative_slope=0.01,inplace=True)
            
        layer_dropout = nn.Identity() if (dropout_perc == 0) else nn.Dropout(p=dropout_perc)
        
        
        return nn.Sequential(
            layer_linear,
            layer_actf,
            layer_dropout
            
        )
        
    
    
    def __init__(self,num_features:int,
                 in_channels:int,
                 activation_function:str,
                 dropout_perc: int,
                 hidden_layer_dims: list):
        
        super(linear_model, self).__init__()
        
        self.num_features = num_features
        
        self.layers = nn.ModuleList([])
        
        #initial layer
        self.layers.append(
            self.linear_block(num_features,in_channels,activation_function,dropout_perc)
            
        )
        
        #hidden layers
        for dim in hidden_layer_dims:
            self.layers.append(
                self.linear_block(in_channels,dim,activation_function,dropout_perc)
            )
            
            
            in_channels = dim
            
            
        #output layer - a number spanning from -inf to inf
        output_layer = nn.Linear(in_channels,1)
        
        self.layers.append(
            output_layer
        )
        
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()
    
    
 
    
        
class CustomDataset(Dataset):
    '''
    A custom dataset for pd.DataFrame object that returns the x and y data
    
    '''
    def __init__(self, x_df,y_df):
        self.x_df = x_df
        self.y_df = y_df

    def __len__(self):
        return len(self.x_df)

    def __getitem__(self, idx):

        return (torch.tensor(self.x_df.iloc[idx])  , torch.tensor(self.y_df.iloc[idx]))
        
        
        
        
class random_search:
    
    def __init__(self,
                 device,
                 random_seed: int = None):

        self.device = device
        self.best_params = None
        self.best_performance = float('inf')
        self.best_model =  None
        self.random_seed = random_seed
        

        self.results = {}
        
        if (random_seed is not None):
            random.seed(random_seed) 
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            
    def save(self,dir):
        
        with open(dir,'wb') as w:
            pickle.dump((self.best_params,self.best_performance,self.best_model, self.results,self.random_seed),w)
    
    def load(self,dir):
        with open(dir,'rb') as w:
            self.best_params, self.best_performance, self.best_model,self.results, self.random_seed = pickle.load(w)
    
    def save_best_model(self,dir):
        if self.best_model is not None:
            torch.save(self.best_model, dir)
    
    
    def get_result(self):
        
        if len(self.results) == 0:
            return None
        
        print("Best results:")
        
        self.results = dict(sorted(self.results.items(), key = lambda item: item[1][1]))
        
        for trial_name, content in self.results.items():
            print(trial_name)
            print(f'result: {content[1]}')
            print(f'hyperparameters: {content[0]}')
            print()
            print()
            
        
        
    def search(self, 
               train_data: tuple, #tuple of x and y dataframe
               val_data:tuple, #tuple of x and y dataframe
               test_data: tuple, # tuple of x and y dataframe
               max_iter:int,
               epochs: int,
               search_space: dict,
               batchsize: int = 256,
               tolerance: int = 3,
               
               
               ):

        
        #train_x, train_y should be a pd.DataFrame object
        train_x, train_y = train_data
        
        train_dataset = CustomDataset(train_x,train_y)
        
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        
        
        
        val_x, val_y = val_data
        
        val_dataset = CustomDataset(val_x,val_y)
        
        val_loader = DataLoader(val_dataset, batch_size=batchsize)
        
        
        test_x, test_y = test_data
            
        test_dataset = CustomDataset(test_x,test_y)

        test_loader = DataLoader(test_dataset, batch_size=batchsize)
        
        for i in range(max_iter):
            
            this_hyperparameter = {

            }
            
            this_hyperparameter['learning_rate'] = random.choice(search_space['learning_rate'])

            
            this_hyperparameter['num_layers'] = random.choice(search_space['num_layers'])
            this_hyperparameter['hidden_units_list'] = [random.choice(search_space['hidden_units']) for _ in range(this_hyperparameter['num_layers']) ]
            this_hyperparameter['dropout_perc'] = random.choice(search_space['dropout_perc'])
            this_hyperparameter['activation'] = random.choice(search_space['activation'])
            this_hyperparameter['in_channels'] = random.choice(search_space['in_channels'])
        
            testing_model = linear_model(
                num_features = len(train_x.columns),
                in_channels  = this_hyperparameter['in_channels'],
                activation_function = this_hyperparameter['activation'],
                dropout_perc = this_hyperparameter['dropout_perc'],
                hidden_layer_dims = this_hyperparameter['hidden_units_list']
                
            ).to(self.device)
            
            
            
            
            
            
            
            optimizer = optim.Adam(testing_model.parameters(), lr=this_hyperparameter['learning_rate'])
            criterion = nn.MSELoss()
        

            val_threshold = float('inf')
            strike = 0
            
            

            print(f"Trial: {i}, Params: {this_hyperparameter}")
            
            loop = tqdm(range(epochs), total=len(range(epochs)))
            
            for epoch in loop:
                
                train_total_loss = 0
                val_total_loss = 0

                #training process
                testing_model.train()

                for data in train_loader:
                    inputs, labels = data
                    
                    inputs = inputs.to(torch.float32)
                    labels = labels.float()
                    
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = testing_model(inputs)
                
                    
                    loss = criterion(outputs, labels)
                    
                    train_total_loss += loss.item()
                    
                    loss.backward()
                    optimizer.step()
                    

                #val process
                testing_model.eval()
                with torch.no_grad():
                    for data in val_loader:
                        inputs, labels = data
                        inputs = inputs.to(torch.float32)
                        labels = labels.float()
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        
                        outputs = testing_model(inputs)
                        val_loss = criterion(outputs, labels) 
                        
                        val_total_loss += val_loss.item()
                        
                        
                if val_total_loss < val_threshold: #model is improving according to val
                        val_threshold = val_total_loss
                        strike = 0
                        
                else: #model is not improving according to val
                    strike+=1

                    
                
                loop.set_postfix(
                    training_total_loss = train_total_loss,
                    val_total_loss = val_total_loss,
                    val_threshold = val_threshold,
                    strike = strike
                )
                
                if strike >= tolerance:
                    break
        
        

            #testing phase
            testing_model.eval()
            
            testing_total_loss = 0
            
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data
                    inputs = inputs.to(torch.float32)
                    labels = labels.float()
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = testing_model(inputs)
                    testing_loss = criterion(outputs, labels) 
                    
                    testing_total_loss+= testing_loss.item()
                    
                    
            self.results[f'Trial {i}'] = [this_hyperparameter,testing_total_loss]
            if testing_total_loss < self.best_performance:
                    
                
                
                self.best_model = testing_model
                
                self.best_performance=testing_total_loss
                
            print(f'Testing Total loss: {testing_total_loss}')
            print() 

class ResidualLinearBlock(nn.Module):
    
    @staticmethod
    def return_normalize(normalize, dim = None):
        return nn.InstanceNorm1d(dim) if normalize else nn.Identity()
    
    def __init__(self,in_dim:int, out_dim:int, normalize = True):
        super(ResidualLinearBlock, self ).__init__()
        
        self.layers = nn.Sequential(
            he_linear(in_dim, out_dim),
            nn.GELU(inplace=True),
            self.return_normalize(normalize, out_dim),
            
            he_linear(out_dim, out_dim),
            nn.GELU(inplace=True),
            self.return_normalize(normalize, out_dim),
            
            he_linear(out_dim, out_dim),
            nn.GELU(inplace=True),
            self.return_normalize(normalize, out_dim),
               
        ) 
        
        self.residual_linear = he_linear(in_dim, out_dim)
        
    def forward(self,x):
        x_res = self.residual_linear(x)
        
        return self.layers(x) + x_res
    
    
################################################################################################################################
    
    
class EncoderLikeDNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.norm = nn.InstanceNorm1d(600)
        
        self.encoding_layers = nn.Sequential(
            
            ResidualLinearBlock(600,512, normalize=True),
            ResidualLinearBlock(512,256, normalize=True),
            ResidualLinearBlock(256,128, normalize=True),
            ResidualLinearBlock(128,64, normalize=True),
            ResidualLinearBlock(64,32, normalize=True),
            ResidualLinearBlock(32,16, normalize=True),
        )
        
        self.ffs = nn.Sequential(
            he_linear(16, 16),
            nn.GELU(inplace=True),
            nn.InstanceNorm1d(16),
            
            he_linear(16, 16),
            nn.GELU(inplace=True),
            nn.InstanceNorm1d(16),
            
            he_linear(16, 16),
            nn.GELU(inplace=True),
            nn.InstanceNorm1d(16),
        )
        
        self.residual_linear  = nn.Sequential(
            
            he_linear(600,16),
            nn.GELU(inplace=True),
            nn.InstanceNorm1d(16),
        )
        
        self.final_output = nn.Linear(16,1)
        
    def forward(self, x):
        x = self.norm(x)
        
        x_res = self.residual_linear(x)
        
        x = self.encoding_layers(x)
        x = self.ffs(x) + x_res
        
        return self.final_output(x)
        
        
        
        
        
        
if __name__ == "__main__":
    #testing to see if it works
    random_train_x = pd.DataFrame({'x1':[random.random() for _ in range(20)],
                                'x2':[random.random() for _ in range(20)],
                                'x3':[random.random() for _ in range(20)],
                                'x4':[random.random() for _ in range(20)],
                                'x5':[random.random() for _ in range(20)],
                                
                                })
    random_train_y = pd.DataFrame({
        
        'y':[0 if x < 0.5 else 1 for x in [random.random() for _ in range(20)]]
    })
    
    train_df = (random_train_x,random_train_y)
    
    
    
    random_val_x = pd.DataFrame({'x1':[random.random() for _ in range(20)],
                                'x2':[random.random() for _ in range(20)],
                                'x3':[random.random() for _ in range(20)],
                                'x4':[random.random() for _ in range(20)],
                                'x5':[random.random() for _ in range(20)],
                                
                                })
    random_val_y = pd.DataFrame({
        
        'y':[0 if x < 0.5 else 1 for x in [random.random() for _ in range(20)]]
    })
    
    val_df = (random_val_x,random_val_y)
    
    
    
    
    random_test_x = pd.DataFrame({'x1':[random.random() for _ in range(20)],
                                'x2':[random.random() for _ in range(20)],
                                'x3':[random.random() for _ in range(20)],
                                'x4':[random.random() for _ in range(20)],
                                'x5':[random.random() for _ in range(20)],
                                
                                })
    random_test_y = pd.DataFrame({
        
        'y':[0 if x < 0.5 else 1 for x in [random.random() for _ in range(20)]]
    })
    
    test_df = (random_test_x,random_test_y)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    search_obj = random_search(DEVICE,123)
    
    search_space = {
        'learning_rate': [0.001, 0.01, 0.1],
        'num_layers': [1, 2, 3],
        'hidden_units': [64, 128, 256],
        'dropout_perc': [0,0.5],
        'activation': ['relu','leaky_relu'],
        'in_channels': [32, 64, 128, 256],
        }
    
    search_obj.search(
        train_df,
        val_df,
        test_df,
        max_iter= 10,
        epochs = 5,
        search_space=search_space,
        batchsize=5,
        tolerance = 5
    )
    
    
    search_obj.get_result()
    
    os.makedirs("./test/",exist_ok=True)
    
    search_obj.save('./test/test_search.p')
    search_obj.load('./test/test_search.p')
    
    search_obj.save_best_model('./test/test_best_model.pth')
    
    print("Everything worked successfully")