#importing libraries for implementing sequential models
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
import optuna as opt
from deep_models_without_pad import *
from data_manipulation import *
from contextlib import contextmanager
from multiprocessing import Manager
import gc


#####################################################################################################################################
#this class creates a synchronized way to use all the GPUs
class GPU_Queue(object):
    def __init__(self, n_GPUs):
        self.queue = Manager().Queue()
        #list of ids of all gpus
        all_ids = list(range(n_GPUs)) if n_GPUs > 0 else [None]
        for id_num in all_ids:
            self.queue.put(id_num)
    #overriding it
    @contextmanager
    def one_gpu_perTrial(self):
        current_id_num = self.queue.get()
        #stores the local copy until exec
        yield current_id_num
        self.queue.put(current_id_num)


class Objective(object):
    def __init__(self, gpu_queue:GPU_Queue, vocab_size, train_S1, train_S2, train_y, cv_S1, cv_S2, cv_y, model_name, optim_name, hyp_sugg, hyp_non_opt):
	
        self.gpu_queue = gpu_queue
        self.vocab_size = vocab_size
        self.train_S1 = train_S1
        self.train_S2 = train_S2
        self.train_y = train_y
        self.cv_S1 = cv_S1
        self.cv_S2 = cv_S2
        self.cv_y = cv_y
        self.model_name = model_name
        self.optim_name = optim_name
        self.hyp_sugg = hyp_sugg
        self.hyp_non_opt = hyp_non_opt

    def __call__(self, trial):
        with self.gpu_queue.one_gpu_perTrial() as gpu_pres:
                cv_acc = 0
                #has all the keys to the hyper_parameter suggestion
                model = None
                optimizer = None
    	   	#order of parameters: vocab_size, input_size, hidden_size, fc_size, num_layers, num_labels, dropout_rnn, dropout_fc, bi_direct
                if self.model_name == 'RNN':
                        model = RNN(self.hyp_non_opt['vocab_size'],
                	            trial.suggest_categorical("emb_size", self.hyp_sugg['emb_size']),
                	            trial.suggest_categorical("hidden_size", self.hyp_sugg['hidden_size']),
                                    trial.suggest_categorical("fc_size", self.hyp_sugg['fc_size']),
                                    trial.suggest_categorical("num_layers", self.hyp_sugg['num_layers']),
                                    self.hyp_non_opt['num_labels'],
                                    trial.suggest_float("dropout_rnn", self.hyp_sugg['dropout_rnn'][0], self.hyp_sugg['dropout_rnn'][1]),
                                    trial.suggest_float("dropout_fc", self.hyp_sugg['dropout_fc'][0], self.hyp_sugg['dropout_fc'][1]))
    	  
                elif self.model_name == 'LSTM':
                        model = LSTM(self.hyp_non_opt['vocab_size'],
                                    trial.suggest_categorical("emb_size", self.hyp_sugg['emb_size']),
                                    trial.suggest_categorical("hidden_size", self.hyp_sugg['hidden_size']),
                                    trial.suggest_categorical("fc_size", self.hyp_sugg['fc_size']),
                                    trial.suggest_categorical("num_layers", self.hyp_sugg['num_layers']),
                                    self.hyp_non_opt['num_labels'],
                                    trial.suggest_float("dropout_rnn", self.hyp_sugg['dropout_rnn'][0], self.hyp_sugg['dropout_rnn'][1]),
                                    trial.suggest_float("dropout_fc", self.hyp_sugg['dropout_fc'][0], self.hyp_sugg['dropout_fc'][1]))

	
		#after execution 'model' will have the entire hyperparameter optuna for
                if self.optim_name == "adam":
                        optimizer = Adam(model.parameters(),
                   	                lr=trial.suggest_float("learning_rate", self.hyp_sugg['learning_rate'][0], self.hyp_sugg['learning_rate'][1],  log=True),
                                        weight_decay = trial.suggest_float("l2_reg", self.hyp_sugg['l2_reg'][0], self.hyp_sugg['l2_reg'][1],  log=True))
                elif self.optim_name == "sgd":
                        optimizer = SGD(model.parameters(),
                                       lr=trial.suggest_float("learning_rate", self.hyp_sugg['learning_rate'][0], self.hyp_sugg['learning_rate'][1],  log=True),
                                       weight_decay = trial.suggest_float("l2_reg", self.hyp_sugg['l2_reg'][0], self.hyp_sugg['l2_reg'][1],  log=True))
 	
    	        #compiles the model
                cv_acc = SequentialModel_hp_tuner(self.model_name, model, optimizer, self.train_S1, self.train_S2, self.train_y, self.cv_S1, self.cv_S2, self.cv_y, self.hyp_sugg, self.hyp_non_opt, gpu_pres)
                
                gc.collect()
                return cv_acc
	    
		
######################################################Multiprocessing aid################################################################

#this will act as a general caller for all the Deep Sequential models
#def SequentialModel_hp_tuner(self, model, optimizer, train_S1, train_S2, train_y, cv_S1, cv_S2, cv_y, hyp_sugg, hyp_non_opt, device_id):
  
def SequentialModel_hp_tuner(model_name, model, optimizer, train_S1, train_S2, train_y, cv_S1, cv_S2, cv_y, hyp_sugg, hyp_non_opt, device_id):
  #convert the datasets in the desired form
  train_set = get_dataset(train_S1, train_S2, train_y)
  cv_set = get_dataset(cv_S1, cv_S2, cv_y)
  #sees if cuda device is present
  print(torch.cuda.is_available(), torch.cuda.device_count())
  device = torch.device('cuda:'+str(device_id) if torch.cuda.is_available() else 'cpu')
  print(device)

  #initialize the batch maker and shuffler
  batch_maker = DataLoader(dataset=train_set, batch_size=hyp_non_opt['batch_size'], shuffle=True, drop_last=True)
  batch_maker_cv = DataLoader(dataset=cv_set, batch_size=hyp_non_opt['batch_size'], shuffle=True, drop_last=True)
 
   
  model = model.to(device)
  loss_func = nn.CrossEntropyLoss()
  #running iterative optimization
  for epoch_num in tqdm(range(hyp_non_opt['num_epochs'])):
    #gives us batches of the desired size
    for train_batch_X_S1, train_batch_X_S2, train_batch_y in batch_maker:

      #send the data to the gpu
      train_batch_X_S1, train_batch_X_S2, train_batch_y = train_batch_X_S1.to(device), train_batch_X_S2.to(device), train_batch_y.to(device)
      #feed the data to the network, pytorch doesn't need a0/ c0: automatically starts with 0
      y_hat = model(train_batch_X_S1, train_batch_X_S2)

      mini_batch_loss = loss_func(y_hat, train_batch_y)

      #backpropapagate the loss to learn the model
      optimizer.zero_grad()
      #all the gradients are set to 0 so that it doesn't accumulate (the add up in pytorch)
      mini_batch_loss.backward()
      #compute the gradients and update the weights
      optimizer.step()

    #calculating the results on the cv
  with torch.set_grad_enabled(False):
    n_correct = 0
    for cv_batch_X_S1, cv_batch_X_S2, cv_batch_y in batch_maker_cv:
      cv_batch_X_S1, cv_batch_X_S2, cv_batch_y = cv_batch_X_S1.to(device), cv_batch_X_S2.to(device), cv_batch_y.to(device)
       
      #feed the data to the network
      y_hat = model(cv_batch_X_S1, cv_batch_X_S2)
        
      #property of softmax: max of the labels is selected (it is monotonic)
      #max per column (1st index is the maxval, 2nd is the argmax)
      _, predict_y = torch.max(y_hat.data, 1)
        
      n_correct += ((predict_y == cv_batch_y).sum().item())
       
        
      cv_acc = n_correct/len(cv_y)
      
  return cv_acc

def optimize_hyperparameters(self, vocab_size, train_S1, train_S2, train_y, cv_S1, cv_S2, cv_y, model_name, optim_name, hyp_sugg, hyp_non_opt):
     
    cv_acc = 0
    #has all the keys to the hyper_parameter suggestion
    
    model = None
    optimizer = None
    #order of parameters: vocab_size, input_size, hidden_size, fc_size, num_layers, num_labels, dropout_rnn, dropout_fc, bi_direct
    if model_name == 'RNN':
      model = RNN(hyp_non_opt['vocab_size'], 
                  self.suggest_categorical("emb_size", hyp_sugg['emb_size']), 
                  self.suggest_categorical("hidden_size", hyp_sugg['hidden_size']), 
                  self.suggest_categorical("fc_size", hyp_sugg['fc_size']), 
                  self.suggest_categorical("num_layers", hyp_sugg['num_layers']), 
                  hyp_non_opt['num_labels'], 
                  self.suggest_float("dropout_rnn", hyp_sugg['dropout_rnn'][0], hyp_sugg['dropout_rnn'][1]), 
                  self.suggest_float("dropout_fc", hyp_sugg['dropout_fc'][0], hyp_sugg['dropout_fc'][1]))

    elif model_name == 'LSTM':
      model = LSTM(hyp_non_opt['vocab_size'],
                  self.suggest_categorical("emb_size", hyp_sugg['emb_size']),
                  self.suggest_categorical("hidden_size", hyp_sugg['hidden_size']),
                  self.suggest_categorical("fc_size", hyp_sugg['fc_size']),
                  self.suggest_categorical("num_layers", hyp_sugg['num_layers']),
                  hyp_non_opt['num_labels'],
                  self.suggest_float("dropout_rnn", hyp_sugg['dropout_rnn'][0], hyp_sugg['dropout_rnn'][1]),
                  self.suggest_float("dropout_fc", hyp_sugg['dropout_fc'][0], hyp_sugg['dropout_fc'][1]))

    
    if optim_name == "adam":
      optimizer = Adam(model.parameters(),
                    lr=self.suggest_float("learning_rate", hyp_sugg['learning_rate'][0], hyp_sugg['learning_rate'][0],  log=True),
                    weight_decay = self.suggest_float("l2_reg", hyp_sugg['l2_reg'][0], hyp_sugg['l2_reg'][0],  log=True))
    elif optim_name == "sgd":
      optimizer = SGD(model.parameters(),
                    lr=self.suggest_float("learning_rate", hyp_sugg['learning_rate'][0], hyp_sugg['learning_rate'][1],  log=True),
                    weight_decay = self.suggest_float("l2_reg", hyp_sugg['l2_reg'][0], hyp_sugg['l2_reg'][1],  log=True))

    #after execution 'model' will have the entire hyperparameter optuna form

    #compiles the model
    cv_acc = SequentialModel_hp_tuner(self, model, optimizer, train_S1, train_S2, train_y, cv_S1, cv_S2, cv_y, hyp_sugg, hyp_non_opt)
    gc.collect()

    return cv_acc

#learning rates, choice of optimizers, number of layers, number of units in each layer, regularization
def hyperparameter_tuner(vocab_size, train_S1, train_S2, train_y, cv_S1, cv_S2, cv_y, model_name, hyp_saveloc):

    hyp_sugg = {'learning_rate': [1e-5, 0.1],
                'l2_reg': [0.001, 0.5],
                'dropout_rnn': [0, 0.8],
                'dropout_fc': [0, 0.9],
                'optimizer': ['adam', 'sgd'],
                'num_layers': [2, 3],
                'hidden_size': [128, 256],
                'fc_size': [64, 128], 
                'emb_size': [100, 200] 
                }
    hyp_non_opt = {'vocab_size': vocab_size+1, 
                     'batch_size': 128,
                      'num_labels': 3,
                      'num_epochs': 10
                     }

    for optim_name in hyp_sugg['optimizer']:
      #This is to get exact hyperparameter
      hyperparameters = {}

      #with lambda function: here multi gpu won't work
      #objective = lambda self: optimize_hyperparameters(self,  hyp_non_opt['vocab_size'], train_S1, train_S2, train_y, cv_S1, cv_S2, cv_y, model_name, optim_name, hyp_sugg, hyp_non_opt)
      #with callable class
      
      num_gpu = torch.cuda.device_count()
      print(num_gpu)
      objective = Objective(GPU_Queue(num_gpu), vocab_size, train_S1, train_S2, train_y, cv_S1, cv_S2, cv_y, model_name, optim_name, hyp_sugg, hyp_non_opt)     
      #maximize the cross validation accuracy      
      optimizer = opt.create_study(sampler=opt.samplers.TPESampler(), direction='maximize')
      
      #number of trials = 50, and garbage collection is set to true
      optimizer.optimize(objective, n_trials=60, timeout=None, gc_after_trial=True, n_jobs = num_gpu)
      #The best params class member is a dict of best params
      hyperparameters = optimizer.best_params
      cv_acc_max = optimizer.best_value
      print("for optimizer: {} cv accuracy maximized to {} with following hyperparameters: ".format(optim_name, cv_acc_max), file=hyp_saveloc)
      print(hyperparameters, file=hyp_saveloc)
      print("---------------------------------------------------------------------------", file=hyp_saveloc)
