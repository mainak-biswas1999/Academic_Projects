import deep_models as dm
import deep_models_without_pad as dm2

import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
#importing the jsonl files
import jsonlines
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import matplotlib.pyplot as plt



def get_packed_batch(X):
  p_S = []
  #find the length of each sentence and return packed sequence
  l_S = []
  max_l_S = 0
  for i in range(X.shape[0]):
    temp_s = None
    ctr = X.shape[1]
    for j in range(X.shape[1]):
      if X[i, j] != 0 or j == X.shape[1]-1:
        p_S.append(X[i, j:].numpy())
        l_S.append(ctr)
        if ctr>max_l_S:
          max_l_S = ctr
        break
      else:
        ctr -= 1
   
  p_S_r = np.array([]).reshape((0, max_l_S))
  for i in range(len(p_S)):
    t = np.concatenate((p_S[i], np.zeros(max_l_S - l_S[i]))).reshape((1, max_l_S))
    p_S_r = np.append(p_S_r, t, axis=0)

  l_S = np.array(l_S)
  #returns the indices of lengths, for descending order negating it 
  sort_index = np.argsort(-l_S, kind='stable')
  
  l_S = l_S[sort_index]
  p_S_r = p_S_r[sort_index, :]
  return torch.from_numpy(p_S_r).int(), l_S, np.argsort(sort_index, kind='stable'), max_l_S


def plot_curve2(pts1, pts2, x_label, y_label, title, legend, plt_loc):
  plt.clf()
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)

  index = np.arange(1, len(pts1)+1, 1)
  plt.plot(index, pts1, '-b', label=legend[0])
  plt.plot(index, pts2, '-r', label=legend[1])

  plt.legend()
  #plt.show()
  plt.savefig(plt_loc)


class get_dataset(Dataset):
  def __init__(self, s1, s2, y):
    self.S1 = s1
    self.S2 = s2
    self.y = y 
  def __len__(self):
    return len(self.y)

  def __getitem__(self, index):
    #overriding this function to get batch
    #or do some preprocessing like converting to word to vec/embeddig layer
    S1 = self.S1[index]
    S2 = self.S2[index]
    y = self.y[index]
    return S1, S2, y

#this will act as a general caller for all the Deep Sequential models
def SequentialModel_Driver(model_name, hyperparameters, train_S1, train_S2, train_y, cv_S1, cv_S2, cv_y, test_S1, test_S2, test_y, model_saveloc, test_result_loc, S1_all, S2_all, plt_saveloc, is_padded = False, embedding_w=None, is_trainable=True):    
  #convert the datasets in the desired form
  train_set = get_dataset(train_S1, train_S2, train_y)
  cv_set = get_dataset(cv_S1, cv_S2, cv_y)
  test_set = get_dataset(test_S1, test_S2, test_y)
  #sees if cuda device is present
  print(torch.cuda.is_available())
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  #initialize the batch maker and shuffler
  batch_maker = DataLoader(dataset=train_set, batch_size=hyperparameters['batch_size'], shuffle=True) #drop_last=True)
  batch_maker_cv = DataLoader(dataset=cv_set, batch_size=hyperparameters['batch_size'], shuffle=True) #drop_last=True)
  #initialize the model
  model = None
  #(self, vocab_size, input_size, hidden_size, fc_size, num_layers, num_labels, dropout_rnn, dropout_fc)
  if model_name == "RNN":
    #+1 for unknown words
    if is_padded == True:
      model = dm.RNN(hyperparameters['vocab_size'], hyperparameters['emb_size'], hyperparameters['hidden_size'], hyperparameters['fc_size'], hyperparameters['num_layers'], hyperparameters['num_labels'],  hyperparameters['dropout_rnn'], hyperparameters['dropout_fc'])
    else:
      model = dm2.RNN(hyperparameters['vocab_size'], hyperparameters['emb_size'], hyperparameters['hidden_size'], hyperparameters['fc_size'], hyperparameters['num_layers'], hyperparameters['num_labels'],  hyperparameters['dropout_rnn'], hyperparameters['dropout_fc'])

  elif model_name == "LSTM":
    #+1 for unknown words
    if is_padded == True: 
      model = dm.LSTM(hyperparameters['vocab_size'], hyperparameters['emb_size'], hyperparameters['hidden_size'], hyperparameters['fc_size'], hyperparameters['num_layers'], hyperparameters['num_labels'], hyperparameters['dropout_rnn'], hyperparameters['dropout_fc'])
    
    else:
      model = dm2.LSTM(hyperparameters['vocab_size'], hyperparameters['emb_size'], hyperparameters['hidden_size'], hyperparameters['fc_size'], hyperparameters['num_layers'], hyperparameters['num_labels'], hyperparameters['dropout_rnn'], hyperparameters['dropout_fc'], embedding_w, is_trainable)
  
  elif model_name == "GRU":
    model = dm2.GRU(hyperparameters['vocab_size'], hyperparameters['emb_size'], hyperparameters['hidden_size'], hyperparameters['fc_size'], hyperparameters['num_layers'], hyperparameters['num_labels'], hyperparameters['dropout_rnn'], hyperparameters['dropout_fc'])
  model = model.to(device)
  
  #a0 = torch.zeros(hyperparameters['num_layers'], hyperparameters['batch_size'], hyperparameters['hidden_size'], device=device)
  #a0 = a0.to(device)
  #initialize the hyperparameters
  optimizer = None
  loss_func = nn.CrossEntropyLoss()
  if hyperparameters['optimizer'] == "adam":
   optimizer = Adam(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['l2_reg'])
  elif hyperparameters['optimizer'] == "sgd":
   optimizer = SGD(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['l2_reg'])
  
  train_acc = []
  train_loss = []
  cv_accuracy = []
  cv_loss = []
  #running iterative optimization
  for epoch_num in tqdm(range(hyperparameters['num_epochs'])):
    #gives us batches of the desired size
    model.train()
    for train_batch_X_S1, train_batch_X_S2, train_batch_y in batch_maker:
      y_hat = None
      if is_padded == True:
        train_batch_X_S1, len_S1, sort_index_S1, _ = get_packed_batch(train_batch_X_S1)
        train_batch_X_S2, len_S2, sort_index_S2, _ = get_packed_batch(train_batch_X_S2) 
        #send the data to the gpu
        train_batch_X_S1, train_batch_X_S2, train_batch_y = train_batch_X_S1.to(device), train_batch_X_S2.to(device), train_batch_y.to(device)
        #len_S1, len_S2 = len_S1.to(device), len_S2.to(device)
        #feed the data to the network
        y_hat = model(train_batch_X_S1, len_S1, sort_index_S1, train_batch_X_S2, len_S2, sort_index_S2)
      
      else:
        train_batch_X_S1, train_batch_X_S2, train_batch_y = train_batch_X_S1.to(device), train_batch_X_S2.to(device), train_batch_y.to(device)
        y_hat = model(train_batch_X_S1, train_batch_X_S2)
     
      mini_batch_loss = loss_func(y_hat, train_batch_y)

      #backpropapagate the loss to learn the model
      optimizer.zero_grad()
      #all the gradients are set to 0 so that it doesn't accumulate (the add up in pytorch)
      mini_batch_loss.backward()
      #compute the gradients and update the weights
      optimizer.step()

    model.eval()
    #calculating the results on the cv set and train set after each epoch
    with torch.set_grad_enabled(False):
      loss = 0
      n_correct = 0
      for cv_batch_X_S1, cv_batch_X_S2, cv_batch_y in batch_maker_cv:

        y_hat = None
        if is_padded == True:
          cv_batch_X_S1, len_S1, sort_index_S1, _ = get_packed_batch(cv_batch_X_S1)
          cv_batch_X_S2, len_S2, sort_index_S2, _ = get_packed_batch(cv_batch_X_S2)
          #send the data to the gpu
          cv_batch_X_S1, cv_batch_X_S2, cv_batch_y = cv_X_S1.to(device), cv_X_S2.to(device), cv_y.to(device)
          #len_S1, len_S2 = len_S1.to(device), len_S2.to(device)
          #feed the data to the network
          y_hat = model(cv_batch_X_S1, len_S1, sort_index_S1, cv_batch_X_S2, len_S2, sort_index_S2)

        else:
          cv_batch_X_S1, cv_batch_X_S2, cv_batch_y = cv_batch_X_S1.to(device), cv_batch_X_S2.to(device), cv_batch_y.to(device)
          y_hat = model(cv_batch_X_S1, cv_batch_X_S2)
 

        mini_batch_loss = loss_func(y_hat, cv_batch_y)
        loss += mini_batch_loss*(cv_batch_y.size(0)/(len(cv_y)))
        #property of softmax: max of the labels is selected (it is monotonic)
        #max per column (1st index is the maxval, 2nd is the argmax)
        _, predict_y = torch.max(y_hat.data, 1)
        
        n_correct += ((predict_y == cv_batch_y).sum().item())
       
        
      cv_accuracy.append(n_correct/len(cv_y))
      cv_loss.append(loss)

      loss = 0
      n_correct = 0
      for train_batch_X_S1, train_batch_X_S2, train_batch_y in batch_maker:
        y_hat = None
        if is_padded == True:
          train_batch_X_S1, len_S1, sort_index_S1, _ = get_packed_batch(train_batch_X_S1)
          train_batch_X_S2, len_S2, sort_index_S2, _ = get_packed_batch(train_batch_X_S2)
          #send the data to the gpu
          train_batch_X_S1, train_batch_X_S2, train_batch_y = train_batch_X_S1.to(device), train_batch_X_S2.to(device), train_batch_y.to(device)
          #len_S1, len_S2 = len_S1.to(device), len_S2.to(device)
          #feed the data to the network
          y_hat = model(train_batch_X_S1, len_S1, sort_index_S1, train_batch_X_S2, len_S2, sort_index_S2)

        else:
          train_batch_X_S1, train_batch_X_S2, train_batch_y = train_batch_X_S1.to(device), train_batch_X_S2.to(device), train_batch_y.to(device)
          y_hat = model(train_batch_X_S1, train_batch_X_S2)
        
        
        mini_batch_loss = loss_func(y_hat, train_batch_y)
        loss += mini_batch_loss*(train_batch_y.size(0)/(len(train_y)))

        _, predict_y = torch.max(y_hat.data, 1)
        n_correct += ((predict_y == train_batch_y).sum().item())
        
      train_acc.append(n_correct/len(train_y))
      train_loss.append(loss)
    if epoch_num%10 == 0:
      print("Train acc: {}, cv acc: {}".format(train_acc[-1], cv_accuracy[-1]), flush=True)	  
     
  torch.save(model, model_saveloc)
  model = torch.load(model_saveloc)
  
  test_acc = test_model(model, hyperparameters, test_S1, test_S2, test_y, test_result_loc, device, S1_all, S2_all, is_padded) 
  print("Train Accuracy: {}  CV accuracy: {} Test accuracy: {}".format(train_acc[-1], cv_accuracy[-1], test_acc), flush=True)
  print("Train Accuracy: {}  CV accuracy: {} Test accuracy: {}".format(train_acc[-1], cv_accuracy[-1], test_acc), file=test_result_loc, flush=True)
  
  plot_curve2(train_loss, cv_loss, "epoch", "Loss", "Loss curve over Epoch", ['train loss', 'cv loss'], plt_saveloc+'loss_curve.png')
  plot_curve2(train_acc, cv_accuracy,"epoch", "Accuracy", "Learning Curve", ['train accuracy', 'cv accuracy'], plt_saveloc+"lr_curve.png")
  

def test_model(model,  hyperparameters, test_S1, test_S2, test_y, result_loc, device, S1_all, S2_all, is_padded):
  map_output = {}
  map_output[0] = 'entailment'
  map_output[1] = 'contradiction'
  map_output[2] = 'neutral'
  test_set = get_dataset(test_S1, test_S2, test_y)  
  batch_maker_test = DataLoader(dataset=test_set, batch_size=hyperparameters['batch_size'], shuffle=False) #drop_last=True)
  test_acc = [] 
 
  model.eval() 
  with torch.set_grad_enabled(False):
      n_correct = 0
      ctr = 0
      for cv_batch_X_S1, cv_batch_X_S2, cv_batch_y in batch_maker_test:
        y_hat = None
        if is_padded == True:
          cv_batch_X_S1, len_S1, sort_index_S1, _ = get_packed_batch(cv_batch_X_S1)
          cv_batch_X_S2, len_S2, sort_index_S2, _ = get_packed_batch(cv_batch_X_S2)
          #send the data to the gpu
          cv_batch_X_S1, cv_batch_X_S2, cv_batch_y = cv_X_S1.to(device), cv_X_S2.to(device), cv_y.to(device)
          #len_S1, len_S2 = len_S1.to(device), len_S2.to(device)
          #feed the data to the network
          y_hat = model(cv_batch_X_S1, len_S1, sort_index_S1, cv_batch_X_S2, len_S2, sort_index_S2)

        else:
          cv_batch_X_S1, cv_batch_X_S2, cv_batch_y = cv_batch_X_S1.to(device), cv_batch_X_S2.to(device), cv_batch_y.to(device)
          y_hat = model(cv_batch_X_S1, cv_batch_X_S2)


        #property of softmax: max of the labels is selected (it is monotonic)
        #max per column (1st index is the maxval, 2nd is the argmax)
        _, predict_y = torch.max(y_hat.data, 1)

        n_correct += ((predict_y == cv_batch_y).sum().item())
	
        for i in range(len(predict_y)):
          print("Test id: {} \n S1: {} \n S2: {} \n Actual label: {}       Predicted label: {}  \n-------------------------------------------------------------------------------------------------------".format(ctr*hyperparameters['batch_size']+i, S1_all[ctr*hyperparameters['batch_size']+i], S2_all[ctr*hyperparameters['batch_size']+i], map_output[cv_batch_y[i].item()], map_output[predict_y[i].item()]), file=result_loc, flush=True)	  
        ctr += 1	
      test_acc = (n_correct/len(test_y))

  return test_acc
