#importing libraries for implementing sequential models
from deep_models import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

def get_packed_batch(X):
  p_S = []
  #find the length of each sentence and return packed sequence
  l_S = []  
  max_l_S = 0
  for i in range(X.shape[0]):
    temp_s = None
    ctr = X.shape[1]
    for j in range(X.shape[1]):  
      if X[i, j] != 0:
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
  return torch.from_numpy(p_S_r), l_S, max_l_S

#RNNs (Recurrent Neural Network)
class RNN(nn.Module):
  #initializer
  #this contains input_layer size (length of the word embeddings: Here word2vec/embedding layers), 
  def __init__(self, vocab_size, input_size, hidden_size, fc_size, num_layers, num_labels, dropout_rnn, dropout_fc):
    #construct the parent
    super(RNN, self).__init__()
    #all information about the RNN stored
    self.input_size = input_size 
    self.num_layers = num_layers 
    self.hidden_size = hidden_size
    #number of classes in the classification problem
    self.num_labels = num_labels
    #embeddings are of the shape (vocab_size-->150: standard size is 300 for 4B words)
    self.Emb_layer = nn.Embedding(vocab_size, input_size, padding_idx=0)
    #from the torch.nn module we initialize an rnn object (if multiple stacked RNNs are present num_layers is used)
    #batch first is done in accordance to keras convention of (None, sentence length, word_emb-size
    self.rnn_s1 = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rnn)
    self.rnn_s2 = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rnn)
    #concatenate outputs of network 1 and 2 and then define a fully connected layer followed by softmax classification
    #fully connected layer: (taking the input from both the recurrent unit)
    self.fully_connected_1 = nn.Linear(2*hidden_size, fc_size)
    self.fc_dropout = nn.Dropout(dropout_fc)
    self.fully_connected_2 = nn.Linear(fc_size, num_labels)
    self.initialize_weights()

  #idea of py-torch: keep all the components ready, an link them while overriding forward
  #forward propagation definition: process the sentences differently
  def forward(self, s1, len_s1, sort_index_s1, s2, len_s2, sort_index_s2):

    #data for the inputs is of the form (num_layers, batch_size, output_size)
    
    #s1, len_s1, _ = get_packed_batch(s1)
    #s1_tensor = torch.tensor(s1, dtype=torch.long)
    #s2_tensor = torch.tensor(s2, dtype=torch.long)
    #embeddings = self
    s1_emb = self.Emb_layer(s1)
    s1_emb_packed = nn.utils.rnn.pack_padded_sequence(s1_emb, lengths=len_s1, batch_first=True)
    
    #s2, len_s2, _ = get_packed_batch(s2) 
    s2_emb = self.Emb_layer(s2)
    s2_emb_packed = nn.utils.rnn.pack_padded_sequence(s2_emb, lengths=len_s2, batch_first=True)
    
 
    _, h_t = self.rnn_s1(s1_emb_packed)    # a0)
    out_s1 = h_t[-1, sort_index_s1, :]  #reorder
    #storing only the output of the last timestamp
    #out_s1 = out_s1[:, -1, :]
  
    #output is of (batch_size, sentence_l, output) -> just need the last o/p (as a feature for the sentence)
    _, h_t = self.rnn_s1(s2_emb_packed)   #a0)
    #out_s2 = out_s2[:, -1, :]
    out_s2 = h_t[-1, sort_index_s2, :]
    #desired output is of the form (batch_size, (f_s1+f_s2))
    input_fc = torch.cat((out_s1, out_s2), dim=1) 
    #gives y=Ax+b, to be passed through cross entropy loss in pytorch to get softmax classifier
    output_fc1 = func.relu(self.fully_connected_1(input_fc))
    output_dropout = self.fc_dropout(output_fc1)
    output_nn = self.fully_connected_2(output_dropout)
    return output_nn

  def initialize_weights(self): 
    for name, param in self.rnn_s1.named_parameters():
      if 'bias' in name:
        nn.init.constant_(param, 0.0)
      elif 'weight_ih' in name:
        nn.init.xavier_normal_(param)
      elif 'weight_hh' in name:
        nn.init.xavier_normal_(param)

    for name, param in self.rnn_s2.named_parameters():
      if 'bias' in name:
        nn.init.constant_(param, 0.0)
      elif 'weight_ih' in name:
        nn.init.xavier_normal_(param)
      elif 'weight_hh' in name:
        nn.init.xavier_normal_(param)

    for name, param in self.fully_connected_1.named_parameters():
      if 'bias' in name:
        nn.init.constant_(param, 0.0)
      elif 'weight' in name:
        nn.init.xavier_normal_(param)

    for name, param in self.fully_connected_2.named_parameters():
      if 'bias' in name:
        nn.init.constant_(param, 0.0)
      elif 'weight' in name:
        nn.init.xavier_normal_(param)
##########################################################################################################################################################

#LSTM (long short term memory)
class LSTM(nn.Module):
  #initializer
  #this contains input_layer size (length of the word embeddings: Here word2vec/embedding layers),
  def __init__(self, vocab_size, input_size, hidden_size, fc_size, num_layers, num_labels, dropout_rnn, dropout_fc):
    #construct the parent
    super(LSTM, self).__init__()
    #all information about the RNN stored
    self.input_size = input_size
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    #number of classes in the classification problem
    self.num_labels = num_labels
    #embeddings are of the shape (vocab_size-->150: standard size is 300 for 4B words)
    self.Emb_layer = nn.Embedding(vocab_size, input_size, padding_idx=0)
    #from the torch.nn module we initialize an rnn object (if multiple stacked RNNs are present num_layers is used)
    #batch first is done in accordance to keras convention of (None, sentence length, word_emb-size
    self.lstm_s1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rnn)
    self.lstm_s2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rnn)
    #concatenate outputs of network 1 and 2 and then define a fully connected layer followed by softmax classification
    #fully connected layer: (taking the input from both the recurrent unit)
    self.fully_connected_1 = nn.Linear(2*hidden_size, fc_size)
    self.fc_dropout = nn.Dropout(dropout_fc)
    self.fully_connected_2 = nn.Linear(fc_size, num_labels)

    self.initialize_weights()

  #idea of py-torch: keep all the components ready, an link them while overriding forward
  #forward propagation definition: process the sentences differently
  def forward(self, s1, len_s1, sort_index_s1, s2, len_s2, sort_index_s2):
    #data for the inputs is of the form (num_layers, batch_size, output_size)
    #s1_tensor = torch.tensor(s1, dtype=torch.long)
    #s2_tensor = torch.tensor(s2, dtype=torch.long)
    #embeddings = self
    #s1, len_s1, _ = get_packed_batch(s1)
    #s1_tensor = torch.tensor(s1, dtype=torch.long)
    #s2_tensor = torch.tensor(s2, dtype=torch.long)
    #embeddings = self
    s1_emb = self.Emb_layer(s1)
    s1_emb_packed = nn.utils.rnn.pack_padded_sequence(s1_emb, lengths=len_s1, batch_first=True)

    #s2, len_s2, _ = get_packed_batch(s2)
    s2_emb = self.Emb_layer(s2)
    s2_emb_packed = nn.utils.rnn.pack_padded_sequence(s2_emb, lengths=len_s2, batch_first=True)


    _, (h_t, c_t) = self.lstm_s1(s1_emb_packed)    # a0)
    out_s1 = h_t[-1, sort_index_s1, :]  #reorders properly
    #storing only the output of the last timestamp
    #out_s1 = out_s1[:, -1, :]

    #output is of (batch_size, sentence_l, output) -> just need the last o/p (as a feature for the sentence)
    _, (h_t, c_t) = self.lstm_s1(s2_emb_packed)   #a0)
    #out_s2 = out_s2[:, -1, :]
    out_s2 = h_t[-1, sort_index_s2, :]
    
    
    input_fc = torch.cat((out_s1, out_s2), dim=1)
    #print(input_fc.shape)
    #gives y=Ax+b, to be passed through cross entropy loss in pytorch to get softmax classifier
    output_fc1 = func.relu(self.fully_connected_1(input_fc))
    #print(output_fc1.shape) 
    output_dropout = self.fc_dropout(output_fc1)
    #print(output_dropout.shape)
    output_nn = self.fully_connected_2(output_dropout)
    #print(output_nn.shape)
    return output_nn
 
  def initialize_weights(self):
    for name, param in self.lstm_s1.named_parameters():
      if 'bias' in name:
        nn.init.constant_(param, 0.0)
      elif 'weight_ih' in name:
        nn.init.xavier_normal_(param)
      elif 'weight_hh' in name:
        nn.init.xavier_normal_(param)

    for name, param in self.lstm_s2.named_parameters():
      if 'bias' in name:
        nn.init.constant_(param, 0.0)
      elif 'weight_ih' in name:
        nn.init.xavier_normal_(param)
      elif 'weight_hh' in name:
        nn.init.xavier_normal_(param)

    for name, param in self.fully_connected_1.named_parameters():
      if 'bias' in name:
        nn.init.constant_(param, 0.0)
      elif 'weight' in name:
        nn.init.xavier_normal_(param)
    
    for name, param in self.fully_connected_2.named_parameters():
      if 'bias' in name:
        nn.init.constant_(param, 0.0)
      elif 'weight' in name:
        nn.init.xavier_normal_(param)

##########################################################################################################################################################
