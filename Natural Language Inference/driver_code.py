import numpy as np
import torch
import jsonlines
#functions to read dataset
def getDataSet_from_jsonl(dataloc):
  #dataset extraction from jsonl files (fields extracted): gold_label (neutral, contradiction, neutral), sentence1 (this is the premise, is assumed to be true), sentence2 (this is the hypothesis: to see if it matches the premise)
  #will extract from the jsonfile and entire dictionary of dictionaries (each internal dictionary will have an example) 
  data_dict = {}
  S1_all = []
  S2_all = []
  with jsonlines.open(dataloc) as fptr:
    for line in fptr.iter():
      #reads the data line by line and returns a dictionary for each line
      #gold_label is the label chosen by majority of annotators (there are some ambigious one where gold label is "-": needs to be excluded)
      if line['gold_label'] != '-':
        S1_all.append(line['sentence1'])
        S2_all.append(line['sentence2']) 

  return S1_all, S2_all



train_S1 = torch.from_numpy(np.load('/home/mainakbiswas/database_snli/train_S1.npy'))
train_S2 = torch.from_numpy(np.load('/home/mainakbiswas/database_snli/train_S2.npy'))
train_y = torch.from_numpy(np.load('/home/mainakbiswas/database_snli/train_y.npy'))
cv_S1 = torch.from_numpy(np.load('/home/mainakbiswas/database_snli/cv_S1.npy'))
cv_S2 = torch.from_numpy(np.load('/home/mainakbiswas/database_snli/cv_S2.npy'))
cv_y = torch.from_numpy(np.load('/home/mainakbiswas/database_snli/cv_y.npy'))
test_S1 = torch.from_numpy(np.load('/home/mainakbiswas/database_snli/test_S1.npy'))
test_S2 = torch.from_numpy(np.load('/home/mainakbiswas/database_snli/test_S2.npy'))
test_y = torch.from_numpy(np.load('/home/mainakbiswas/database_snli/test_y.npy'))
max_index = 24448

##################Tuning###############################################
#print(train_loss)
#print(train_acc)
#print(cv_loss)
#print(cv_accuracy)


#from hyperparameter_tuning import *
#hyp_saveloc = open("/home/mainakbiswas/results/RNN/tuned_hyperparameters_adam.txt", 'w')
#hyperparameter_tuner(max_index, train_S1, train_S2, train_y, cv_S1, cv_S2, cv_y, "RNN", hyp_saveloc)
#hyp_saveloc.close()


#################################testing and training##################
from data_manipulation import *
hyperparameters = {'learning_rate': 0.0005281647879556434,
                'l2_reg': 0.000054793046215121786,
                'dropout_rnn':  0.00029220761733241096,
                'dropout_fc': 0.00033489402875076857,
                'optimizer': 'adam',
                'num_layers': 3,
                'hidden_size': 256,
                'fc_size': 128,
                'emb_size': 200,
                'vocab_size': max_index+1,
                'batch_size': 128,
                'num_labels': 3,
                'num_epochs': 50
            }
S1_all, S2_all = getDataSet_from_jsonl('/home/mainakbiswas/database_snli/snli_1.0_test.jsonl')
result_dir = "/home/mainakbiswas/results/RNN/final_train_latest/"
test_result_loc = open(result_dir+"results_RNN.txt", 'w')
#(model_name, hyperparameters, train_S1, train_S2, train_y, cv_S1, cv_S2, cv_y, test_S1, test_S2, test_y, model_saveloc, test_result_loc, S1_all, S2_all, plt_saveloc
SequentialModel_Driver("RNN", hyperparameters, train_S1, train_S2, train_y, cv_S1, cv_S2, cv_y, test_S1, test_S2, test_y, result_dir+"model_RNN.pt", test_result_loc, S1_all, S2_all, result_dir+"plots", False)
test_result_loc.close()
