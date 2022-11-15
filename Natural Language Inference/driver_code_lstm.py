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



train_S1 = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/train_S1.npy'))
train_S2 = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/train_S2.npy'))
train_y = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/train_y.npy'))
cv_S1 = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/cv_S1.npy'))
cv_S2 = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/cv_S2.npy'))
cv_y = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/cv_y.npy'))
test_S1 = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/test_S1.npy'))
test_S2 = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/test_S2.npy'))
test_y = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/test_y.npy'))
max_index = 24448



#train_S1 = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/pretrained_emb_data/train_S1.npy'))
#train_S2 = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/pretrained_emb_data/train_S2.npy'))
#train_y = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/pretrained_emb_data/train_y.npy'))
#cv_S1 = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/pretrained_emb_data/cv_S1.npy'))
#cv_S2 = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/pretrained_emb_data/cv_S2.npy'))
#cv_y = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/pretrained_emb_data/cv_y.npy'))
#test_S1 = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/pretrained_emb_data/test_S1.npy'))
#test_S2 = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/pretrained_emb_data/test_S2.npy'))
#test_y = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/pretrained_emb_data/test_y.npy'))
#embedding_matrix = torch.from_numpy(np.load('/home/mainakbiswas/Project_final_mainak/database_snli/pretrained_emb_data/embedding_weights.npy')).float()
#max_index = 36785
#################################testing and training##################
#{'emb_size': 100, 'hidden_size': 256, 'fc_size': 128, 'num_layers': 2, 'dropout_rnn':  0.00149272402480633876, 'dropout_fc': 0.00157838795346678, 'learning_rate': 2.0428471092763382e-04, 'l2_reg': 0.00005176383818158911}
from data_manipulation import *
hyperparameters = {'learning_rate': 2.0428471092763382e-04,
                'l2_reg': 0.00005176383818158911,
                'dropout_rnn':  0.00149272402480633876,
                'dropout_fc': 0.0015783879534667837,
                'optimizer': 'adam',
                'num_layers': 2,
                'hidden_size': 256,
                'fc_size': 128,
                'emb_size': 400,
                'vocab_size': max_index+1,
                'batch_size': 128,
                'num_labels': 3,
                'num_epochs': 1
            }
S1_all, S2_all = getDataSet_from_jsonl('/home/mainakbiswas/Project_final_mainak/database_snli/snli_1.0_test.jsonl')
#result_dir = "/home/mainakbiswas/Project_final_mainak/results/pretrained/no_tuning_early_stop/"
result_dir = "/home/mainakbiswas/Project_final_mainak/results/demo/"
test_result_loc = open(result_dir+"results_LSTM.txt", 'w')

#SequentialModel_Driver("LSTM", hyperparameters, train_S1, train_S2, train_y, cv_S1, cv_S2, cv_y, test_S1, test_S2, test_y, result_dir+"model_LSTM_early_untuned.pt", test_result_loc, S1_all, S2_all, result_dir+"plots", False, embedding_matrix, False)

SequentialModel_Driver("LSTM", hyperparameters, train_S1, train_S2, train_y, cv_S1, cv_S2, cv_y, test_S1, test_S2, test_y, result_dir+"model_LSTM.pt", test_result_loc, S1_all, S2_all, result_dir+"plots", False, None, True)

test_result_loc.close()



#################################################################tuning case##############################################################################################

#from hyperparameter_tuning_lstm import *
#hyp_saveloc = open("/home/mainakbiswas/Project_final_mainak/results/demo/tuned_hyperparameters.txt", 'w')
#hyperparameter_tuner(max_index, train_S1, train_S2, train_y, cv_S1, cv_S2, cv_y, "LSTM", hyp_saveloc)
#hyp_saveloc.close()
