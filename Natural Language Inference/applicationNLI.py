import torch
import numpy as np
import pickle
import nltk
import string
from nltk.tokenize import word_tokenize

class ApplicationNLI(object):
	def __init__(self, model_loc, vocab_loc):
		#parameters required
		self.model_loc = model_loc
		self.vocab_loc = vocab_loc
		#parameters to be initialized
		self.labels = {0: "entailment",
			       1: "contradiction",
			       2: "neutral"}
		self.model = None
		self.vocab_dict = None

		self.premise = None
		self.hypothesis = None
		self.prediction = None
		self.load()

	def load(self):
		device = torch.device('cpu')
		self.model = torch.load(self.model_loc).to(device)
		self.model.eval()   #evaluation setup
		with open(self.vocab_loc, 'rb') as file_ptr:
			self.vocab_dict = pickle.load(file_ptr)

	def accept_strings(self):
		self.premise = input("Enter premise: ")
		self.hypothesis = input("Enter hypothesis: ")
	
	def get_one_hot_Enc(self, sentence):
		index = []
		stopwords = nltk.corpus.stopwords.words('english')
		l1 = 0
		s1 = word_tokenize(sentence.lower())
   
		for j in range(len(s1)):
			if (s1[j] not in string.punctuation) and (s1[j] not in stopwords):
				#adds the key for each word
				l1 += 1 
				try:
					index.append(self.vocab_dict[s1[j]])
				except:
					index.append(0)
		if l1 == 0:
			index.append(0)
			l1 = 1

		return np.array(index).reshape(1, l1)

	def get_Encoding(self):
		premise_enc, hypo_enc = None, None
		premise_enc = torch.from_numpy(self.get_one_hot_Enc(self.premise))
		hypo_enc = torch.from_numpy(self.get_one_hot_Enc(self.hypothesis))

		return premise_enc, hypo_enc

	def make_prediction(self):
		premise_enc, hypo_enc = self.get_Encoding() 
		#predict
		_, max_label = torch.max(self.model(premise_enc, hypo_enc), 1)
		self.prediction = self.labels[max_label.item()]
	
	def display_result(self):
		print("This is a: {}".format(self.prediction))

	def run_application(self):
		cont = True
		while cont:
			#accepts data
			self.accept_strings()
			#give the opinion
			self.make_prediction()
			#print
			self.display_result()

			to_cont = input("Press y/Y to continue, any other key to exit: ")
			if to_cont == 'y' or to_cont == 'Y':
				cont = True
			else:
				cont = False


if __name__ == "__main__":
	app = ApplicationNLI("/home/mainakbiswas/Project_final_mainak/results/pretrained/no_tuning_early_stop/model_LSTM_early_untuned.pt", "/home/mainakbiswas/Project_final_mainak/database_snli/pretrained_emb_data/vocab_dict")
	app.run_application()
