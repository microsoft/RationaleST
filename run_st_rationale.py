"""
Author: Meghana Bhat (bhat.89@osu.edu)
Code for Self-training for Rationale using few-shot learning.
This code base is adapted from UST (https://github.com/microsoft/UST)
"""

from huggingface_utils import MODELS
from preprocessing import generate_rationale_data
from sklearn.utils import shuffle
from transformers import *
from st_rationale import train_model

import argparse
import logging
import numpy as np
import os
import random
import sys

# logging
logger = logging.getLogger('STRationale')
logging.basicConfig(level = logging.INFO)

GLOBAL_SEED = int(os.getenv("PYTHONHASHSEED"))
logger.info ("Global seed {}".format(GLOBAL_SEED))

if __name__ == '__main__':

	# construct the argument parse and parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--task", required=True, help="path of the task directory containing train, test and unlabeled data files")
	parser.add_argument("--model_dir", required=True, help="path to store model files")
	parser.add_argument("--seq_len", required=True, type=int, help="sequence length")
	parser.add_argument("--sup_batch_size", nargs="?", type=int, default=4, help="batch size for fine-tuning base model")
	parser.add_argument("--unsup_batch_size", nargs="?", type=int, default=4, help="batch size for self-training on pseudo-labeled data")
	parser.add_argument("--sample_size", nargs="?", type=int, default=16384, help="number of unlabeled samples for evaluating uncetainty on in each self-training iteration")
	parser.add_argument("--unsup_size", nargs="?", type=int, default=4096, help="number of pseudo-labeled instances drawn from sample_size and used in each self-training iteration")
	parser.add_argument("--sample_scheme", nargs="?", default="easy_bald_class_conf", help="Sampling scheme to use")
	parser.add_argument("--sup_labels", nargs="?", type=int, default=60, help="number of labeled samples per class for training and validation (total)")
	parser.add_argument("--T", nargs="?", type=int, default=30, help="number of masked models for uncertainty estimation")
	parser.add_argument("--alpha", nargs="?", type=float, default=0.1, help="hyper-parameter for confident training loss")
	parser.add_argument("--valid_split", nargs="?", type=float, default=0.5, help="percentage of sup_labels to use for validation for each class")
	parser.add_argument("--sup_epochs", nargs="?", type=int, default=70, help="number of epochs for fine-tuning base model")
	parser.add_argument("--unsup_epochs", nargs="?", type=int, default=10, help="number of self-training iterations")
	parser.add_argument("--N_base", nargs="?", type=int, default=5, help="number of times to randomly initialize and fine-tune few-shot base encoder to select the best starting configuration")
	parser.add_argument("--pt_teacher", nargs="?", default="TFBertModel",help="Pre-trained teacher model")
	parser.add_argument("--pt_teacher_checkpoint", nargs="?", default="bert-base-uncased", help="teacher model checkpoint to load pre-trained weights")
	parser.add_argument("--do_pairwise", action="store_true", default=False, help="whether to perform pairwise classification tasks like MNLI")
	parser.add_argument("--hidden_dropout_prob", nargs="?", type=float, default=0.2, help="dropout probability for hidden layer of teacher model")
	parser.add_argument("--attention_probs_dropout_prob", nargs="?", type=float, default=0.2, help="dropout probability for attention layer of teacher model")
	parser.add_argument("--dense_dropout", nargs="?", type=float, default=0.5, help="dropout probability for final layers of teacher model")
	parser.add_argument("--model_type", nargs="?", default="token", help="Toekn classification or MTL?")
	args = vars(parser.parse_args())
	logger.info(args)

	task_name = args["task"]
	max_seq_length = args["seq_len"]
	sup_batch_size = args["sup_batch_size"]
	unsup_batch_size = args["unsup_batch_size"]
	unsup_size = args["unsup_size"]
	sample_size = args["sample_size"]
	model_dir = args["model_dir"]
	sample_scheme = args["sample_scheme"]
	sup_labels = args["sup_labels"]
	T = args["T"]
	alpha = args["alpha"]
	valid_split = args["valid_split"]
	sup_epochs = args["sup_epochs"]
	unsup_epochs = args["unsup_epochs"]
	N_base = args["N_base"]
	pt_teacher = args["pt_teacher"]
	pt_teacher_checkpoint = args["pt_teacher_checkpoint"]
	do_pairwise = args["do_pairwise"]
	dense_dropout = args["dense_dropout"]
	attention_probs_dropout_prob = args["attention_probs_dropout_prob"]
	hidden_dropout_prob = args["hidden_dropout_prob"]
	type_ = args["model_type"]

	#create output directory
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	#Get pre-trained model, tokenizer and config
	for indx, model in enumerate(MODELS):
		if model[0].__name__ == pt_teacher:
			TFModel, Tokenizer, Config = MODELS[indx]

	#get pre-trained tokenizer
	tokenizer = Tokenizer.from_pretrained(pt_teacher_checkpoint, from_pt=True)
	fl_loss = True if 'focal' in type_ else False

	X_train_all, y_train_all, _ = generate_rationale_data(max_seq_length, task_name, "train" ,tokenizer, unlabeled=False, do_pairwise=do_pairwise, type_=type_, fl_loss=fl_loss, task=task_name)
	X_val_all, y_val_all, _ = generate_rationale_data(max_seq_length, task_name, "val" ,tokenizer, unlabeled=False, do_pairwise=do_pairwise, type_=type_, fl_loss=fl_loss, task=task_name)
	X_test, y_test, test_data = generate_rationale_data(max_seq_length, task_name, "test", tokenizer, unlabeled=False, do_pairwise=do_pairwise, type_=type_, fl_loss=fl_loss, task=task_name)
	
	X_unlabeled, unlabeled_data = dict(X_train_all), [[-1]*len(X_train_all['input_ids'][0]) for i in range(len(y_train_all))]
	X_unlabeled['input_ids'] = X_unlabeled['input_ids'] #[:100000]
	X_unlabeled['token_type_ids'] = X_unlabeled['token_type_ids'] #[:100000]
	X_unlabeled['attention_mask'] = X_unlabeled['attention_mask'] #[:100000] #np.zeros((len(X_unlabeled['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_unlabeled))]
	
	if 'joint' in type_:
		X_train_all['input_ids_r'] = X_train_all['input_ids']
		X_train_all['token_type_ids_r'] = X_train_all['token_type_ids']
		X_train_all['attention_mask_r'] = np.zeros((len(X_train_all['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_train_all))]

		X_val_all['input_ids_r'] = X_val_all['input_ids']
		X_val_all['token_type_ids_r'] = X_val_all['token_type_ids']
		X_val_all['attention_mask_r'] = np.zeros((len(X_val_all['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_train_all))]

		X_test['input_ids_r'] = X_test['input_ids']
		X_test['token_type_ids_r'] = X_test['token_type_ids']
		X_test['attention_mask_r'] = np.zeros((len(X_test['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_test))]
              
		X_unlabeled['input_ids_r'] = X_unlabeled['input_ids']
		X_unlabeled['token_type_ids_r'] = X_unlabeled['token_type_ids']
		X_unlabeled['attention_mask_r'] = np.zeros((len(X_unlabeled['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_unlabeled))]
		logger.info(X_train_all['attention_mask_r'].shape) 

	if '_neg' in type_ :
		X_train_all['input_ids_neg'] = X_train_all['input_ids']
		X_train_all['token_type_ids_neg'] = X_train_all['token_type_ids']
		X_train_all['attention_mask_neg'] = np.zeros((len(X_train_all['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_train_all))]

		X_val_all['input_ids_neg'] = X_val_all['input_ids']
		X_val_all['token_type_ids_neg'] = X_val_all['token_type_ids']
		X_val_all['attention_mask_neg'] = np.zeros((len(X_val_all['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_train_all))]
                
		X_test['input_ids_neg'] = X_test['input_ids']
		X_test['token_type_ids_neg'] = X_test['token_type_ids']
		X_test['attention_mask_neg'] = np.zeros((len(X_test['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_test))]
              
		X_unlabeled['input_ids_neg'] = X_unlabeled['input_ids']
		X_unlabeled['token_type_ids_neg'] = X_unlabeled['token_type_ids']
		X_unlabeled['attention_mask_neg'] = np.zeros((len(X_unlabeled['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_unlabeled))]
	
	'''	
	if 'joint' in type_ and 'fiine_tune_teacher' in type_:
		mask = np.where(y_train_all==1, y_train_all, 0)
		mask[:,0] = 1
		X_train_all['input_ids_r'] = np.where(mask==1, X_train_all['input_ids'], 103)
		X_train_all['token_type_ids_r'] = X_train_all['token_type_ids']
		X_train_all['attention_mask_r'] = np.where(mask==1, X_train_all['attention_mask'], 0) #np.zeros((len(X_train_all['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_train_all))]

		mask_val = np.where(y_val_all==1, y_val_all, 0)
		mask_val[:,0] = 1
		X_val_all['input_ids_r'] = np.where(mask_val==1, X_val_all['input_ids'], 103)
		X_val_all['token_type_ids_r'] = X_val_all['token_type_ids']
		X_val_all['attention_mask_r'] = np.where(mask_val==1, X_val_all['attention_mask'], 0) #np.zeros((len(X_train_all['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_train_all))]
	
		mask_test = np.where(y_test==1, y_test, 0)
		mask_test[:,0] = 1
		X_test['input_ids_r'] = np.where(mask_test==1, X_test['input_ids'], 103)
		X_test['token_type_ids_r'] = X_test['token_type_ids']
		X_test['attention_mask_r'] = np.where(mask_test==1, X_test['attention_mask'], 0) #np.zeros((len(X_test['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_test))]
              
		X_unlabeled['input_ids_r'] = X_unlabeled['input_ids']
		X_unlabeled['token_type_ids_r'] = X_unlabeled['token_type_ids']
		X_unlabeled['attention_mask_r'] = np.zeros((len(X_unlabeled['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_unlabeled))]
		logger.info(X_train_all['attention_mask_r'].shape) 

	if '_neg' in type_ and 'fiine_tune_teacher' in type_:
		neg_mask = np.where(y_train_all==1, 0, 1)
		neg_mask[:, 0] = 1
		X_train_all['input_ids_neg'] = np.where(neg_mask==1, X_train_all['input_ids'], 103) #X_train_all['input_ids']
		X_train_all['token_type_ids_neg'] = X_train_all['token_type_ids']
		X_train_all['attention_mask_neg'] = np.where(neg_mask==1, X_train_all['attention_mask'], 0)                           #np.zeros((len(X_train_all['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_train_all))]

		val_neg_mask = np.where(y_val_all==1, 0, 1)
		val_neg_mask[:, 0] = 1
		X_val_all['input_ids_neg'] = np.where(val_neg_mask==1, X_val_all['input_ids'], 103) #X_train_all['input_ids']
		X_val_all['token_type_ids_neg'] = X_val_all['token_type_ids']
		X_val_all['attention_mask_neg'] = np.where(val_neg_mask==1, X_val_all['attention_mask'], 0)                           #np.zeros((len(X_train_all['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_train_all))]
                
		test_neg_mask = np.where(y_test==1, 0, 1)
		test_neg_mask[:,0] = 1
		X_test['input_ids_neg'] = np.where(test_neg_mask==1, X_test['input_ids'], 103)
		X_test['token_type_ids_neg'] = X_test['token_type_ids']
		X_test['attention_mask_neg'] = np.where(test_neg_mask==1, X_test['attention_mask'], 0)  #np.zeros((len(X_test['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_test))]
              
		X_unlabeled['input_ids_neg'] = X_unlabeled['input_ids']
		X_unlabeled['token_type_ids_neg'] = X_unlabeled['token_type_ids']
		X_unlabeled['attention_mask_neg'] = np.zeros((len(X_unlabeled['input_ids']), max_seq_length)) #[[0]*max_seq_length for x in range(len(X_unlabeled))]
	'''
		
	for i in range(5):
		logger.info("***Train***")
		logger.info ("Example {}".format(i))
		logger.info ("Token ids {}".format(X_train_all["input_ids"][i]))
		logger.info (tokenizer.convert_ids_to_tokens(X_train_all["input_ids"][i]))
		logger.info ("Token mask {}".format(X_train_all["attention_mask"][i]))
		logger.info ("Label {}".format(y_train_all[i]))

	for i in range(5):
		logger.info("***Validation***")
		logger.info ("Example {}".format(i))
		logger.info ("Token ids {}".format(X_val_all["input_ids"][i]))
		logger.info (tokenizer.convert_ids_to_tokens(X_val_all["input_ids"][i]))
		logger.info ("Token mask {}".format(X_val_all["attention_mask"][i]))
		logger.info ("Label {}".format(y_val_all[i]))

	for i in range(5):
		logger.info("***Test***")
		logger.info ("Example {}".format(i))
		logger.info ("Token ids {}".format(X_test["input_ids"][i]))
		logger.info (tokenizer.convert_ids_to_tokens(X_test["input_ids"][i]))
		logger.info ("Token mask {}".format(X_test["attention_mask"][i]))
		logger.info ("Label {}".format(y_test[i]))

	for i in range(3):
		logger.info("***Unlabeled***")
		logger.info ("Example {}".format(i))
		logger.info ("Token ids {}".format(X_unlabeled["input_ids"][i]))
		logger.info (tokenizer.convert_ids_to_tokens(X_unlabeled["input_ids"][i]))
		logger.info ("Token mask {}".format(X_unlabeled["attention_mask"][i]))

	#labels indexed from 0
	'''
	labels = set(y_train_all)
	if 0 not in labels:
		y_train_all -= 1
		y_test -= 1 	
	labels = set(y_train_all)	
	logger.info ("Labels {}".format(labels))
    '''
	labels = list(set(y_train_all[:,0]))   #[0, 1]
	if 'boolq' in task_name:	
		class_weight = {0:1, 1:3}
	elif 'evidence' in task_name:
		class_weight = {0:2, 1:3, 2:1}
	else:
		class_weight = {0:1, 1:1}

	#if sup_labels < 0, then use all training labels in train file for learning
	if sup_labels < 0:
		X_train = X_train_all
		y_train = y_train_all
	else:
		X_input_ids, X_token_type_ids, X_attention_mask, y_train = [], [], [], []
		unlabel_input_ids, unlabel_token_type_ids, unlabel_attention_mask = [], [], []
		if 'joint' in type_:
			X_input_ids_r, X_token_type_ids_r, X_attention_mask_r = [], [], []
			unlabel_input_ids_r, unlabel_token_type_ids_r, unlabel_attention_mask_r = [], [], []
		if '_neg' in type_:
			X_input_ids_neg, X_token_type_ids_neg, X_attention_mask_neg = [], [], []
			unlabel_input_ids_neg, unlabel_token_type_ids_neg, unlabel_attention_mask_neg = [], [], []
		
		for i in labels: 
		   	#get sup_labels from each class
			indx = np.where(y_train_all[:,0]==i)[0]
                    	#indx = np.zeros(len(y_train_all))
			import random
			random.Random(GLOBAL_SEED).shuffle(indx)
			index = indx
			indx = indx[:sup_labels]
			non_train = index[sup_labels:]
			X_input_ids.extend(X_train_all["input_ids"][indx])
			X_token_type_ids.extend(X_train_all["token_type_ids"][indx])
			X_attention_mask.extend(X_train_all["attention_mask"][indx])

			unlabel_input_ids.extend(X_train_all["input_ids"][non_train])
			unlabel_token_type_ids.extend(X_train_all["token_type_ids"][non_train])
			unlabel_attention_mask.extend(X_train_all["attention_mask"][non_train])
			'''
			for ids in range(len(unlabel_input_ids)):
				X_unlabeled['input_ids']= np.concatenate((X_unlabeled['input_ids'], np.array(unlabel_input_ids)), axis=0)
				X_unlabeled['token_type_ids'] = np.concatenate((X_unlabeled['token_type_ids'], np.array(unlabel_token_type_ids)), axis=0)
				X_unlabeled['attention_mask'] = np.concatenate((X_unlabeled['attention_mask'], np.array(unlabel_attention_mask)), axis=0)
			'''
			if 'joint' in type_:
				X_input_ids_r.extend(X_train_all["input_ids_r"][indx])
				X_token_type_ids_r.extend(X_train_all["token_type_ids_r"][indx])
				X_attention_mask_r.extend(X_train_all["attention_mask_r"][indx])
				
				unlabel_input_ids_r.extend(X_train_all["input_ids_r"][non_train])
				unlabel_token_type_ids_r.extend(X_train_all["token_type_ids_r"][non_train])
				unlabel_attention_mask_r.extend(X_train_all["attention_mask_r"][non_train])
			
			if '_neg' in type_:
				X_input_ids_neg.extend(X_train_all["input_ids_neg"][indx])
				X_token_type_ids_neg.extend(X_train_all["token_type_ids_neg"][indx])
				X_attention_mask_neg.extend(X_train_all["attention_mask_neg"][indx])
				
				'''
				X_unlabeled['input_ids_r'] = np.concatenate((X_unlabeled['input_ids_r'], np.array(unlabel_input_ids_r)), axis=0)
				X_unlabeled['token_type_ids_r'] = np.concatenate((X_unlabeled['token_type_ids_r'], np.array(unlabel_token_type_ids_r)), axis=0)
				X_unlabeled['attention_mask_r'] = np.concatenate((X_unlabeled['attention_mask_r'], np.array(unlabel_attention_mask_r)), axis=0)
				'''
			y_train.extend(y_train_all[indx])
		
		logger.info("X_input_ids unlabeled shape {}".format(len(X_unlabeled['input_ids'])))
		logger.info("Y_train shape {}".format(len(y_train)))
		if '_neg' in type_:
			X_input_ids, X_token_type_ids, X_attention_mask, X_input_ids_r, X_token_type_ids_r, X_attention_mask_r, X_input_ids_neg, X_token_type_ids_neg, X_attention_mask_neg, y_train = shuffle(X_input_ids, X_token_type_ids, X_attention_mask, X_input_ids_r, X_token_type_ids_r, X_attention_mask_r, X_input_ids_neg, X_token_type_ids_neg, X_attention_mask_neg, y_train, random_state=GLOBAL_SEED)
			X_train = {"input_ids": np.array(X_input_ids), "token_type_ids": np.array(X_token_type_ids), "attention_mask": np.array(X_attention_mask), "input_ids_r": np.array(X_input_ids_r), "token_type_ids_r": np.array(X_token_type_ids_r), "attention_mask_r": np.array(X_attention_mask_r), "input_ids_neg": np.array(X_input_ids_neg), "token_type_ids_neg": np.array(X_token_type_ids_neg), "attention_mask_neg": np.array(X_attention_mask_neg)}
		elif 'joint' in type_:
			X_input_ids, X_token_type_ids, X_attention_mask, X_input_ids_r, X_token_type_ids_r, X_attention_mask_r, y_train = shuffle(X_input_ids, X_token_type_ids, X_attention_mask, X_input_ids_r, X_token_type_ids_r, X_attention_mask_r, y_train, random_state=GLOBAL_SEED)
			X_train = {"input_ids": np.array(X_input_ids), "token_type_ids": np.array(X_token_type_ids), "attention_mask": np.array(X_attention_mask), "input_ids_r": np.array(X_input_ids_r), "token_type_ids_r": np.array(X_token_type_ids_r), "attention_mask_r": np.array(X_attention_mask_r)}
		else:
			X_input_ids, X_token_type_ids, X_attention_mask, y_train = shuffle(X_input_ids, X_token_type_ids, X_attention_mask, y_train, random_state=GLOBAL_SEED)

			X_train = {"input_ids": np.array(X_input_ids), "token_type_ids": np.array(X_token_type_ids), "attention_mask": np.array(X_attention_mask)}
		y_train = np.array(y_train)
		ct = 0
		for i in y_train:
			ct += sum(i)
		wt = ct//len(y_train)	

	train_model(max_seq_length, X_train, y_train, X_test, y_test, X_unlabeled, model_dir, tokenizer, sup_batch_size=sup_batch_size, unsup_batch_size=unsup_batch_size, unsup_size=unsup_size, sample_size=sample_size, TFModel=TFModel, Config=Config, pt_teacher_checkpoint=pt_teacher_checkpoint, sample_scheme=sample_scheme, T=T, alpha=alpha, valid_split=valid_split, sup_epochs=sup_epochs, unsup_epochs=unsup_epochs, N_base=N_base, dense_dropout=dense_dropout, attention_probs_dropout_prob=attention_probs_dropout_prob, hidden_dropout_prob=hidden_dropout_prob, test_data=test_data, unlabeled_data=unlabeled_data, class_weight=class_weight, type_=type_, X_dev=X_val_all, y_dev=y_val_all, task=task_name)

