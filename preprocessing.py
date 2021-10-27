"""
Author: Meghana Bhat (bhat.89@osu.edu)
Code for Self-training for Rationale using few-shot learning.
This code base is adapted from UST (https://github.com/microsoft/UST)
"""

from collections import defaultdict

import csv
import logging
import numpy as np
import six
import re
import string
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

import os
from rationale_benchmark.utils import load_documents, load_datasets, annotations_from_jsonl, Annotation

logger = logging.getLogger('STRationale')

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


#formatting datasets from ERASER to our framework
def format_input_eraser_datasets(MAX_SEQUENCE_LENGTH, evidences, documents, unlabeled=False, tokenizer=None, mode='train', clf=None, type_='joint', fl_loss=False, task=None, query=None):
    st_sent, end_sent = -1, -1
    sentence_idx = []
    (docid,) = set(ev.docid for ev in evidences)
    sentences = documents[docid]
    s, input_sent = [], []
    st_idx = 0 
    for sent in sentences:
        if sent is None:
            continue
        if not sent == '\n': 
            p = []
            for x in sent:
                if x.isspace():
                    continue
                if isinstance(x, str):
                    p.append(x)
            sent = ' '.join(p)
            sent = sent.translate(str.maketrans('','', string.punctuation))
            #print(sent, len(sent))
            if len(sent) > 5:
                s.append(''.join(sent))
               
    if 'movies' in task or 'evidence' in task:
        for ev in evidences:
            for e in range(ev.start_sentence, ev.end_sentence+1):
                sentence_idx.append(e)
        sentence_idx = list(set(sentence_idx))
        sentence_idx.sort()
    #(docid,) = set(ev.docid for ev in evidences)
    #sentences = documents[docid]
    #s, input_sent = [], []
    #for sent in sentences:
    #    s.append(' '.join(sent))
    if mode == 'train' and ('movies' in task or 'evidence_inference' in task or 'boolq' in task): 
        for i in sentence_idx:
            if i < len(s):
                input_sent.append(s[i])
    elif ((mode == 'test') or (mode == 'val') or (not 'movies'in task and not 'evidence_inference' in task)):
        input_sent = s
    
    #only for focal loss
    #if fl_loss:
    #    input_sent = s

    #remove extra preprocessing being done
    #if not 'boolq' in task:
    input_sent = s 
    ev_text = []
    for ev in evidences:
        ev_text.append((ev.text, ev.start_token, ev.end_token))
   
    ev_sorted = sorted(ev_text, key=lambda x:x[1])
     
    if not unlabeled and (not 'movies' in task and not 'evidence_inference' in task): 
        l = [0]*MAX_SEQUENCE_LENGTH
        input_example = ' '.join(input_sent)
        n_words = len(input_example.split())
        seq_tag, ev_tag = 1, 0
        for i in range(0, n_words):
            if seq_tag > MAX_SEQUENCE_LENGTH:
                break
            if ev_tag >= len(ev_sorted):
                break
            text, st, end = ev_sorted[ev_tag]
            tx = tokenizer.tokenize(input_example[i])
            j = 0
            while(j<len(tx) and (seq_tag+j)<MAX_SEQUENCE_LENGTH):
                if (i>=st) and (i<=end):
                    l[seq_tag+j] = 1
                    if i==end:
                        ev_tag+=1
                else:
                    l[seq_tag+j] = 0
                j+=1
            seq_tag+=len(tx)
    elif unlabeled:
        l = [-1]*MAX_SEQUENCE_LENGTH
    elif not unlabeled and ('movies' in task or 'evidence_inference' in task):
        l = [0]*MAX_SEQUENCE_LENGTH
        i = 1 

        for idx, s in enumerate(input_sent):
            w = 0
            while(w < len(s.split())):
                t = len(tokenizer.tokenize(s.split()[w]))
                flag = False
                if i > MAX_SEQUENCE_LENGTH-1:
                    break
                sent_list = s.split()
                for ix, inst in enumerate(ev_sorted):
                    if sent_list[w] in ev_sorted[ix][0]:
                        text = ev_sorted[ix][0] 
                        #print("Printing evidence text (sentence wise")
                        #print(text, s)
                        #print(' '.join(sent_list[w:(w+len(text.split()))]), ":::", text, ":::", len(text.split()))
                        if ' '.join(sent_list[w:(w+len(text.split()))]) == text:
                            words = text.split()
                            for each in words:
                                tx = tokenizer.tokenize(each)
                                j = 0
                                while(j<len(tx) and (i+j)<MAX_SEQUENCE_LENGTH):
                                    l[i+j] = 1
                                    j+=1
                                i+=len(tx)
                            w+=len(text.split())
                            flag = True
                            break
                if not flag:
                    #print(s.split()[w])
                    i+=t  
                    w+=1
        
    else:
        l = [-1]*MAX_SEQUENCE_LENGTH
    
    
    return ' '.join(input_sent), l

def generate_rationale_data(MAX_SEQUENCE_LENGTH, input_file, mode, tokenizer, unlabeled=False, do_pairwise=False, type_='joint', fl_loss=False, task=None):
    X1 = []
    X2 = []
    y = []
    q = []
    data_root = os.path.join(input_file)
    documents = load_documents(data_root)
    label_count = defaultdict(int)
    c = 0
    data_label = annotations_from_jsonl(os.path.join(data_root, 'train.jsonl'))
    for l in data_label:
        if not l.classification in label_count.keys():
            label_count[l.classification] = c
            c+=1
    if mode == 'train':
        data = annotations_from_jsonl(os.path.join(data_root, 'train.jsonl'))
    elif mode == 'test':
        data = annotations_from_jsonl(os.path.join(data_root, 'test.jsonl'))
    elif mode == 'val':
        data = annotations_from_jsonl(os.path.join(data_root, 'val.jsonl'))
    
    else:
        #data = annotations_from_jsonl(os.path.join(data_root, 'val.jsonl'))
        train, val, _ = load_datasets(data_root)
        data = train
    #label_count = defaultdict(int)
    labels_dict = defaultdict(int)
    c = 0
    for ann in data:
        evidences = ann.all_evidences()
        query = None
        if len(evidences) ==  0:
            continue

        if do_pairwise and not unlabeled:
            query = ann.query
            X2.append(convert_to_unicode(query))
            q_l = tokenizer.tokenize(query)
            q.append([1]*(len(q_l)+1))
        #print(ann.query, evidences) 
        if not unlabeled:
            text, label = format_input_eraser_datasets(MAX_SEQUENCE_LENGTH, evidences, documents, False, tokenizer, mode, ann.classification, type_, fl_loss, task, query=query)
            X1.append(convert_to_unicode(text))
            label[0] = label_count[ann.classification]
            labels_dict[ann.classification]+=1
            y.append(label)
        else:
            text, label = format_input_eraser_datasets(MAX_SEQUENCE_LENGTH, evidences, documents, True, tokenizer)
            X1.append(text)
            y.append(label)
            
        if len(text) == 0:
            continue
   
    if do_pairwise:
        for i in range(len(y)):
            y[i] = np.concatenate([[np.array(y[i][0])],np.array(q[i]), np.array(y[i][1:])])
            y[i] = y[i][:MAX_SEQUENCE_LENGTH]
        X = tokenizer(X2, X1, padding='max_length', truncation=True, max_length=MAX_SEQUENCE_LENGTH)
    else:
        X = tokenizer(X1, padding='max_length', truncation=True, max_length=MAX_SEQUENCE_LENGTH)
    
    for key in labels_dict.keys():
        logger.info("Count of instances with label {} is {}".format(key, labels_dict[key]))
    
    if "token_type_ids" not in X:
        token_type_ids = np.zeros((len(X["input_ids"]), MAX_SEQUENCE_LENGTH))
    else:
        token_type_ids = np.array(X["token_type_ids"])
    logger.info(label_count)
    logger.info("Task name: {}".format(task))

    return {"input_ids": np.array(X["input_ids"]), "token_type_ids": token_type_ids,
            "attention_mask": np.array(X["attention_mask"])}, np.array(y), X2