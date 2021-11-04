"""
Author: Meghana Bhat (bhat.89@osu.edu)
Code for Self-training for Rationale using few-shot learning.
This code base is adapted from UST (https://github.com/microsoft/UST)
"""

from collections import defaultdict
from sklearn.utils import shuffle
from transformers import *

import logging
import math
import models
import numpy as np
import os, sys
import json
import nltk
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.backend as kb
import tensorflow_addons as tfa
from focal_loss import BinaryFocalLoss, SparseCategoricalFocalLoss
import random
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger('STRationale')

def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler

def train_model(max_seq_length, X, y, X_test, y_test, X_unlabeled, model_dir, tokenizer, sup_batch_size=4, unsup_batch_size=32, unsup_size=4096, sample_size=16384, TFModel=TFBertModel, Config=BertConfig, pt_teacher_checkpoint='bert-base-uncased', sample_scheme='easy_bald_class_conf', T=30, alpha=0.1, valid_split=0.5, sup_epochs=70, unsup_epochs=25, N_base=10, dense_dropout=0.5, attention_probs_dropout_prob=0.3, hidden_dropout_prob=0.3, test_data=None, unlabeled_data=None, class_weight=None, type_="token", X_dev=None, y_dev=None, task=None):
        #labels = [0, 1]  #fix hardcoding
        labels = set(y[:,0])
        logger.info ("Class labels {}".format(labels))

        #split X and y to train and dev with valid_split
        if valid_split > 0:
            train_size = int((1. - valid_split)*len(X["input_ids"]))
            if '_neg' in type_:
                X_train, y_train = {"input_ids": X["input_ids"][:train_size], "token_type_ids": X["token_type_ids"][:train_size], "attention_mask": X["attention_mask"][:train_size], "input_ids_r":X["input_ids_r"][:train_size], "token_type_ids_r":X["token_type_ids_r"][:train_size], "attention_mask_r":X["attention_mask_r"][:train_size], "input_ids_neg":X["input_ids_neg"][:train_size], "token_type_ids_neg":X["token_type_ids_neg"][:train_size], "attention_mask_neg":X["attention_mask_neg"][:train_size]}, y[:train_size]
                X_dev, y_dev = {"input_ids": X["input_ids"][train_size:], "token_type_ids": X["token_type_ids"][train_size:], "attention_mask": X["attention_mask"][train_size:], "input_ids_r":X["input_ids_r"][train_size:], "token_type_ids_r":X["token_type_ids_r"][train_size:], "attention_mask_r":X["attention_mask_r"][train_size:], "input_ids_neg":X["input_ids_neg"][train_size:], "token_type_ids_neg":X["token_type_ids_neg"][train_size:], "attention_mask_neg":X["attention_mask_neg"][train_size:]}, y[train_size:]
            elif 'joint' in type_:
                X_train, y_train = {"input_ids": X["input_ids"][:train_size], "token_type_ids": X["token_type_ids"][:train_size], "attention_mask": X["attention_mask"][:train_size], "input_ids_r":X["input_ids_r"][:train_size], "token_type_ids_r":X["token_type_ids_r"][:train_size], "attention_mask_r":X["attention_mask_r"][:train_size]}, y[:train_size]
                X_dev, y_dev = {"input_ids": X["input_ids"][train_size:], "token_type_ids": X["token_type_ids"][train_size:], "attention_mask": X["attention_mask"][train_size:], "input_ids_r":X["input_ids_r"][train_size:], "token_type_ids_r":X["token_type_ids_r"][train_size:], "attention_mask_r":X["attention_mask_r"][train_size:]}, y[train_size:]
            else:
                X_train, y_train = {"input_ids": X["input_ids"][:train_size], "token_type_ids": X["token_type_ids"][:train_size], "attention_mask": X["attention_mask"][:train_size]}, y[:train_size]

                X_dev, y_dev = {"input_ids": X["input_ids"][train_size:], "token_type_ids": X["token_type_ids"][train_size:], "attention_mask": X["attention_mask"][train_size:]}, y[train_size:]
        else:
            X_train, y_train = X, y
            X_dev, y_dev = X_dev, y_dev


        logger.info("X Train Shape: {} {}".format(X_train["input_ids"].shape, y_train.shape))
        logger.info("X Dev Shape: {} {}".format(X_dev["input_ids"].shape, y_dev.shape))
        logger.info("X Test Shape: {} {}".format(X_test["input_ids"].shape, y_test.shape))
        logger.info ("X Unlabeled Shape: {}".format(X_unlabeled["input_ids"].shape))

        strategy = tf.distribute.MirroredStrategy()
        gpus = strategy.num_replicas_in_sync
        logger.info('Number of devices: {}'.format(gpus))

        #run the base model n times with different initialization to select best base model based on validation loss
        best_base_model = None
        best_validation_loss = np.inf
        
        for counter in range(N_base): #original N_base=10
            with strategy.scope():
                if 'mtl' in type_:
                    rat_loss = None
                    if 'focal' in type_:
                        rat_loss = SparseCategoricalFocalLoss(gamma=2)
                    else:
                        rat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                   
                    model = models.construct_teacher_mtl(TFModel, Config, pt_teacher_checkpoint, max_seq_length, len(labels), dense_dropout=dense_dropout, attention_probs_dropout_prob=attention_probs_dropout_prob, hidden_dropout_prob=hidden_dropout_prob)
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), rat_loss], metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="dense_3_classification_acc")])#, tf.keras.metrics.SparseCategoricalAccuracy(name="token_acc")]) #, sample_weight_mode="temporal")
                elif type_ == 'joint':
                    rat_loss = None
                    if 'focal' in type_:
                        rat_loss = SparseCategoricalFocalLoss(gamma=2)
                    else:
                        rat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                   
                    model = models.construct_teacher_joint(TFModel, Config, pt_teacher_checkpoint, max_seq_length, len(labels), dense_dropout=dense_dropout, attention_probs_dropout_prob=attention_probs_dropout_prob, hidden_dropout_prob=hidden_dropout_prob)
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), loss={'task_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'rationale_classifier':tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'rationale_task_classifier': None, 'l2_distance': None}, metrics={'task_classifier':[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], 'rationale_classifier':[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], 'rationale_task_classifier':[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], 'l2_distance': None})
                elif 'joint_neg' in type_:
                    rat_loss = None
                    if 'focal' in type_:
                        rat_loss = SparseCategoricalFocalLoss(gamma=2)
                    else:
                        rat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

                    def custom_loss(y_true, y_pred):
                        cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
 
                        if 'focal' in type_:
                            cce = SparseCategoricalFocalLoss(gamma=2, reduction=tf.keras.losses.Reduction.NONE)

                        cce_loss = ((cce(y_true, y_pred))* 1/(unsup_batch_size*gpus))
                        l1_loss = tf.reduce_mean(tf.reduce_sum(tf.math.abs(y_pred),axis=0))
                        coh_loss = tf.reduce_mean(tf.reduce_sum(tf.math.abs(y_pred[1:]-y_pred[:-1]), axis=0))
                        #l2_loss = 0.0
                        #logger.info(l1_loss)
                        return cce_loss + 0.01*l1_loss + 0.01*coh_loss
                
                    def custom_loss_neg(y_true, y_pred):
                        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
                        return tf.reduce_sum(cce(y_true, y_pred))*(1/(unsup_batch_size*gpus))
                     
                    model = models.construct_teacher_joint_neg(TFModel, Config, pt_teacher_checkpoint, max_seq_length, len(labels), dense_dropout=dense_dropout, attention_probs_dropout_prob=attention_probs_dropout_prob, hidden_dropout_prob=hidden_dropout_prob)
                    loss_weights = [1.0, 1.0, 1.0, 1.0]
                    if '_noexp' in type_:
                        loss_weights = [1.0, 0.0, 0.0, 0.0]
                    elif '_no_suffcomp' in type_:
                        loss_weights = [1.0, 1.0, 0, 0]
                    
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), loss={'task_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'rationale_classifier':rat_loss, 'rationale_task_classifier': None, 'not_rationale_task_classifier': None}, metrics={'task_classifier':[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], 'rationale_classifier':[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], 'rationale_task_classifier':[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], 'not_rationale_task_classifier': None}, loss_weights=loss_weights)
                    
                if counter == 0:
                    logger.info(model.summary())

            model_file = os.path.join(model_dir, "model_label.h5")
            model_file_task = os.path.join(model_dir, "model_task.h5")
            model_file_best = os.path.join(model_dir, "model_best.h5")
            
            if os.path.exists(model_file):
                model.load_weights(model_file)
                #model_task.load_weights(model_file_task)
                best_base_model = model
                logger.info ("Model file loaded from {}".format(model_file))
                break
            elif 'mtl' in type_ :
                logger.info(y_train.shape)
                model.fit(x=X_train, y=[y_train[:,0], y_train[:,1:]], shuffle=True, epochs=sup_epochs, validation_data=(X_dev, [y_dev[:,0], y_dev[:,1:]]), batch_size=sup_batch_size*gpus, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]) # class_weight=class_weight)
 
                val_loss = model.evaluate(X_dev, [y_dev[:,0], y_dev[:,1:]])
            elif '_neg' in type_ :
                y_neg = np.full((len(y_train),len(labels)), 1/len(labels))
                model.fit(x=X_train, y=[y_train[:,0], y_train[:,1:], y_train[:,0], y_neg], shuffle=True, epochs=sup_epochs, validation_data=(X_dev, [y_dev[:,0], y_dev[:,1:], y_dev[:,0], np.full((len(y_dev), len(labels)), 1/len(labels))]), batch_size=sup_batch_size*1, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]) #, class_weight=class_weight)
                val_loss = model.evaluate(X_dev, [y_dev[:,0], y_dev[:,1:], y_dev[:,0], np.full((len(y_dev), len(labels)), 1/len(labels))])
            elif 'joint' in type_:
                _placeholder_labels = np.empty((y_train.shape[0], y_train.shape[0])) 
                model.fit(x=X_train, y=[y_train[:,0], y_train, y_train[:,0], np.ones(len(y_train))], shuffle=True, epochs=sup_epochs, validation_data=(X_dev, [y_dev[:,0], y_dev, y_dev[:,0], np.ones(len(y_dev))]), batch_size=sup_batch_size*gpus, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]) # class_weight=class_weight)
                val_loss = model.evaluate(X_dev, [y_dev[:,0], y_dev, y_dev[:,0], np.ones(len(y_dev))])
            
            logger.info ("Validation loss for run {} : {}".format(counter, val_loss))
            if val_loss[0] <  best_validation_loss:
                best_base_model = model
                best_validation_loss = val_loss[0]

        model = best_base_model
        '''
        if 'mtl' in type_:
            logger.info ("Best validation acc for base model {}: {}".format(best_validation_loss, model.evaluate(X_dev, [y_dev[:,0],y_dev[:,1:]])))
        '''
        if not os.path.exists(model_file):
            model.save_weights(model_file)
            logger.info ("Model file saved to {}".format(model_file))

        best_val_acc = 0.
        best_test_acc = 0.
        max_test_acc = 0.
        max_task_acc = 0.
        max_best_acc = 0.
        val_loss = 0.
        if  'mtl' in type_:
            logger.info("y_test: {}".format(y_test))
            test_acc = model.evaluate(X_test, [y_test[:,0], y_test[:,1:]], verbose=0)[4]
            task_acc = model.evaluate(X_test, [y_test[:,0], y_test[:,1:]], verbose=0)[3]
            val_loss = model.evaluate(X_test, [y_test[:,0], y_test[:,1:]], verbose=0)[0]
        elif '_neg' in type_:
            out = model.evaluate(X_test, [y_test[:,0], y_test[:,1:], y_test[:,0], np.full((len(y_test), len(labels)), 1/len(labels))]) 
            task_acc, test_acc, r_acc = out[3], out[4], out[5]
        elif 'joint' in type_:
            out = model.evaluate(X_test, [y_test[:,0], y_test, y_test[:,0], np.ones(len(y_test))]) 
            task_acc, test_acc, r_acc = out[3], out[4], out[5]
        logger.info ("Test token acc for run {} : {}".format(counter, test_acc))    
        logger.info ("Best Test task acc for run {} with total loss : {}".format(counter, task_acc))
        
        if 'mtl' in type_:
            class_acc = model.predict(X_test)[0]
            test_pred = model.predict(X_test)[1]
            class_acc = np.argmax(class_acc, axis=-1)
        elif 'joint' in type_:
            out = model.predict(X_test)   
            class_acc, test_pred, r_acc = out[0], out[1], out[2]
            class_acc = np.argmax(class_acc, axis=-1)
            logger.info("Class predictions shape {}".format(class_acc.shape))
        
        logger.info("Teacher model best score (macro/task): {}".format(precision_recall_fscore_support(class_acc, y_test[:,0], average='macro')))
        logger.info("Teacher model best score (micro/task): {}".format(precision_recall_fscore_support(class_acc, y_test[:,0], average='micro')))
            
        logger.info("Token Predictions shape {}".format(test_pred.shape))
        
        pred, truth = [], []
        
        logger.info(test_pred)
        test_pred = np.argmax(tf.nn.softmax(test_pred, axis=-1), axis=-1)
        logger.info("Printing prediction data on teacher model for run {}: {}".format(counter, test_pred))
        tp, fn, fp =  0, 0, 0
        pred_1, pred_0, truth_1, truth_0 = 0, 0, 0, 0
        
        for i in range(len(test_pred)):
            temp_p, temp_t, ct = [],[], 0
            temp = tokenizer.convert_ids_to_tokens(X_test["input_ids"][i])[1:]
            
            for j in range(0,len(test_pred[0])-1):
                if test_pred[i][j] == 1:
                    temp_p.append(temp[j])
                if y_test[i][j+1] == 1: #to skip evaluation of the task label
                    temp_t.append(temp[j])
            pred_1 += test_pred[i].sum()
            pred_0+= max_seq_length-pred_1

            truth_1 += y_test[i].sum()
            truth_0+= max_seq_length-truth_1
            pred.append(' '.join(temp_p))
            truth.append(' '.join(temp_t))
            for word in temp_p:
                if word in temp_t:
                    ct+=1
                    temp_t.remove(word)
                else:
                    fp+=1
            tp +=ct
            fn += (y_test[i].sum()-ct)
            
        p = tp/(tp+fp+0.0000001)
        r = tp/(tp+fn+0.0000001)
        logger.info("Token-level: {}".format((tp)/(tp+(0.5*(fp+fn)))))
        logger.info("Rationale coverage (recall): {}".format(r))
        logger.info("Token Precision: {}".format(p))
        logger.info("Token overlap: {}".format(tp/(tp+fp+fn)))
        
        score1, score2, score3, score4 = 0.0, 0.0, 0.0, 0.0
        for i in range(len(pred)):
            score1 += nltk.translate.bleu_score.sentence_bleu([truth[i].split()],pred[i].split(), weights=(1, 0, 0, 0))
            score2 += nltk.translate.bleu_score.sentence_bleu([truth[i].split()],pred[i].split(), weights=(0, 1, 0, 0))
            score3 += nltk.translate.bleu_score.sentence_bleu([truth[i].split()],pred[i].split(), weights=(0, 0, 1, 0))
            score4 += nltk.translate.bleu_score.sentence_bleu([truth[i].split()],pred[i].split(), weights=(0, 0, 0, 1))
        
        logger.info("BLEU-1 score of rationales on test set (teacher model): {} ".format(score1/len(pred)))
        logger.info("BLEU-2 score of rationales on test set (teacher model): {} ".format(score2/len(pred)))
        logger.info("BLEU-3 score of rationales on test set (teacher model): {} ".format(score3/len(pred)))
        logger.info("BLEU-4 score of rationales on test set (teacher model): {} ".format(score4/len(pred)))

        best_loss = np.inf
        data = []
        for i in range(len(X_test["input_ids"])):
            text = tokenizer.convert_ids_to_tokens(X_test["input_ids"][i])
            temp = dict()
            temp['text'] = ' '.join(text)
            temp['truth'] = truth[i]
            temp['pred'] = pred[i]
            temp['score'] = nltk.translate.bleu_score.sentence_bleu([truth[i].split()],pred[i].split())
            data.append(temp)

        with open(os.path.join(model_dir, 'rationale_output_test_teacher_'+type_+'.json'), 'w') as f:
            json.dump(data, f)
        
        model_student = None # model_task
        for epoch in range(unsup_epochs):

            logger.info ("Starting loop {}".format(epoch))
            if type_ == 'mtl':
                test_acc = model.evaluate(X_dev, [y_dev[:,0], y_dev[:,1:]], verbose=0)[-1]
                task_acc = model.evaluate(X_dev, [y_dev[:,0], y_dev[:,1:]], verbose=0)[-2]
                val_loss = model.evaluate(X_dev, [y_dev[:,0], y_dev[:,1:]], verbose=0)[0]
                if task_acc > max_task_acc:
                    logger.info ("Val acc (task) {}".format(task_acc))
                    max_task_acc = task_acc
                    model.save_weights(model_file_best)
                val_acc = model.evaluate(X_dev, [y_dev[:,0], y_dev[:,1:]], verbose=0)[-2]
                test_task_acc = model.evaluate(X_test, [y_test[:,0], y_test[:,1:]], verbose=0)[-2]
            elif 'joint_neg' in type_:
                y_neg_dev = np.full((len(y_dev), len(labels)), 1/len(labels))
                y_neg_test = np.full((len(y_test), len(labels)), 1/len(labels))
                y_dev_plg = [y_dev[:,1:], y_dev[:,0], np.full((len(y_dev),len(labels)), 1/len(labels))]
                y_test_plg = [y_test[:,1:], y_test[:,0], np.full((len(y_test),len(labels)), 1/len(labels))]
                test_acc = model.evaluate(X_dev, [y_dev[:,0], y_dev[:,1:], y_dev[:,0], y_neg_dev], verbose=0)[-2]

                task_acc = model.evaluate(X_dev, [y_dev[:,0], y_dev[:,1:], y_dev[:,0], y_neg_dev], verbose=0)[-3]
                out1 = model.predict(X_test)
                acc1, y_pred1, r_acc1 = out1[0], out1[1], out1[2]
                y_pred1 = np.argmax(y_pred1, axis=-1)
                acc1 = np.argmax(acc1, axis=-1)
                r_acc1 = np.argmax(r_acc1, axis=-1)
                logger.info("Model performance for token (macro/task): {}".format(precision_recall_fscore_support(y_pred1, y_test[:,1:], average='micro')))
                logger.info("Model performance for token (macro/task): {}".format(precision_recall_fscore_support(y_pred1, y_test[:,1:], average='macro')))
                logger.info("Model performance for task (macro/task): {}".format(precision_recall_fscore_support(acc1, y_test[:,0], average='macro')))

                val_loss = model.evaluate(X_dev, [y_dev[:,0], y_dev[:,1:], y_dev[:,0], y_neg_dev], verbose=0)[0]
                if task_acc > max_task_acc:
                    logger.info ("Val acc (task) {}".format(task_acc))
                    max_task_acc = task_acc
                    best_val_acc = task_acc
                    model.save_weights(model_file_best) #_student = deepcopy(model)
                val_acc = task_acc #model.evaluate(X_dev, [y_dev[:,0], y_dev, y_dev[:,0], y_neg_dev], verbose=0)[-3]
                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                test_task_acc = model.evaluate(X_test, [y_test[:,0], y_test[:,1:], y_test[:,0], y_neg_test], verbose=0)[-3]
           
            elif type_ == 'joint': # or 'joint_neg' in type_:
                test_acc = model.evaluate(X_dev, [y_dev[:,0], y_dev[:,1:], y_dev[:,0], np.ones(len(y_dev))], verbose=0)[-2]
                task_acc = model.evaluate(X_dev, [y_dev[:,0], y_dev[:,1:], y_dev[:,0], np.ones(len(y_dev))], verbose=0)[-3]
              
                val_loss = model.evaluate(X_dev, [y_dev[:,0], y_dev[:,1:], y_dev[:,0], np.ones(len(y_dev))], verbose=0)[0]
                if task_acc > max_task_acc:
                    logger.info ("Val acc (task) {}".format(task_acc))
                    max_task_acc = task_acc
                    best_val_acc = task_acc
                    model.save_weights(model_file_best) #_student = deepcopy(model)
                val_acc = model.evaluate(X_dev, [y_dev[:,0], y_dev, y_dev[:,0], np.ones(len(y_dev))], verbose=0)[-3]
                '''
                if val_loss < best_loss:
                    best_loss = val_loss
                    model.save_weights(model_file_best) #_student = deepcopy(model)
                '''
                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                test_task_acc = model.evaluate(X_test, [y_test[:,0], y_test[:,1:], y_test[:,0], np.ones(len(y_test))], verbose=0)[-3]
            
            if '_neg' in type_: 
                y_neg_dev = np.full((len(y_dev), len(labels)), 1/len(labels))
                y_neg_test = np.full((len(y_test), len(labels)), 1/len(labels))
                temp = model.evaluate(X_test, [y_test[:,0], y_test[:,1:], y_test[:,0], y_neg_test], verbose=0)
            elif 'joint' in type_:
                temp = model.evaluate(X_test, [y_test[:,0], y_test[:,1:], y_test[:,0], np.ones(len(y_test))], verbose=0)
            elif 'mtl' in type_:
                 temp = model.evaluate(X_test, [y_test[:,0], y_test[:,1:]], verbose=0)

            logger.info("Print acc (task) for joint {}".format(temp)) 
            logger.info ("Val acc (token) {}".format(test_acc))
            logger.info ("Val acc (task) {}".format(task_acc))
            logger.info ("Test acc (task) {}".format(test_task_acc))
            if test_task_acc >= max_best_acc:
                max_best_acc = test_task_acc    
            model_file = os.path.join(model_dir, "model_token_{}_{}.h5".format(epoch, sample_scheme))
            model_file_task = os.path.join(model_dir, "model_task_{}_{}.h5".format(epoch, sample_scheme))
            

            if os.path.exists(model_file):
               model.load_weights(model_file)
               logger.info ("Model file loaded from {}".format(model_file))
               continue
            if 'mtl' in type_ :    
               acc, y_pred = model.predict(X_unlabeled, batch_size=256)
               #y_val = np.amax(acc, axis=-1)
               #y_rat = np.amax(y_pred, axis=-1)
               y_pred = np.argmax(y_pred, axis=-1) #.flatten()
               acc = np.argmax(acc, axis=-1)
            elif 'joint' in type_:
               out = model.predict(X_unlabeled, batch_size=64)
               acc, y_pred, r_acc = out[0], out[1], out[2]
               #y_val = np.amax(acc, axis=-1)
               #y_rat = np.amax(y_pred, axis=-1)
               y_pred = np.argmax(y_pred, axis=-1) #.flatten()
               acc = np.argmax(acc, axis=-1)
               r_acc = np.argmax(r_acc, axis=-1)
            
            #compute confidence on the unlabeled set
            if sample_size < len(X_unlabeled["input_ids"]):
                logger.info ("Evaluating confidence on {} number of instances sampled from {} unlabeled instances".format(sample_size, len(X_unlabeled["input_ids"])))
                indices = np.random.choice(len(X_unlabeled["input_ids"]), sample_size, replace=False)
                if '_neg' in type_:
                    X_unlabeled_sample, y_pred = {'input_ids': X_unlabeled["input_ids"][indices], 'token_type_ids': X_unlabeled["token_type_ids"][indices], 'attention_mask': X_unlabeled["attention_mask"][indices], 'input_ids_r':X_unlabeled['input_ids_r'][indices], 'token_type_ids_r':X_unlabeled['token_type_ids_r'][indices], 'attention_mask_r':X_unlabeled['attention_mask_r'][indices], 'input_ids_neg':X_unlabeled['input_ids_neg'][indices], 'token_type_ids_neg':X_unlabeled['token_type_ids_neg'][indices], 'attention_mask_neg':X_unlabeled['attention_mask_neg'][indices]}, y_pred[indices]
                elif 'joint' in type_:
                    X_unlabeled_sample, y_pred = {'input_ids': X_unlabeled["input_ids"][indices], 'token_type_ids': X_unlabeled["token_type_ids"][indices], 'attention_mask': X_unlabeled["attention_mask"][indices], 'input_ids_r':X_unlabeled['input_ids_r'][indices], 'token_type_ids_r':X_unlabeled['token_type_ids_r'][indices], 'attention_mask_r':X_unlabeled['attention_mask_r'][indices]}, y_pred[indices]
                else:
                    X_unlabeled_sample, y_pred = {'input_ids': X_unlabeled["input_ids"][indices], 'token_type_ids': X_unlabeled["token_type_ids"][indices], 'attention_mask': X_unlabeled["attention_mask"][indices]}, y_pred[indices]
            else:
                logger.info ("Evaluating confidence on {} number of instances".format(len(X_unlabeled["input_ids"])))
                X_unlabeled_sample = X_unlabeled
                #X_unlabeled_sample = {'input_ids': X_unlabeled["input_ids"][indices], 'token_type_ids': X_unlabeled["token_type_ids"][indices], 'attention_mask': X_unlabeled["attention_mask"][indices]}
 
            #logger.info (X_unlabeled_sample["input_ids"][:5])
            if 'joint' in type_:
                ids = []
                attention_mask_r = np.ones((len(y_pred), max_seq_length))
                attention_mask_r[:,1:] = np.array(y_pred)
                #logger.info(y_pred.shape)
                #logger.info("Percentage of rationales selected: {}".format(np.mean(np.sum(attention_mask_r, axis=-1))))
                attention_mask_r[:,0] = 1
                negation_mask = np.where(attention_mask_r==0, 1, 0)
                negation_mask[:,0] = 1
                X_sample =  {"input_ids": np.array(X_unlabeled_sample["input_ids"]), "token_type_ids": np.array(X_unlabeled_sample['token_type_ids']), "attention_mask": attention_mask_r}
                #mask tokens that are not rationales u-r
                if '_neg' in type_:
                    X_negation_sample =  {"input_ids": np.array(X_unlabeled_sample["input_ids"]), "token_type_ids": np.array(X_unlabeled_sample['token_type_ids']), "attention_mask": negation_mask}
                
                for i in range(len(y_pred)):
                    X_sample["input_ids"][i, 1:] = np.where(y_pred[i]==0, 103, X_sample["input_ids"][i, 1:])
                    if '_neg' in type_:
                        X_negation_sample["input_ids"][i, 1:] = np.where(y_pred[i]==0, X_negation_sample["input_ids"][i, 1:], 103)
                        X_negation_sample["input_ids"][:,0] = 101

                X_sample["input_ids"][:,0] = 101
                logger.info("Extracted rationale from teacher model as input for task: {}".format(X_sample["input_ids"][:5]))
                logger.info("Extracted rationale from teacher model as input for task: {}".format(X_negation_sample["input_ids"][:5]))
                
            y_mean, y_var, y_T = None, None, None
            

            if 'mtl' in type_:
                acc, y_pred = model.predict(X_unlabeled_sample, batch_size=256)
                y_val = np.amax(tf.math.softmax(acc, axis=-1).numpy(), axis=-1)
                y_rat = np.amax(tf.math.softmax(y_pred, axis=-1).numpy(), axis=-1)
           
                y_pred = np.argmax(y_pred, axis=-1) #.flatten()
                acc = np.argmax(acc, axis=-1)
            elif 'joint' in type_:
                if 'pruthi_' in type_:
                    out = y_train
                    acc, y_pred, r_acc = y_train[:,0], y_train[:,1:], y_train[:,0]
                    y_val = acc
                    y_rat = np.array(y_pred).astype('float')
                    #y_rat = y_rat[:,1:]
                    #y_pred = y_pred[:,1:]
                else:
                    out = model.predict(X_unlabeled_sample, batch_size=64)
                    acc, y_pred, r_acc = out[0], out[1], out[2]
                    y_val = np.amax(tf.math.softmax(acc, axis=-1).numpy(), axis=-1)
                    y_rat = np.amax(tf.math.softmax(y_pred, axis=-1).numpy(), axis=-1)
           
                    y_pred = np.argmax(y_pred, axis=-1) #.flatten()
                    acc = np.argmax(acc, axis=-1)
                    r_acc = np.argmax(r_acc, axis=-1) 
                    #y_rat = y_rat[:, 1:] 
                    #y_pred = y_pred[:,1:]
                
            # sample from unlabeled set
            
            if 'uni' in sample_scheme:
                logger.info ("Sampling uniformly")
                if unsup_size < len(X_unlabeled_sample['input_ids']):
                    '''X_unlabeled_sample, y_pred = {"input_ids": X_unlabeled_sample['input_ids'][indices], "token_type_ids": X_unlabeled_sample['token_type_ids'][indices], "attention_mask": X_unlabeled_sample['attention_mask'][indices]}, y_pred[indices]
                    if type_ == 'decoupled' or ('joint' in type_):
                        X_sample = {"input_ids": X_sample['input_ids'][indices], "token_type_ids": X_sample['token_type_ids'][indices], "attention_mask": X_sample['attention_mask'][indices]}
                    '''
                    #acc = acc[:,None]
                    #y_batch = np.concatenate((acc[indices], y_pred), axis=1)
                    acc = acc[:,None]
                    y_batch = np.concatenate((acc, y_pred), axis=1)
                    logging.info("y_batch shape {}".format(y_batch.shape)) 
                    indices = []
                    
                    for i in labels:
                        indx = np.where(y_batch[:,0]==i)[0]
                        GLOBAL_SEED = int(os.getenv("PYTHONHASHSEED"))
                        random.Random(GLOBAL_SEED).shuffle(indx)
                        if len(indx) > unsup_size:
                            indx = indx[:unsup_size]
                        logger.info("Shape of predicted labels for class {} : {}".format(i, len(indx)))
                        indices.extend(indx)
                    indices = np.asarray(indices)    
                    
                    #indices = np.random.choice(len(X_unlabeled_sample['input_ids']), unsup_size, replace=False)
                    X_batch, y_batch = {"input_ids": X_unlabeled_sample['input_ids'][indices], "token_type_ids": X_unlabeled_sample['token_type_ids'][indices], "attention_mask": X_unlabeled_sample['attention_mask'][indices]}, y_batch[indices]
                    if 'joint' in type_:
                        X_rationale_batch = {"input_ids_r": X_sample['input_ids'][indices], "token_type_ids_r": X_sample['token_type_ids'][indices], "attention_mask_r": X_sample['attention_mask'][indices]}
                        if '_neg' in type_:
                            X_neg_rationale_batch = {"input_ids_neg": X_negation_sample['input_ids'][indices], "token_type_ids_neg": X_negation_sample['token_type_ids'][indices], "attention_mask_neg": X_negation_sample['attention_mask'][indices]}
                else:
                    indices = np.array([i for i in range(len(y_pred))])
                    acc = acc[:,None]
                    y_batch = np.concatenate((acc[indices], y_pred[indices]), axis=1)
                    X_batch = {"input_ids": X_unlabeled_sample['input_ids'][indices], "token_type_ids": X_unlabeled_sample['token_type_ids'][indices], "attention_mask": X_unlabeled_sample['attention_mask'][indices]}
                    if 'joint' in type_:
                        X_rationale_batch = {"input_ids_r": X_sample['input_ids'][indices], "token_type_ids_r": X_sample['token_type_ids'][indices], "attention_mask_r": X_sample['attention_mask'][indices]}
                        if '_neg' in type_:
                            X_neg_rationale_batch = {"input_ids_neg": X_negation_sample['input_ids'][indices], "token_type_ids_neg": X_negation_sample['token_type_ids'][indices], "attention_mask_neg": X_negation_sample['attention_mask'][indices]}
                '''
                probs = y_val[indices]
                X_conf = np.ones((len(y_batch), max_seq_length))
                X_conf[:,0] = np.log(probs+1e-10)*alpha
                '''
            else:
                logger.info("No sampling at the moment; choose all the unlabeled examples")
                X_batch = {"input_ids": X_unlabeled_sample['input_ids'][indices], "token_type_ids": X_unlabeled_sample['token_type_ids'][indices], "attention_mask": X_unlabeled_sample['attention_mask'][indices]}
                if 'joint' in type_:
                    X_rationale_batch = {"input_ids_r": X_sample['input_ids'][indices], "token_type_ids_r": X_sample['token_type_ids'][indices], "attention_mask_r": X_sample['attention_mask'][indices]}
 
                if '_neg' in type_:
                    X_neg_rationale_batch = {"input_ids_neg": X_negation_sample['input_ids'][indices], "token_type_ids_neg": X_negation_sample['token_type_ids'][indices], "attention_mask_neg": X_negation_sample['attention_mask'][indices]}

                elif 'joint' in type_:
                    acc = acc[:,None]
                    y_batch = np.concatenate((acc[indices], y_pred[indices][:, 1:]), axis=1)
                logger.info("y_batch shape: {}".format(y_batch.shape))
                #X_batch, y_batch, X_conf = f_(tokenizer, X_unlabeled_sample, y_mean, y_var, acc, unsup_size, len(labels), y_T=y_T, type_=type_) 

            probs = y_val[indices]
            probs_rat = y_rat[indices]
            cls = list(acc[indices])
            logger.info(cls)
            X_conf = np.ones((len(y_batch), max_seq_length))
            log_probs = (probs+1e-10) #+(1-y_batch[:,0])*np.log(1-probs+1e-10))
            log_rationale = (probs_rat+1e-10)
            if 'rwt' in type_: #re-weight labels
                X_conf[:,0] = np.where(log_probs>0, log_probs, 0.00000001)
                if 'norm' in type_:
                    X_conf[:,0] = tf.nn.softmax(X_conf[:,0], axis=0)
                if '_r_' in type_: #re-weight rationales
                    X_conf[:,1:] = np.where(log_rationale>0, log_rationale, 0.000000001) 
                    if 'norm' in type_:
                        X_conf[:,1:] = tf.nn.softmax(X_conf[:,1:], axis=0)
            #X_conf = np.ones((len(X_batch['input_ids']), max_seq_length))
            for i in range(len(cls)):
                X_conf[i,0] = class_weight[cls[i][0]]*X_conf[i,0]
            
            #logger.info ("Weights {}".format(X_conf[:10]))
            
            logger.info("X_connf shape: {}".format(X_conf.shape))
            if 'mtl' in type_:
                #model = model_student
                logger.info(y_batch.shape)
                model.fit(x=X_batch, y=[y_batch[:,0], y_batch[:,1:]], shuffle=True, epochs=unsup_epochs, validation_data=(X_dev, [y_dev[:,0], y_dev[:,1:]]), batch_size=unsup_batch_size*gpus, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='dense_3_classification_acc', patience=5, restore_best_weights=True)], sample_weight=[X_conf[:,0], X_conf[:,1:]])
                if 'fine_tune_teacher' in type_:
                    rat_loss = None
                    if 'focal' in type_:
                        rat_loss = SparseCategoricalFocalLoss(gamma=2)
                    else:
                        rat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                    loss_weights = None
                    if '_noexp' in type_:
                        loss_weights = [1.0, 0.0]
                    else:
                        loss_weights = [0.5, 0.5]
                   
                    with strategy.scope():
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), rat_loss], metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="dense_3_classification_acc")])#, tf.keras.metrics.SparseCategoricalAccuracy(name="token_acc")]) #, sample_weight_mode="temporal")
                    model.fit(x=X_train, y=[y_train[:,0], y_train[:,1:]], shuffle=True, epochs=unsup_epochs, validation_data=(X_dev, [y_dev[:,0], y_dev[:,1:]]), batch_size=unsup_batch_size*gpus, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_task_classifier_acc', patience=5, restore_best_weights=True)]) #, sample_weight=[X_conf[:,0], X_conf[:,1:]])
            elif type_ == 'joint':
                logger.info(type_)
                def custom_loss(y_true, y_pred):
                    logger.info(y_pred)
                    return kb.mean(y_true*y_pred, axis=-1)

                with strategy.scope():
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), loss={'task_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'rationale_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'rationale_task_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'l2_distance': custom_loss}, metrics={'task_classifier':[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], 'rationale_classifier':[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], 'rationale_task_classifier':[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], 'l2_distance':None})
                    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), loss={'task_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'rationale_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'rationale_task_classifier': None, 'l2_distance': custom_loss}, metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc"), tf.keras.metrics.SparseCategoricalAccuracy(name="acc"), tf.keras.metrics.SparseCategoricalAccuracy(name="acc"), tf.keras.metrics.Mean(name='mean')]) 
                #X_batch.update(X_rationale_batch)
                X_batch['input_ids_r'], X_batch['token_type_ids_r'], X_batch['attention_mask_r'] = X_rationale_batch['input_ids_r'], X_rationale_batch['token_type_ids_r'], X_rationale_batch['attention_mask_r']
                model.fit(x=X_batch, y=[y_batch[:,0], y_batch, y_batch[:, 0], np.ones(len(y_batch))], shuffle=True, epochs=unsup_epochs, validation_data=(X_dev, [y_dev[:,0], y_dev, y_dev[:,0], np.ones(len(y_dev))]), batch_size=unsup_batch_size*gpus, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_task_classifier_acc', patience=5, restore_best_weights=True)]) # class_weight=class_weight)
            elif 'joint_neg' in type_:
                logger.info("Training for without rationales")
                with strategy.scope():
                    def custom_loss(y_true, y_pred):
                        cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
                        tf.print(tf.size(y_true), tf.size(y_pred))
 
                        cce_loss = ((cce(y_true, y_pred))* 1/(unsup_batch_size*gpus))
                        l1_loss = tf.reduce_mean(tf.reduce_sum(tf.math.abs(y_pred),axis=0))
                        coh_loss = tf.reduce_mean(tf.reduce_sum(tf.math.abs(y_pred[1:]-y_pred[:-1]), axis=0))
                        #l2_loss = 0.0
                        #logger.info(l1_loss)
                        return cce_loss + 0.1*l1_loss + 0.01*coh_loss
                
                    def custom_loss_neg(y_true, y_pred):
                        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
                        return tf.reduce_sum(cce(y_true, y_pred))*(1/(unsup_batch_size*gpus))
                        
                    rat_loss = None
                    if 'focal' in type_:
                        rat_loss = SparseCategoricalFocalLoss(gamma=2)
                    else:
                        rat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

                    loss_weights = [1.0, 1.0, 1.0, 1.0]
                    '''
                    if '_noexp' in type_:
                        loss_weights = [1.0, 0, 0, 0]
                    if '_no_suffcomp' in type_:
                        loss_weights = [1.0, 1.0, 0, 0]
                    '''
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), loss={'task_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'rationale_classifier': custom_loss, 'rationale_task_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'not_rationale_task_classifier': custom_loss_neg}, metrics={'task_classifier':[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], 'rationale_classifier':[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], 'rationale_task_classifier':[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], 'not_rationale_task_classifier':None}, loss_weights=loss_weights)
                    X_batch['input_ids_r'], X_batch['token_type_ids_r'], X_batch['attention_mask_r'] = X_rationale_batch['input_ids_r'], X_rationale_batch['token_type_ids_r'], X_rationale_batch['attention_mask_r']
                    X_batch['input_ids_neg'], X_batch['token_type_ids_neg'], X_batch['attention_mask_neg'] = X_neg_rationale_batch['input_ids_neg'], X_neg_rationale_batch['token_type_ids_neg'], X_neg_rationale_batch['attention_mask_neg']
                    model.fit(x=X_batch, y=[y_batch[:,0], y_batch[:,1:], y_batch[:, 0], np.full((len(y_batch),len(labels)), 1/len(labels))], shuffle=True, epochs=unsup_epochs, validation_data=(X_dev, [y_dev[:,0], y_dev[:,1:], y_dev[:,0], np.full((len(y_dev), len(labels)), 1/len(labels))]), batch_size=unsup_batch_size*gpus, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)], sample_weight=[X_conf[:,0], X_conf[:,1:], X_conf[:,0], np.ones((len(y_batch)))]) # class_weight=class_weight) 
                    
                if 'fine_tune_teacher' in type_:    
                    rat_loss = None
                    if 'focal' in type_:
                        rat_loss = SparseCategoricalFocalLoss(gamma=2)
                    else:
                        rat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                   
                    with strategy.scope():
                        loss_weights = [1.0, 1.0, 1.0, 1.0]
                        '''
                        if '_noexp' in type_:
                            loss_weights = [1.0, 0, 0, 0]
                        elif '_no_suffcomp' in type_:
                            loss_weights = [1.0, 1.0, 0, 0]
                        '''
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), loss={'task_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'rationale_classifier': rat_loss, 'rationale_task_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'not_rationale_task_classifier': None}, metrics={'task_classifier':[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], 'rationale_classifier':[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], 'rationale_task_classifier':[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], 'not_rationale_task_classifier':None}, loss_weights=loss_weights)
                        y_neg = np.full((len(y_train),len(labels)), 1/len(labels))
                        model.fit(x=X_train, y=[y_train[:,0], y_train[:,1:], y_train[:,0], y_neg], shuffle=True, epochs=sup_epochs, validation_data=(X_dev, [y_dev[:,0], y_dev[:,1:], y_dev[:,0], np.full((len(y_dev), len(labels)), 1/len(labels))]), batch_size=unsup_batch_size*gpus, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_task_classifier_acc', patience=5, restore_best_weights=True)]) # class_weight=class_weight)
                        
            tf.keras.backend.clear_session()
            if not os.path.exists(model_file):
                model.save_weights(model_file)
                logger.info ("Model file saved to {}".format(model_file))

        model_student = model
        model_student.load_weights(model_file_best)
        if 'mtl' in type_:   
            acc, y_pred = model_student.predict(X_test)
            y_pred = np.argmax(y_pred, axis=-1)
            acc = np.argmax(acc, axis=-1)
            #logger.info("Micro score (task): {}".format(precision_recall_fscore_support(acc, y_test[:,0], average='micro')))
        elif 'joint' in type_:   
            out = model_student.predict(X_test)
            acc, y_pred, r_acc = out[0], out[1], out[2]
            logger.info("Raw logits: {}".format(acc))
            y_pred = np.argmax(y_pred, axis=-1)
            acc = np.argmax(acc, axis=-1)
            r_acc = np.argmax(r_acc, axis=-1)
            
        logger.info("Best task acc score: {}".format(precision_recall_fscore_support(acc, y_test[:,0], average='micro')))
        logger.info("Best token acc score: {}".format(precision_recall_fscore_support(y_pred, y_test[:,1:], average='macro')))

        pred, truth = [], []
        #sys.exit(1)
        test_pred =  y_pred #np.argmax(y_pred, axis=-1)
        logger.info("Printing prediction data on student model for run {}: {}".format(counter, test_pred))
               
        tp, fn, fp =  0, 0, 0
        pred_1, pred_0, truth_1, truth_0 = 0, 0, 0, 0
        for i in range(len(test_pred)):
            temp_p, temp_t, ct = [],[], 0
            temp = tokenizer.convert_ids_to_tokens(X_test["input_ids"][i])[1:]
            #logger.info("Test sample {}".format(temp))
            for j in range(0,len(test_pred[0])-1):
                if test_pred[i][j] == 1:
                    temp_p.append(temp[j])
                if y_test[i][j+1] == 1:
                    temp_t.append(temp[j])
            pred_1 += test_pred[i].sum()
            pred_0+= max_seq_length-pred_1

            truth_1 += y_test[i].sum()
            truth_0+= max_seq_length-truth_1
            pred.append(' '.join(temp_p))
            truth.append(' '.join(temp_t))
            for word in temp_p:
                if word in temp_t:
                    ct+=1
                    temp_t.remove(word)
                else:
                    fp+=1
            tp +=ct
            fn += (y_test[i].sum()-ct)
            
        p = tp/(tp+fp+0.0000001)
        r = tp/(tp+fn+0.0000001)
        logger.info("Token-level: {}".format((tp)/(tp+(0.5*(fp+fn)))))
        logger.info("Rationale coverage (recall): {}".format(r))
        logger.info("Token Precision: {}".format(p))
        logger.info("Token overlap: {}".format(tp/(tp+fp+fn)))
        
        score1, score2, score3, score4 = 0.0, 0.0, 0.0, 0.0
        for i in range(len(pred)):
            score1 += nltk.translate.bleu_score.sentence_bleu([truth[i].split()],pred[i].split(), weights=(1, 0, 0, 0))
            score2 += nltk.translate.bleu_score.sentence_bleu([truth[i].split()],pred[i].split(), weights=(0, 1, 0, 0))
            score3 += nltk.translate.bleu_score.sentence_bleu([truth[i].split()],pred[i].split(), weights=(0, 0, 1, 0))
            score4 += nltk.translate.bleu_score.sentence_bleu([truth[i].split()],pred[i].split(), weights=(0, 0, 0, 1))
        
        logger.info("BLEU-1 score of rationales on test set (student model): {} ".format(score1/len(pred)))
        logger.info("BLEU-2 score of rationales on test set (student model): {} ".format(score2/len(pred)))
        logger.info("BLEU-3 score of rationales on test set (student model): {} ".format(score3/len(pred)))
        logger.info("BLEU-4 score of rationales on test set (student model): {} ".format(score4/len(pred)))

        data = []
        for i in range(len(X_test["input_ids"])):
            text = tokenizer.decode(X_test["input_ids"][i])
            temp = dict()
            temp['text'] = text
            temp['truth'] = truth[i]
            temp['pred'] = pred[i]
            temp['score'] = nltk.translate.bleu_score.sentence_bleu([truth[i].split()],pred[i].split())
            data.append(temp)
        
        with open(os.path.join(model_dir, 'rationale_output_test_'+type_+'.json'), 'w') as f:
            json.dump(data, f)

        logger.info ("Best accuracy (task) across all self-training iterations {}".format(max_best_acc))
        
 
        
       
