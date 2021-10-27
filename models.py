"""
Author: Meghana Bhat (bhat.89@osu.edu)
Code for Self-training for Rationale using few-shot learning.
This code base is adapted from UST (https://github.com/microsoft/UST)
"""

import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Lambda
from tensorflow.keras.models import Model
import logging
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
#from tensorflow_addons.layers import CRF
from tf2crf import CRF

logger = logging.getLogger('STRationale')


def construct_teacher_task(encoder, max_seq_length, classes, dense_dropout=0.5, attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2):

    input_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_seq_length,), dtype=tf.int32, name="attention_mask")
    token_type_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="token_type_ids")
    config = encoder.config

    output = encoder(input_ids, token_type_ids=token_type_ids,  attention_mask=attention_mask)
    output = Dropout(dense_dropout)(output[0])[:,0]
    output = Dense(classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range))(output)
    model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=output)
    return model


def penalized_loss(noise):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_true - noise), axis=-1)
    return loss

def loss_selection(select, model, output):
    if select == 'student':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), loss={'task_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'rationale_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'rationale_task_classifier': penalized_loss(noise=output)}, metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]) 
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), loss={'task_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'rationale_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'rationale_task_classifier': None}, metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]) 
 
def construct_teacher_joint_neg(TFModel, Config, pt_teacher_checkpoint, max_seq_length, classes, dense_dropout=0.5, attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2):

    config = Config.from_pretrained(pt_teacher_checkpoint, num_labels=classes, from_pt=True)
    config.attention_probs_dropout_prob = attention_probs_dropout_prob
    config.hidden_dropout_prob = hidden_dropout_prob
    encoder = TFModel.from_pretrained(pt_teacher_checkpoint, config=config, name="teacher", from_pt=True)

    input_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_seq_length,), dtype=tf.int32, name="attention_mask")
    token_type_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="token_type_ids")

    input_ids_token = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids_r")
    attention_mask_r = Input(shape=(max_seq_length,), dtype=tf.int32, name="attention_mask_r")
    token_type_ids_r = Input(shape=(max_seq_length,), dtype=tf.int32, name="token_type_ids_r")
   
    input_ids_neg = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids_neg")
    attention_mask_neg = Input(shape=(max_seq_length,), dtype=tf.int32, name="attention_mask_neg")
    token_type_ids_neg = Input(shape=(max_seq_length,), dtype=tf.int32, name="token_type_ids_neg")
    
    out_neg = encoder(input_ids_neg, token_type_ids=token_type_ids_neg,  attention_mask=attention_mask_neg)
    output = encoder(input_ids, token_type_ids=token_type_ids,  attention_mask=attention_mask)
    output_ru = encoder(input_ids_token, token_type_ids=token_type_ids_r,  attention_mask=attention_mask_r)
    
    output_label_ = Dropout(dense_dropout)(output[0])[:,0]
    output_label = Dense(classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range), name='task_classifier')(output_label_)

    output_r = Dropout(dense_dropout)(output[0])[:,1:]
    output_r = Dense(2, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range), name='rationale_classifier')(output_r)
  
    output_ru_ = Dropout(dense_dropout)(output_ru[0])[:,0]
    output_ru = Dense(classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range), name='rationale_task_classifier')(output_ru_)

    output_neg_ = Dropout(dense_dropout)(out_neg[0])[:,0]
    output_neg = Dense(classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range), activation='softmax', name='not_rationale_task_classifier')(output_neg_)

    #output_label_l2 = Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range))(output_label_)
    #output_ru_l2 = Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range))(output_ru_)
    #l2_layer = Lambda(lambda tensors:K.square(tensors[0] - tensors[1])+0.01*K.sum(K.cast_to_floatx(K.argmax(tensors[2], axis=-1))), name="l2_distance")
    #l2_layer = Lambda(lambda tensors:K.square(tensors[0] - tensors[1])+0.01*tf.math.reduce_sum(tf.math.abs(tensors[2]),axis=-1), name="l2_distance")
    #l2_distance = l2_layer([output_label_l2, output_ru_l2, output_r])
 
    model = tf.keras.Model(inputs=[input_ids,token_type_ids, attention_mask, input_ids_token, token_type_ids_r, attention_mask_r, input_ids_neg, token_type_ids_neg, attention_mask_neg], outputs=[output_label, output_r, output_ru, output_neg])
 
    return model

def construct_teacher_joint(TFModel, Config, pt_teacher_checkpoint, max_seq_length, classes, dense_dropout=0.5, attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2):

    config = Config.from_pretrained(pt_teacher_checkpoint, num_labels=classes, from_pt=True)
    config.attention_probs_dropout_prob = attention_probs_dropout_prob
    config.hidden_dropout_prob = hidden_dropout_prob
    encoder = TFModel.from_pretrained(pt_teacher_checkpoint, config=config, name="teacher", from_pt=True)

    input_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_seq_length,), dtype=tf.int32, name="attention_mask")
    token_type_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="token_type_ids")

    input_ids_token = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids_r")
    attention_mask_r = Input(shape=(max_seq_length,), dtype=tf.int32, name="attention_mask_r")
    token_type_ids_r = Input(shape=(max_seq_length,), dtype=tf.int32, name="token_type_ids_r")
    
    output = encoder(input_ids, token_type_ids=token_type_ids,  attention_mask=attention_mask)
    output_ru = encoder(input_ids_token, token_type_ids=token_type_ids_r,  attention_mask=attention_mask_r)
    
    output_label_ = Dropout(dense_dropout)(output[0])[:,0]
    output_label = Dense(classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range), name='task_classifier')(output_label_)

    output_r = Dropout(dense_dropout)(output[0])[:,1:]
    output_r = Dense(2, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range), name='rationale_classifier')(output_r)
  
    output_ru_ = Dropout(dense_dropout)(output_ru[0])[:,0]
    output_ru = Dense(classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range), name='rationale_task_classifier')(output_ru_)
 
    output_label_l2 = Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range))(output_label_)
    output_ru_l2 = Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range))(output_ru_)

    #l2_layer = Lambda(lambda tensors:K.square(tensors[0] - tensors[1])+0.01*K.sum(K.cast_to_floatx(K.argmax(tensors[2], axis=-1))), name="l2_distance")
    l2_layer = Lambda(lambda tensors:K.square(tensors[0] - tensors[1])+0.01*tf.math.reduce_sum(tf.math.abs(tensors[2]),axis=-1), name="l2_distance")
    l2_distance = l2_layer([output_label_l2, output_ru_l2, output_r])
 
    model = tf.keras.Model(inputs=[input_ids,token_type_ids, attention_mask, input_ids_token, token_type_ids_r, attention_mask_r], outputs=[output_label, output_r, output_ru, l2_distance])
 
    return model

def construct_teacher(TFModel, Config, pt_teacher_checkpoint, max_seq_length, classes, dense_dropout=0.5, attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2):

    config = Config.from_pretrained(pt_teacher_checkpoint, num_labels=classes, from_pt=True)
    config.attention_probs_dropout_prob = attention_probs_dropout_prob
    config.hidden_dropout_prob = hidden_dropout_prob
    encoder = TFModel.from_pretrained(pt_teacher_checkpoint, config=config, name="teacher", from_pt=True)

    input_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_seq_length,), dtype=tf.int32, name="attention_mask")
    token_type_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="token_type_ids")

    output = encoder(input_ids, token_type_ids=token_type_ids,  attention_mask=attention_mask)
    output = Dropout(dense_dropout)(output[0])
    output = Dense(classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range))(output)
    model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=output)
    return model

def construct_teacher_mtl(TFModel, Config, pt_teacher_checkpoint, max_seq_length, classes, dense_dropout=0.5, attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2):

    config = Config.from_pretrained(pt_teacher_checkpoint, num_labels=classes, from_pt=True)
    config.attention_probs_dropout_prob = attention_probs_dropout_prob
    config.hidden_dropout_prob = hidden_dropout_prob
    encoder = TFModel.from_pretrained(pt_teacher_checkpoint, config=config, name="teacher", from_pt=True)

    input_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_seq_length,), dtype=tf.int32, name="attention_mask")
    token_type_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="token_type_ids")

    output = encoder(input_ids, token_type_ids=token_type_ids,  attention_mask=attention_mask)

    output_sent = Dropout(dense_dropout)(output[0])[:,0]
    output_sent = Dense(classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range))(output_sent)

    output_token = Dropout(dense_dropout)(output[0])[:,1:]
    output_token = Dense(classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range))(output_token)
    #logger.info(" outputs shape: {}, {}".format(output_sent.shape, output_token.shape)) 
    model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=[output_sent, output_token])
    return model


