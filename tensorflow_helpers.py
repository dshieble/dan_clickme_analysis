import os
import sys
sys.path.append("../clickme_modeling")
sys.path.append("../clickme_modeling/models")

import tensorflow as tf
import h5py
import pandas as pd
import numpy as np
import helper_functions as hf
import matplotlib.pyplot as plt
from scipy.misc import imread
from collections import Counter
   
from config import clickMeConfig, InceptionConfig
import baseline_vgg16 as vgg16
from tqdm import tqdm
import time
import tensorflow_helpers as tfhf
from ops.get_slim_ops import inception_input_processing
import models.slim_inception_v3 as inception

import pickledb
db = pickledb.load('databases/signature_to_saved_data_path.db', True)
config = clickMeConfig()


def get_vgg_x():
	# Returns a vgg object and a placeholder x variable
	print "building the vgg..."
	vgg = vgg16.Vgg16(
		vgg16_npy_path=config.vgg16_weight_path,
		fine_tune_layers=config.fine_tune_layers)
	x = tf.placeholder(tf.float32, (None, 224, 224, 3), name="x") #the input variable
	validation_mode = tf.Variable(False, name='training')
	vgg.build(
		x, output_shape=config.output_shape,
		train_mode=validation_mode)
	print "vgg built!"
	return vgg, x

def initialize_session_vgg(saved_weights_path, vars=None):

	# Initialize the graph
	print "initializing the graph..."
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
	sess.run(
		tf.group(
			tf.global_variables_initializer(),
			tf.local_variables_initializer()))
	print "initialized the graph!"

	# Load the saved weights
	if saved_weights_path:
		if vars is None:
			vars = tf.trainable_variables()
		print "Restoring {} variables".format(len(vars))
		saver = tf.train.Saver(vars)
		saver.restore(sess, saved_weights_path)	
	return sess

def inception_model(x, saved_weights_path):
	with tf.contrib.slim.arg_scope(inception.inception_v3_arg_scope()):
		logits, _ = inception.inception_v3(inception_input_processing(x), is_training=True, dropout_keep_prob=1)
	return logits

def initialize_session_inception(saved_weights_path, sess=None):
	# Initialize the graph
	print "initializing the graph..."
	if not sess:
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
	sess.run(
		tf.group(
			tf.global_variables_initializer(),
			tf.local_variables_initializer()))
	print "initialized the graph!"

	# Load the saved weights
	if saved_weights_path:
		print "restoring {}...".format(saved_weights_path)
		saver = tf.train.Saver(tf.contrib.slim.get_model_variables())
		saver.restore(sess, saved_weights_path)	
		print "restored!"
	return sess

