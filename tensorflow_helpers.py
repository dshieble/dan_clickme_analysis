import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
sys.path.append("clickme_modeling")
sys.path.append("clickme_modeling/models")

import tensorflow as tf
import h5py
import pandas as pd
import numpy as np
import helper_functions as hf
import matplotlib.pyplot as plt
from scipy.misc import imread
from collections import Counter
   
from config import clickMeConfig
import baseline_vgg16 as vgg16
from tqdm import tqdm
import time
import tensorflow_helpers as tfhf

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

def initialize_session(saved_weights_path):

	# Initialize the graph
	print "initializing the graph..."
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
	sess.run(
		tf.group(
			tf.global_variables_initializer(),
			tf.local_variables_initializer()))
	print "initialized the graph!"


	# Load the saved weights
	saver = tf.train.Saver(tf.global_variables())
	saver.restore(sess, saved_weights_path)	
	return sess


# def linear_model(input_size, output_size):
# 	"""
# 	Builds a linear model and returns:
# 		The output variable
# 		The placeholder x variable
# 		The placeholder y variable
# 		The updt object

# 	"""
# 	x = tf.placeholder(tf.float32, (None, input_size), name="x") #the input variable
# 	weights = tf.get_variable("weights", shape=(input_size, output_size), dtype=np.float32)
#     biases = tf.get_variable("bias", initializer=np.zeros(output_size, dtype=np.float32))
#     loss = tf.nn.l2_loss(weights)



