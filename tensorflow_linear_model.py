import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import h5py
import tensorflow as tf
import h5py
import pandas as pd
import numpy as np
import helper_functions as hf
import matplotlib.pyplot as plt
from scipy.misc import imread
from collections import Counter
   
from tqdm import tqdm
import time
import tensorflow_helpers as tfhf


class TFLinearModel():

	def __init__(self, regularization_weight=1e-5, lr=0.001, clip=10, num_epochs=100, 
						batch_size=100, tol=0.0000000001, print_progress=False, sess=None, thresh=0.5, lookback=2):

		"""
			A tensorflow multi-output logistic regression with a scikit learn interface

		"""
		tf.reset_default_graph()
		self.regularization_weight = regularization_weight
		self.lr = lr
		self.clip = clip
		self.num_epochs = num_epochs
		self.batch_size = batch_size
		self.tol = tol
		self.print_progress = print_progress
		self.sess = sess
		self.thresh = 0.5
		self.built = False
		self.lookback = lookback

	def build(self, input_size, output_size):
		"""
			Build the tensorflow graph and set up the loss and training schema
		"""
		self.built = True
		self.input_size = input_size
		self.output_size = output_size

		with tf.variable_scope('linear_model'):
			self.x = tf.placeholder(tf.float32, (None, self.input_size), name="x") #the input variable
			self.y = tf.placeholder(tf.float32, (None, self.output_size), name="y") #the output variable

			self.W = tf.get_variable("weights", shape=(self.input_size, self.output_size), dtype=np.float32)
			self.b = tf.get_variable("bias", initializer=np.zeros(self.output_size, dtype=np.float32))
			self.raw_output = tf.nn.bias_add(tf.matmul(self.x, self.W), self.b) # The pre-sigmoid ouput
			self.output = tf.sigmoid(self.raw_output)

			self.reg_loss = tf.nn.l2_loss(self.W)*self.regularization_weight
			self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.raw_output)
			self.loss = self.reg_loss + self.cross_entropy


			opt_func = tf.train.AdamOptimizer(learning_rate=self.lr)
			gvs = opt_func.compute_gradients(self.loss)
			grads = [[tf.clip_by_value(grad, -self.clip, self.clip), var] for grad, var in gvs if not grad is None]

			# if self.print_progress:
			# 	#If we set this flag, print the mean values of the cost, layer weights, and gradients. Helps spot zero activations and crazy gradient values.
			# 	vars = tf.trainable_variables()
			# 	#Print the cost of each epoch
			# 	self.loss = tf.Print(self.loss, [self.loss], "COST")
			# 	#Print the mean values of the gradients
			# 	for i, g in enumerate(grads):
			# 		self.loss = tf.Print(self.loss, [tf.reduce_mean(g[0])], "MEAN GRADIENT FOR {}".format(g[1].name))
			# 	#Print the mean values of the weights
			# 	for i, v in enumerate(vars):
			# 		self.loss = tf.Print(self.loss, [tf.reduce_mean(v)], "MEAN VALUE OF {}".format(v.name))

			self.updt = opt_func.apply_gradients(grads)
		if self.sess is None:
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
			self.sess.run(
				tf.group(
					tf.global_variables_initializer(),
					tf.local_variables_initializer()))

	def fit(self, X, y):
		"""
			Fit the model to data. 
		"""

		# Build the model if necessary
		if not self.built: self.build(X.shape[1], y.shape[1])
		self.epoch_learning_curve = []
		self.learning_curve = [] 

		# Run the training for each epoch
		for epoch in range(self.num_epochs):
			losses = []
			# For each batch, compute the loss and update the network
			for i in range(0, X.shape[0], self.batch_size):
				xb, yb = X[i:i+self.batch_size], y[i:i+self.batch_size]
				_, loss_val = self.sess.run([self.updt, self.loss], feed_dict={self.x: xb, self.y: yb})
				losses += loss_val.tolist()

			# Account for the losses so far and record them. Early Stop if necessary
			mean_loss = np.mean(losses)
			self.epoch_learning_curve.append(mean_loss)
			self.learning_curve += losses
			print "Epoch {} completed! Mean loss of: {}".format(epoch, mean_loss)
			if len(self.epoch_learning_curve) > self.lookback and (
								not (mean_loss < (np.mean(self.epoch_learning_curve[-self.lookback]) - self.tol))):
				print "Loss has stopped decreasing, breaking {}".format(self.epoch_learning_curve[-5:])
				break

	def predict_proba(self, X):
		out = []
		for i in tqdm(range(0, X.shape[0], self.batch_size)):
			output_vals = self.sess.run(self.output, feed_dict={self.x: X[i:i+self.batch_size]})
			for o in output_vals:
				out.append(o)
		return np.vstack(out)

	def predict(self, X):
		probs = self.predict_proba(X)
		return probs > self.thresh



	def get_vars(self):
		return self.x, self.y, self.output, self.cross_entropy, self.updt






