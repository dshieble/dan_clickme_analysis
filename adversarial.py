# GENERATE ADVERSARIAL IMAGES FOR A GIVEN SAVED WEIGHTS MODEL

import os
import tensorflow as tf
import os
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, CarliniWagnerL2, SaliencyMapMethod, VirtualAdversarialMethod
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
from scipy.misc import imread
from scipy.misc import imsave
import tensorflow_helpers as tfhf
import models.slim_inception_v3 as inception
from ops.get_slim_ops import inception_input_processing

import helper_functions as hf
from collections import Counter
   
from config import clickMeConfig, InceptionConfig
import baseline_vgg16 as vgg16
from tqdm import tqdm
import time
import tensorflow_helpers as tfhf
import pickledb



def generate_adversarial_images(saved_weights_path, model_kind):
	config = clickMeConfig()
	incfg = InceptionConfig()
	db = pickledb.load('/media/data_cifs/danshiebler/databases/generated_adversarial_images.db', True)


	signature = str(int(time.time()))

	image_width       = 224
	image_height      = 224
	batch_size        = 10

	batch_shape = [batch_size, image_height, image_width, 3]

	tf.reset_default_graph()
	data_X, image_dict, file_names, images_meta = hf.get_adversarial_data()
	assert np.std(data_X) > 0 and np.std(data_X) < 1
	print "np.mean(data_X), np.std(data_X)", np.mean(data_X), np.std(data_X)


	#  TODO: ONCE YOU HAVE A BUNCH OF MODELS TO COMPARE - JUST RUN LIKE 15 ADVERSARIAL MODELS

	# attack_methods = [CarliniWagnerL2, FastGradientMethod, BasicIterativeMethod, SaliencyMapMethod]#, VirtualAdversarialMethod]
	# attack_name_list = ["CarliniWagnerL2", "FastGradientMethod", "BasicIterativeMethod", "SaliencyMapMethod"]#, "VirtualAdversarialMethod"]
	# attack_kwargs_list = [{"nb_classes":1000, "learning_rate":1e-2, "max_iterations":100}, {}, {}]


	attack_methods = [CarliniWagnerL2, FastGradientMethod, BasicIterativeMethod]#, VirtualAdversarialMethod]
	attack_name_list = ["CarliniWagnerL2", "FastGradientMethod", "BasicIterativeMethod"]#, "VirtualAdversarialMethod"]
	attack_kwargs_list = [{"nb_classes":1000, "learning_rate":8e-2, 'initial_const':1e-1, "max_iterations":100}, {"eps":0.05}, {"eps":0.05}]

	attack_methods = attack_methods[1:]
	attack_name_list = attack_name_list[1:]
	attack_kwargs_list = attack_kwargs_list[1:]

	# attack_methods += [FastGradientMethod, BasicIterativeMethod]
	# attack_name_list += ["FastGradientMethod_{}".format(e), "BasicIterativeMethod_{}".format(e)]
	# attack_kwargs_list += [{"eps":eps}, {"eps":eps}]


	# attack_methods += [CarliniWagnerL2]
	# attack_name_list += ["CarliniWagnerL2"]
	# attack_kwargs_list += [{}]

	# print attack_name_list
	# attack_methods = attack_methods[:2]
	# attack_name_list = attack_name_list[:2]
	# attack_kwargs_list = attack_kwargs_list[:2]


	x_adv_list = []

	with tf.device("/gpu:0"):
		
		x = tf.placeholder(tf.float32, (batch_size, 224, 224, 3), name="x") #the input variable

		if model_kind == "vgg":

			# Build the VGG initially
			with tf.variable_scope('cnn_'):
				vgg = vgg16.Vgg16(vgg16_npy_path=None, fine_tune_layers=[])
				vgg.build(x, output_shape=config.output_shape, train_mode=tf.Variable(False, name='training'))


			def model(x): 
				with tf.variable_scope('cnn_'):
					# Build the vgg with reused variables
					tf.get_variable_scope().reuse_variables()
					vgg = vgg16.Vgg16(vgg16_npy_path=None, fine_tune_layers=[])
					vgg.build(x, output_shape=config.output_shape, train_mode=tf.Variable(False, name='training'))

				return vgg.fc8/tf.norm(vgg.fc8)


		elif model_kind == "inception":
			logits = tfhf.inception_model(x, saved_weights_path)

			def model(x): 
				with tf.variable_scope("", reuse=True):
					# tf.get_variable_scope().reuse_variables()
					return tfhf.inception_model(x, saved_weights_path)

		else:
			assert False

		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

		# Load the fast gradient computation graph
		for attack_method, kwargs in zip(attack_methods, attack_kwargs_list):	
			attack  = attack_method(model, sess=sess)
			x_adv = attack.generate(x, **kwargs) 
			x_adv_list.append(x_adv)

		# Load the saved weights and initialize the tensorflow graph
		if model_kind == "vgg":
			sess = tfhf.initialize_session_vgg(saved_weights_path, tf.trainable_variables(), sess=sess)
		elif model_kind == "inception":
			sess = tfhf.initialize_session_inception(saved_weights_path, sess=sess)	



		file_to_adv = {}
		for i in tqdm(range(0, data_X.shape[0], batch_size)):
			batch_data, batch_names = data_X[i:i+batch_size], file_names[i:i+batch_size]
			generated_adversarial_images = {}
			for attack_name, x_adv in zip(attack_name_list, x_adv_list):
				print "RUNNING {}".format(attack_name)
				adv = sess.run(x_adv, feed_dict={x:batch_data})
				generated_adversarial_images[attack_name] = adv
				print "adv stats", np.mean(batch_data),  np.mean(adv), np.mean(batch_data - adv), np.std(batch_data - adv) ####



			for j in range(len(batch_names)):
				file_to_adv[batch_names[j]] = {}
				for attack_name in attack_name_list:
					file_to_adv[batch_names[j]][attack_name] = generated_adversarial_images[attack_name][j]

		np.save("/media/data_cifs/danshiebler/data/adversarial/adversarial_images/{}.npy".format(signature), file_to_adv)
		db.set(signature, {"saved_weights_path":saved_weights_path, "model_kind":model_kind, "attack_name_list":attack_name_list})
		# print tf.trainable_variables()
	return signature

if __name__ == "__main__":
	pass
	# saved_weights_path = ""
	# saved_weights_path = "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/baseline_001_50000_2017_06_07_10_19_47/model_252000.ckpt-252000"
	# model_kind = "vgg"
	# # saved_weights_path =  "/media/data_cifs/clicktionary/clickme_experiment/attention_gradient_checkpoints/gradient_001_130671_2017_07_15_15_01_12/model_44000.ckpt-44000"
	# print generate_adversarial_images(saved_weights_path, model_kind)


	# saved_weights_path = "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/baseline_001_50000_2017_06_07_10_19_47/model_252000.ckpt-252000"
	# generate_adversarial_images(saved_weights_path)


	# saved_weights_path = "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_124720_2017_06_07_10_19_49/model_162000.ckpt-162000"
	# generate_adversarial_images(saved_weights_path)



