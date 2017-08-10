# GENERATE ADVERSARIAL IMAGES FOR A GIVEN SAVED WEIGHTS MODEL

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import os
from cleverhans.attacks import FastGradientMethod
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
	batch_size        = 100

	eps = 0.03
	batch_shape = [batch_size, image_height, image_width, 3]

	tf.reset_default_graph()
	data_X, image_dict, file_names, images_meta = hf.get_adversarial_data()
	assert np.std(data_X) > 0 and np.std(data_X) < 1
	print "np.mean(data_X), np.std(data_X)", np.mean(data_X), np.std(data_X)
	
	with tf.device("/gpu:0"):
		
		x = tf.placeholder(tf.float32, (None, 224, 224, 3), name="x") #the input variable

		if model_kind == "vgg":
			with tf.variable_scope('cnn'):
				vgg = vgg16.Vgg16(vgg16_npy_path=None, fine_tune_layers=[])
				vgg.build(x, output_shape=config.output_shape, train_mode=tf.Variable(False, name='training'))

			def model(x): return vgg.fc8/tf.norm(vgg.fc8)
			# Load the fast gradient computation graph
			fgsm  = FastGradientMethod(model)
			x_adv = fgsm.generate(x, eps=eps, clip_min=-1., clip_max=1., ord=np.inf) 

			# Load the saved weights and initialize the tensorflow graph
			sess = tfhf.initialize_session_vgg(saved_weights_path, tf.trainable_variables())
		elif model_kind == "inception":

			logits = tfhf.inception_model(x, saved_weights_path)
			def model(x): return logits
			# Load the fast gradient computation graph
			fgsm  = FastGradientMethod(model)
			x_adv = fgsm.generate(x, eps=eps, clip_min=-1., clip_max=1., ord=np.inf) 

			# Load the saved weights and initialize the tensorflow graph
			sess = tfhf.initialize_session_inception(saved_weights_path)	

		else:
			assert False

		# # Load the fast gradient computation graph
		# fgsm  = FastGradientMethod(model)
		# x_adv = fgsm.generate(x, eps=eps, clip_min=-1., clip_max=1., ord=np.inf) 

		# # Load the saved weights and initialize the tensorflow graph
		# sess = tfhf.initialize_session(saved_weights_path, tf.trainable_variables())		

		file_to_adv = {}
		for i in tqdm(range(0, data_X.shape[0], batch_size)):
			batch_data, batch_names = data_X[i:i+batch_size], file_names[i:i+batch_size]
			adv = sess.run(x_adv, feed_dict={x:batch_data})

			########################################################################################
			print "adv stats", np.mean(batch_data),  np.mean(adv), np.mean(batch_data - adv), np.std(batch_data - adv) ####

			########################################################################################




			for j in range(len(batch_names)):
				file_to_adv[batch_names[j]] = adv[j]

		np.save("/media/data_cifs/danshiebler/data/adversarial/adversarial_images/{}.npy".format(signature), file_to_adv)
		db.set(signature, {"saved_weights_path":saved_weights_path, "model_kind":model_kind })
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



