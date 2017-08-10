# Called by run_adversarial

# TEST THE PERFORMANCE OF A GIVEN MODEL ON AN ADVERSARIAL DATASET

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

def test_adversarial_performance(signature):
	tf.reset_default_graph()
	config = clickMeConfig()
	incfg = InceptionConfig()
	db = pickledb.load('/media/data_cifs/danshiebler/databases/generated_adversarial_images.db', True)

	batch_size = 100

	fetched = db.get(signature)
	model_kind = fetched["model_kind"]
	saved_weights_path = fetched["saved_weights_path"]
	print "prediction for {}".format(saved_weights_path)


	data_X, image_dict, file_names, images_meta = hf.get_adversarial_data()

	adv_image_dict = np.load("/media/data_cifs/danshiebler/data/adversarial/adversarial_images/{}.npy".format(signature)).item()
	data_X_adv = np.vstack([adv_image_dict[fname][None,...] for fname in file_names])
	
	print "\n\n\n\n\n"
	print data_X_adv.shape, data_X.shape, np.mean(data_X_adv), np.mean(data_X), np.std(data_X_adv), np.std(data_X)  
	print "\n\n\n\n\n"


	with tf.device("/gpu:0"):

		if model_kind == "vgg": # instantiate the vgg network
			with tf.variable_scope('cnn'):
				vgg, x = tfhf.get_vgg_x()
				predictions = vgg.fc8
				sess = tfhf.initialize_session_vgg(saved_weights_path)	
		elif model_kind == "inception":# instantiate the inception network
			x = tf.placeholder(tf.float32, (None, 224, 224, 3), name="x") #the input variable
			predictions = tfhf.inception_model(x, saved_weights_path)

			sess = tfhf.initialize_session_inception(saved_weights_path)	

		else:
			assert False

		# Run the predictions
		file_to_prob = {}
		for X, kind in [(data_X, "normal"), (data_X_adv, "adv")]:
			file_to_prob[kind] = {}
			for i in tqdm(range(0, X.shape[0], batch_size)):
				batch_data, batch_names = X[i:i+batch_size], file_names[i:i+batch_size]
				probs = sess.run(predictions, feed_dict={x:batch_data})
				for j in range(len(batch_names)):
					file_to_prob[kind][batch_names[j]] = probs[j]
		np.save("/media/data_cifs/danshiebler/data/adversarial/adversarial_performances/{}.npy".format(signature), file_to_prob)

if __name__ == "__main__":
	signature = "1500433264"
	test_adversarial_performance(signature)

	# signature = "1500344185"
	# test_adversarial_performance(signature)










