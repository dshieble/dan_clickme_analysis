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
from tensorflow.contrib.slim.nets import inception
import tensorflow_helpers as tfhf


import helper_functions as hf
from collections import Counter
   
from config import clickMeConfig
import baseline_vgg16 as vgg16
from tqdm import tqdm
import time
import tensorflow_helpers as tfhf


import pickledb


def generate_adversarial_images(saved_weights_path):
	#TODO: Have this consume imagenet images, and then write those to a new dataset
	db = pickledb.load('databases/generated_adversarial_images.db', True)


	signature = str(int(time.time()))

	max_epsilon       = 16.0
	image_width       = 224
	image_height      = 224
	batch_size        = 16

	eps = 2.0 * max_epsilon / 255.0
	batch_shape = [batch_size, image_height, image_width, 3]

	tf.reset_default_graph()
	data_X, image_dict, file_names, images_meta = hf.get_adversarial_data()

	with tf.device("/gpu:0"):
		with tf.variable_scope('cnn'):
			vgg, x = tfhf.get_vgg_x()

			def model(x): 
				return vgg.fc8

			fgsm  = FastGradientMethod(model)
			x_adv = fgsm.generate(x, eps=eps, clip_min=-1., clip_max=1.)
			sess = tfhf.initialize_session(saved_weights_path)

			file_to_adv = {}
			for i in tqdm(range(0, data_X.shape[0], batch_size)):
				batch_data, batch_names = data_X[i:i+batch_size], file_names[i:i+batch_size]
				adv = sess.run(x_adv, feed_dict={x:batch_data})
				for j in range(len(batch_names)):
					file_to_adv[batch_names[j]] = adv[j]
			np.save("/media/data_cifs/danshiebler/data/adversarial/adversarial_images/{}.npy".format(signature), file_to_adv)
			db.set(signature, {"saved_weights_path":saved_weights_path})
	return signature

if __name__ == "__main__":
	saved_weights_path = "/media/data_cifs/clicktionary/clickme_experiment/attention_gradient_checkpoints/gradient_001_130671_2017_07_15_15_01_12/model_44000.ckpt-44000"
	print generate_adversarial_images(saved_weights_path)


	# saved_weights_path = "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/baseline_001_50000_2017_06_07_10_19_47/model_252000.ckpt-252000"
	# generate_adversarial_images(saved_weights_path)


	# saved_weights_path = "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_124720_2017_06_07_10_19_49/model_162000.ckpt-162000"
	# generate_adversarial_images(saved_weights_path)



