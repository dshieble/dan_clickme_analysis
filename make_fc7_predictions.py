import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
sys.path.append("clickme_modeling")
sys.path.append("clickme_modeling/models")

import h5py
import tensorflow as tf
import h5py
import pandas as pd
import numpy as np
import helper_functions as hf
import matplotlib.pyplot as plt
from scipy.misc import imread
from collections import Counter
import models.slim_inception_v3 as inception

from config import clickMeConfig, InceptionConfig
import baseline_vgg16 as vgg16
from tqdm import tqdm
import time
import tensorflow_helpers as tfhf

import pickledb



def run_predictions(img_ID_to_DATA, saved_weights_path, device, batch_size, model_kind):
	tf.reset_default_graph()
	db = pickledb.load('databases/fc7_features_signature.db', True)
	config = clickMeConfig()
	incfg = InceptionConfig()

	signature = str(int(time.time()))

	# Build the network
	with tf.device(device):
		with tf.variable_scope('cnn'):
			
			if model_kind == "vgg": # instantiate the vgg network
				vgg, x = tfhf.get_vgg_x()
				feature_layer = vgg.fc7
				sess = tfhf.initialize_session_vgg(saved_weights_path)
			elif model_kind == "inception":# instantiate the inception network
				x = tf.placeholder(tf.float32, (None, 224, 224, 3), name="x") #the input variable

				with tf.contrib.slim.arg_scope(inception.inception_v3_arg_scope()):
					train_logits, end_points = inception.inception_v3(x, is_training=False)
				feature_layer = end_points['PreLogits']

				sess = tfhf.initialize_session_inception(saved_weights_path)
			else:
				assert False





			# Run the predictions
			file_to_features = {"train":{}, "val":{}}
			for kind in file_to_features:
				for i in tqdm(range(0, img_ID_to_DATA["{}_data".format(kind)].shape[0], batch_size)):
					batch_data, _, batch_names = hf.get_batch(img_ID_to_DATA, i, i+batch_size, kind)
					assert np.mean(batch_data) <= 1
					assert np.std(batch_data) <= 1
					features = sess.run(feature_layer, feed_dict={x:batch_data})
					for j in range(len(batch_names)):
						file_to_features[kind][batch_names[j]] = features[j]
			np.save("{}/generated_feature_vectors/{}.npy".format(hf.data_path, signature), file_to_features)
			db.set(signature, {"saved_weights_path":saved_weights_path})

if __name__ == "__main__":
	img_ID_to_DATA = h5py.File("{}/img_ID_to_DATA.h5py".format(hf.data_path))
	device = "/gpu:2"
	batch_size = 100

	saved_weights_paths = [
	# "/media/data_cifs/clicktionary/clickme_experiment/attgrad_vgg_checkpoints/gradient_-05_144023_2017_07_27_03_55_12/model_244000.ckpt-244000",
	# "/media/data_cifs/clicktionary/clickme_experiment/attgrad_vgg_checkpoints/gradient_-05_144023_2017_07_27_03_55_03/model_244000.ckpt-244000",
	# "/media/data_cifs/clicktionary/clickme_experiment/attgrad_vgg_checkpoints/gradient_0001_144023_2017_07_27_03_55_01/model_240000.ckpt-240000",
	"/media/data_cifs/clicktionary/clickme_experiment/attgrad_vgg_checkpoints/gradient_001_144023_2017_07_27_03_55_08/model_126000.ckpt-126000"
	]

	model_kinds = ["vgg"]*4

	for swp, mk in zip(saved_weights_paths, model_kinds):
		run_predictions(img_ID_to_DATA, swp, device, batch_size, mk)


	# "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_112341_2017_05_29_17_55_48/model_42000.ckpt-42000"
	# "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_112341_2017_05_15_22_53_23/model_80000.ckpt-80000",
	# "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_112341_2017_05_15_22_53_23/model_56000.ckpt-56000",
	# "/media/data_cifs/clicktionary/clickme_experiment/attention_gradient_checkpoints/gradient_001_130671_2017_07_15_15_01_12/model_44000.ckpt-44000",
	# "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/baseline_001_50000_2017_06_07_10_19_47/model_252000.ckpt-252000",
	# "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_124720_2017_06_07_10_19_49/model_162000.ckpt-162000",
