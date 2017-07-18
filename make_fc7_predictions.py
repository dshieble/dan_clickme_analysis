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
db = pickledb.load('databases/fc7_features_signature.db', True)
config = clickMeConfig()



def run_predictions(saved_weights_path, device, batch_size, kind, ids_path):
	tf.reset_default_graph()
	data_X, _, file_names = hf.get_image_data()
	print data_X.shape, len(file_names)

	signature = str(int(time.time()))


	# Build the network
	with tf.device(device):
		with tf.variable_scope('cnn'):
			vgg, x = tfhf.get_vgg_x()
			sess = tfhf.initialize_session(saved_weights_path)

			# Run the predictions
			file_to_features = {}
			for i in tqdm(range(0, data_X.shape[0], batch_size)):
				batch_data, batch_names = data_X[i:i+batch_size], file_names[i:i+batch_size]
				features = sess.run(vgg.fc7, feed_dict={x:batch_data})
				for j in range(len(batch_names)):
					file_to_features[batch_names[j]] = features[j]
			np.save("{}/generated_feature_vectors/{}".format(hf.data_path, signature), file_to_features)
			db.set(signature, {"saved_weights_path":saved_weights_path, "kind":kind, "ids_path":ids_path})


# saved_weights_path = "/media/data_cifs/clickme/baseline_checkpoints/baseline_001_130081_2017_07_12_11_10_53/model_94000.ckpt-94000"
# gpu = 1
# run_predictions(saved_weights_path, gpu)



# saved_weights_path = "/media/data_cifs/clicktionary/clickme_experiment/attention_gradient_checkpoints/gradient_001_130671_2017_07_15_15_01_12/model_44000.ckpt-44000"
# device = '/gpu:0'
# batch_size = 100
# run_predictions(saved_weights_path, device, batch_size)

# saved_weights_path = "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/baseline_001_50000_2017_06_07_10_19_47/model_252000.ckpt-252000"
# device = '/gpu:0'
# batch_size = 100
# run_predictions(saved_weights_path, device, batch_size)

# saved_weights_path = "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_124720_2017_06_07_10_19_49/model_162000.ckpt-162000"
# device = '/gpu:0'
# batch_size = 100
# run_predictions(saved_weights_path, device, batch_size)




