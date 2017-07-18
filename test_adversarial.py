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

def test_adversarial_performance(signature):
	db = pickledb.load('databases/generated_adversarial_images.db', True)

	batch_size = 100

	fetched = db.get(signature)
	saved_weights_path = fetched["saved_weights_path"]
	print "prediction for {}".format(saved_weights_path)


	tf.reset_default_graph()
	images_meta = pd.read_csv("/media/data_cifs/danshiebler/data/adversarial/images.csv")
	file_names =  images_meta["ImageId"].values

	adv_image_dict = np.load("/media/data_cifs/danshiebler/data/adversarial/adversarial_images/{}.npy".format(signature)).item()
	data_X_adv = np.vstack([adv_image_dict[fname][None,...] for fname in file_names])
	
	image_dict = np.load("/media/data_cifs/danshiebler/data/adversarial/image_dict.npy").item()
	data_X = np.vstack([image_dict[fname][None,...] for fname in file_names])

	print data_X_adv.shape, data_X.shape

	with tf.device("/gpu:0"):
		with tf.variable_scope('cnn'):
			vgg, x = tfhf.get_vgg_x()


			sess = tfhf.initialize_session(saved_weights_path)

			# Run the predictions
			file_to_prob = {}
			for X, kind in [(data_X, "normal"), (data_X_adv, "adv")]:
				file_to_prob[kind] = {}
				for i in tqdm(range(0, X.shape[0], batch_size)):
					batch_data, batch_names = X[i:i+batch_size], file_names[i:i+batch_size]
					probs = sess.run(vgg.fc8, feed_dict={x:batch_data})
					for j in range(len(batch_names)):
						file_to_prob[kind][batch_names[j]] = probs[j]
			np.save("/media/data_cifs/danshiebler/data/adversarial/adversarial_performances/{}.npy".format(signature), file_to_prob)

if __name__ == "__main__":
	pass
	# signature = "1500345943"
	# test_adversarial_performance(signature)

	# signature = "1500344185"
	# test_adversarial_performance(signature)










