# GENERATE ADVERSARIAL IMAGES FOR A GIVEN SAVED WEIGHTS MODEL
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import os
import numpy as np
from scipy.stats import pearsonr


def tf_pearsonr(x1, x2):
	x1_mean = tf.reduce_mean(tf.reduce_mean(x1, axis=[-1]), keep_dims=True, axis=[-1])
	x2_mean = tf.reduce_mean(tf.reduce_mean(x2, axis=[-1]), keep_dims=True, axis=[-1])

	x1_flat = tf.reshape(x1, (-1, 224*224))
	x2_flat = tf.reshape(x2, (-1, 224*224))
	x1_flat_normed = x1_flat - x1_mean
	x2_flat_normed = x2_flat - x2_mean


	cov = tf.div(tf.reduce_sum(tf.multiply(x1_flat_normed, x2_flat_normed), -1), 224*224 - 1)
	x1_std = tf.sqrt(tf.div(tf.reduce_sum(tf.square(x1_flat - x1_mean), -1), 224*224 - 1))
	x2_std = tf.sqrt(tf.div(tf.reduce_sum(tf.square(x2_flat - x2_mean), -1), 224*224 - 1))

	corr = cov/(tf.multiply(x1_std, x2_std))
	return corr

x1 = tf.placeholder(tf.float32, (None, 224, 224), name="x") #the input variable
x2 = tf.placeholder(tf.float32, (None, 224, 224), name="x") #the input variable

corr = tf_pearsonr(x1, x2)


sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
sess.run(
	tf.group(
		tf.global_variables_initializer(),
		tf.local_variables_initializer()))
print "initialized the graph!"

x1_vals = np.random.random((32, 224,224))
x2_vals = x1_vals + np.random.random((32, 224,224))*0.5

print "TRUE", pearsonr(np.ravel(x1_vals), np.ravel(x2_vals))[0]
C = sess.run(corr, feed_dict={x1:x1_vals, x2:x2_vals})
print "EST", C

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# import tensorflow as tf
# import os
# from io import BytesIO
# import numpy as np
# import pandas as pd
# from PIL import Image
# from scipy.misc import imread
# from scipy.misc import imsave
# import tensorflow_helpers as tfhf
# import models.slim_inception_v3 as inception
# from ops.get_slim_ops import inception_input_processing


# import helper_functions as hf
# from collections import Counter
   
# from config import clickMeConfig, InceptionConfig
# import baseline_vgg16 as vgg16
# from tqdm import tqdm
# import time
# import tensorflow_helpers as tfhf
# import pickledb


# saved_weights_path = "/media/data_cifs/clicktionary/clickme_experiment/attngrad_inception_checkpoints/attngrad_0.0001_144023_2017_07_28_03_06_10/model_268000.ckpt-268000"
# # saved_weights_path = None
# config = clickMeConfig()
# incfg = InceptionConfig()

# image_width       = 224
# image_height      = 224
# batch_size        = 100

# batch_shape = [batch_size, image_height, image_width, 3]

# tf.reset_default_graph()
# data_X, image_dict, file_names, images_meta = hf.get_adversarial_data()
# image_df = pd.read_csv("/media/data_cifs/danshiebler/data/adversarial/images.csv")
# file_names =  images_meta["ImageId"].values
		
# inception_kwargs = {
#         'dropout_keep_prob': incfg.dropout_keep_prob,
#     }
# with tf.device("/gpu:0"):

# 	x = tf.placeholder(tf.float32, (None, 224, 224, 3), name="x") #the input variable

# 	xproc = inception_input_processing(x)
# 	with tf.contrib.slim.arg_scope(inception.inception_v3_arg_scope()):
# 		predictions, _ = inception.inception_v3(xproc, is_training=False, **inception_kwargs)

# 	# Load the saved weights and initialize the tensorflow graph
# 	sess = tfhf.initialize_session_inception(saved_weights_path)

# 	file_to_prob = {}
# 	for i in tqdm(range(0, data_X.shape[0], batch_size)):
# 		batch_data, batch_names = data_X[i:i+batch_size], file_names[i:i+batch_size]
# 		probs = sess.run(predictions, feed_dict={x:batch_data})
# 		for j in range(len(batch_names)):
# 			file_to_prob[batch_names[j]] = probs[j]

# 	for i, f in enumerate(file_to_prob):
# 		print file_to_prob[f].argsort()[-10:][::-1] + 1
# 		print image_df["TrueLabel"].values[i] 
# 		print




