# GENERATE ADVERSARIAL IMAGES FOR A GIVEN SAVED WEIGHTS MODEL
from ops.db import get_data, create_dir, get_ims

A = get_data()
print A["clicks"][:4]
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




