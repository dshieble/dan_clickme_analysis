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
from adversarial import generate_adversarial_images
from test_adversarial import test_adversarial_performance

kind, ids_path = "val", "val_img_IDs.npy"

saved_weights_paths = [
"/media/data_cifs/clicktionary/clickme_experiment/attention_gradient_checkpoints/gradient_001_130671_2017_07_15_15_01_12/model_44000.ckpt-44000",
"/media/data_cifs/clicktionary/clickme_experiment/checkpoints/baseline_001_50000_2017_06_07_10_19_47/model_252000.ckpt-252000",
"/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_124720_2017_06_07_10_19_49/model_162000.ckpt-162000"]

for swp in saved_weights_paths:
	signature = generate_adversarial_images(swp, kind, ids_path)
	test_adversarial_performance(signature)


	