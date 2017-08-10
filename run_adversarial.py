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

saved_weights_paths = [
# "/media/cifs_all/charlie/clickme/baseline_inception_checkpoints/inception_v3.ckpt",
# "/media/data_cifs/clicktionary/clickme_experiment/attngrad_inception_checkpoints/attngrad_0.0001_144023_2017_07_28_03_06_10/model_268000.ckpt-268000",
# "/media/data_cifs/clicktionary/clickme_experiment/attngrad_inception_checkpoints/attngrad_0.0001_144023_2017_07_27_21_21_30/model_284000.ckpt-284000",
# "/media/data_cifs/clicktionary/clickme_experiment/attngrad_inception_checkpoints/attngrad_5e-05_144023_2017_07_27_21_21_27/model_284000.ckpt-284000",
# "/media/data_cifs/clicktionary/clickme_experiment/attngrad_inception_checkpoints/attngrad_5e-05_144023_2017_07_27_21_21_28/model_284000.ckpt-284000",
# "/media/data_cifs/clicktionary/clickme_experiment/attngrad_inception_checkpoints/attngrad_5e-05_144023_2017_07_27_21_21_31/model_284000.ckpt-284000",
# "/media/data_cifs/clicktionary/clickme_experiment/attngrad_inception_checkpoints/attngrad_5e-05_144023_2017_07_28_03_06_12/model_268000.ckpt-268000",
# "/media/data_cifs/clicktionary/clickme_experiment/attngrad_inception_checkpoints/attngrad_0.001_144023_2017_07_28_03_06_14/model_268000.ckpt-268000",
# "/media/data_cifs/clicktionary/clickme_experiment/attngrad_inception_checkpoints/attngrad_0.001_144023_2017_07_27_21_21_32/model_284000.ckpt-284000"
# "/media/data_cifs/clicktionary/clickme_experiment/baseline_inception_checkpoints/baseline_0.001_144536_2017_08_03_14_45_26/model_4000.ckpt-4000",
# "/media/data_cifs/clicktionary/clickme_experiment/baseline_inception_checkpoints/baseline_0.001_144536_2017_08_03_14_45_28/model_4000.ckpt-4000",
# "/media/data_cifs/clicktionary/clickme_experiment/baseline_inception_checkpoints/baseline_0.001_144536_2017_08_03_14_45_31/model_4000.ckpt-4000",
# "/media/data_cifs/clicktionary/clickme_experiment/baseline_inception_checkpoints/baseline_0.001_144536_2017_08_03_14_45_19/model_4000.ckpt-4000",
# "/media/data_cifs/clicktionary/clickme_experiment/baseline_inception_checkpoints/baseline_0.001_144536_2017_08_03_14_45_29/model_4000.ckpt-4000"
# "/media/data_cifs/clicktionary/clickme_experiment/attgrad_vgg_checkpoints/gradient_-05_144023_2017_07_27_03_55_12/model_244000.ckpt-244000",
# "/media/data_cifs/clicktionary/clickme_experiment/attgrad_vgg_checkpoints/gradient_-05_144023_2017_07_27_03_55_03/model_244000.ckpt-244000",
# "/media/data_cifs/clicktionary/clickme_experiment/attgrad_vgg_checkpoints/gradient_0001_144023_2017_07_27_03_55_01/model_240000.ckpt-240000",
# "/media/data_cifs/clicktionary/clickme_experiment/attgrad_vgg_checkpoints/gradient_001_144023_2017_07_27_03_55_08/model_126000.ckpt-126000"
"/media/data_cifs/clicktionary/clickme_experiment/baseline_inception_checkpoints/baseline_0.0001_145928_2017_08_08_01_47_20/model_26000.ckpt-26000",
"/media/data_cifs/clicktionary/clickme_experiment/baseline_inception_checkpoints/baseline_0.0001_145928_2017_08_08_01_47_24/model_26000.ckpt-26000",
"/media/data_cifs/clicktionary/clickme_experiment/baseline_inception_checkpoints/baseline_0.0001_145928_2017_08_08_01_47_29/model_26000.ckpt-26000",
"/media/data_cifs/clicktionary/clickme_experiment/baseline_inception_checkpoints/baseline_0.0001_145928_2017_08_08_01_47_18/model_26000.ckpt-26000",
"/media/data_cifs/clicktionary/clickme_experiment/baseline_inception_checkpoints/baseline_0.0001_145928_2017_08_08_01_47_30/model_26000.ckpt-26000",
"/media/data_cifs/clicktionary/clickme_experiment/baseline_inception_checkpoints/baseline_0.0001_145928_2017_08_08_01_47_32/model_26000.ckpt-26000",
"/media/data_cifs/clicktionary/clickme_experiment/baseline_inception_checkpoints/baseline_0.001_145928_2017_08_08_01_47_22/model_26000.ckpt-26000",
"/media/data_cifs/clicktionary/clickme_experiment/baseline_inception_checkpoints/baseline_0.001_145928_2017_08_08_01_47_26/model_26000.ckpt-26000"
]

model_kinds = ["inception"]*8 #+ ["vgg"]*4


for swp, mk in zip(saved_weights_paths, model_kinds):
	signature = generate_adversarial_images(swp, mk)
	test_adversarial_performance(signature)

# "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_112341_2017_05_15_22_53_23/model_80000.ckpt-80000",
# "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_112341_2017_05_15_22_53_23/model_56000.ckpt-56000",
# "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_112341_2017_05_15_22_53_23/model_60000.ckpt-60000",
# "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/baseline_001_50000_2017_06_07_10_19_47/model_252000.ckpt-252000",
# "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_112341_2017_05_15_22_53_23/model_174000.ckpt-174000"
# "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_112341_2017_05_15_22_53_23/model_40000.ckpt-40000",
# "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_112341_2017_05_15_22_53_23/model_80000.ckpt-80000",
# "/media/data_cifs/clicktionary/clickme_experiment/attention_gradient_checkpoints/gradient_001_130671_2017_07_15_15_01_12/model_44000.ckpt-44000",
# "/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_124720_2017_06_07_10_19_49/model_162000.ckpt-162000"
	