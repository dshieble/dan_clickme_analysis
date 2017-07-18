import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
