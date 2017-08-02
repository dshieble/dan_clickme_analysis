# import tensorflow
import h5py
import pandas as pd
import numpy as np
import helper_functions as hf
import matplotlib.pyplot as plt
from scipy.misc import imread
from collections import Counter
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
import pickledb
import sys

if __name__ == "__main__":
	n = int(sys.argv[1]) if (len(sys.argv) > 1) else 5
	prev = int(sys.argv[2]) if (len(sys.argv) > 2) else 3

	db = pickledb.load('databases/generated_adversarial_images.db', True)

	perf_list = [p.split(".")[0] for p in hf.ls_function("/media/data_cifs/danshiebler/data/adversarial/adversarial_performances")]

	for signature in [str(s) for s in sorted([int(p) for p in perf_list])[-prev:]]:
		data_obj = db.get(signature)

		categories = pd.read_csv("/media/data_cifs/danshiebler/data/adversarial/categories.csv")
		categories.index = categories["CategoryId"]
		images_meta = pd.read_csv("/media/data_cifs/danshiebler/data/adversarial/images.csv")
		file_names =  images_meta["ImageId"].values

		image_dict = np.load("/media/data_cifs/danshiebler/data/adversarial/image_dict.npy").item()
		data_X = np.vstack([image_dict[fname][None,...] for fname in file_names])

		performance = np.load("/media/data_cifs/danshiebler/data/adversarial/adversarial_performances/{}.npy".format(signature)).item()

		correct = {"normal":[], "adv":[]}

		print
		print "**********"
		print
		print data_obj
		for k in correct:
			for i, (_, row) in enumerate(images_meta.iterrows()):
				inds = performance[k][row["ImageId"]].argsort()[-n:][::-1] + 1 # Add 1 to the inds to make it match the categories
				correct[k].append(row["TrueLabel"] in inds)
				# for j, ind in enumerate(inds):
				# 	print j, categories["CategoryName"][ind]
			print "top {} accuracy for".format(n), k, float(sum(correct[k]))/len(correct[k])

