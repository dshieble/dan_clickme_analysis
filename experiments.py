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

db = pickledb.load('databases/generated_adversarial_images.db', True)

perf_list = hf.ls_function("/media/data_cifs/danshiebler/data/adversarial/adversarial_performances")
for signature in perf_list[-2:]:
	data_obj = db.get(signature)
	print data_obj

	categories = pd.read_csv("/media/data_cifs/danshiebler/data/adversarial/categories.csv")
	categories.index = categories["CategoryId"]
	images_meta = pd.read_csv("/media/data_cifs/danshiebler/data/adversarial/images.csv")
	file_names =  images_meta["ImageId"].values

	image_dict = np.load("/media/data_cifs/danshiebler/data/adversarial/image_dict.npy").item()
	data_X = np.vstack([image_dict[fname][None,...] for fname in file_names])

	performance = np.load("/media/data_cifs/danshiebler/data/adversarial/adversarial_performances/{}.npy".format(signature)).item()

	correct = {"normal":[], "adv":[]}

	for k in correct:
		for i, (_, row) in enumerate(images_meta.iterrows()):
			inds = performance[k][row["ImageId"]].argsort()[-5:][::-1] + 1 # Add 1 to the inds to make it match the categories
			correct[k].append(row["TrueLabel"] in inds)
			# for j, ind in enumerate(inds):
			# 	print j, categories["CategoryName"][ind]
		print k, float(sum(correct[k]))/len(correct[k])
	print
	print "**********"
	print


# data_files = np.load("data/file_names.npy")

# start = time.time()
# data_X_dict = np.load("data/data_X_dict.npy").item()
# print time.time() - start

# start = time.time()
# data_X = np.load("data/data_X.npy")
# print start - time.time()


# start = time.time()
# loaded_data = np.vstack([np.array(data_X_dict[f])[None,...] for f in data_files])
# print start - time.time() - start


# data_X = np.vstack([np.array(loaded_data[f])[None,...] for f in data_files])

# label_inds, label_onehots, data_files = hf.get_files_labels(load_small=True)
# file_to_labels = dict(zip(data_files, label_inds))
# print "loading..."
# file_to_features = np.load("data/1500057550.npy").item()
# print "loaded!"

# X = np.vstack([v for v in file_to_features.values()])
# y = np.array([file_to_labels[k] for k in file_to_features.keys()])

# indices = np.random.permutation(np.arange(X.shape[0]))
# train, test = indices[:int(0.75*len(indices))], indices[int(0.75*len(indices)):]


# clf = LogisticRegression()
# print "training..."
# clf.fit(X[train], y[train])
# print "trained!"

# probs = clf.predict_proba(X[test])[:, 1]
# predictions = probs > 0.5
# print roc_auc_score(y, probs)
# print classification_report(y, predictions)
# data_X, label_inds, label_onehots, file_names = hf.get_image_data()
# np.save("data/data_X_small.npy", data_X[:3000])
# np.save("data/file_names_small.npy", file_names[:3000])

# data_X = np.load("data/data_X.npy")

# file_names = hf.ls_function(hf.ilsvrc12_val_overlap_path)
# data_X = np.save("data/data_X.npy")

# data_X = np.load("data/data_X.npy")
# dataX_and_filenames = {"data_X":data_X, "file_names":file_names}
# np.save("data/dataX_and_filenames.npy", dataX_and_filenames)

# np.save("data/file_names.npy", file_names)


# data_files = hf.ls_function(hf.ilsvrc12_val_overlap_path)

# loaded_data = np.load("data/data_X_old.npy").item()
# # loaded = [np.array(loaded_data[f])  for f in tqdm(data_files)]
# # shaped = [l for l in loaded if len(l.shape) == 2]

# # plt.figure()
# # for i in range(16):
# # 	plt.subplot(4,4,i + 1)
# # 	plt.imshow(shaped[i])
# # plt.show()

# for f in tqdm(loaded_data):
# 	l = np.array(loaded_data[f])
# 	if len(l.shape) == 2:
# 		loaded_data[f] = np.dstack((l,l,l))

# np.save("data/data_X.npy", loaded_data)


# file_to_labels = dict(zip(file_names, label_inds))
# preds = np.load("data/1500048151.npy")
# labels = np.array([file_to_labels[k] for k in preds.keys()])
# X = np.vstack(preds.values())

# print X.shape, y.shape


# base_path = "/media/data_cifs/image_datasets/coco_2014/coco_images"
# path = "{}/ilsvrc12_val_overlap".format(base_path)
# files = hf.ls_function(path)

# ilsvrc_coco_overlap_categories = np.load("{}/{}".format(base_path, "ilsvrc_coco_overlap_categories.npy"))




# coco_full_im_processed_labels = np.load("{}/{}".format(base_path, "coco_full_im_processed_labels.npz"))
# for k in coco_full_im_processed_labels.keys():
# 	print len(coco_full_im_processed_labels[k])
# 	print coco_full_im_processed_labels[k][:100]
# 	print


# files_831 = [f for f in files if int(f.split("_")[0]) == 831]
# files_859 = [f for f in files if int(f.split("_")[0]) == 859]
# plt.figure()
# for i in range(16):
# 	plt.subplot(4,4,i + 1)
# 	plt.imshow(plt.imread("{}/{}".format(path, files_831[i])))

# plt.figure()
# for i in range(16):
# 	plt.subplot(4,4,i + 1)
# 	plt.imshow(plt.imread("{}/{}".format(path, files_859[i])))
# plt.show()
# print files[:10]
# mat = imread("{}/{}".format(path, files[0]))
# plt.figure()
# plt.imshow(mat)
# plt.show()
# ilsvrc_coco_overlap_categories = np.load("/media/data_cifs/image_datasets/coco_2014/coco_images/ilsvrc_coco_overlap_categories.npy")
# df = pd.read_csv("/media/data_cifs/image_datasets/coco_2014/coco_images/ilsvrc_overlap.csv")

# print ilsvrc_coco_overlap_categories[:10]