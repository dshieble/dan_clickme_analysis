import os
from os import listdir
from scipy.misc import imread
import numpy as np
import time
from tqdm import tqdm
import json
data_path = "/media/data_cifs/danshiebler/data"
base_path = "/media/data_cifs/image_datasets/coco_2014/coco_images"
ilsvrc12_val_overlap_path = "{}/ilsvrc12_val_overlap".format(base_path)


def ls_function(mypath):
	"""
		List all of the files in a directory
	"""
	assert os.path.isdir(mypath)
	return [f for f in listdir(mypath) if not f == '.DS_Store']

def get_ids_labels(kind, ids_path):
	img_ids = np.load("data/{}".format(ids_path))
	img_ID_to_LABELS = np.load("data/{}_img_ID_to_LABELS.npy".format(kind)).item()
	return img_ids, img_ID_to_LABELS

def get_image_data(kind, ids_path):
	print "loading data..."

	# Load the data files and label mappings
	img_ID_to_DATA = np.load("data/{}_img_ID_to_DATA.npy".format(kind)).item()
	img_ids, img_ID_to_LABELS = get_ids_labels(kind, ids_path)
	assert len(img_ids) == len(img_ID_to_LABELS)

	# Filter for incomplete
	included = set(img_ID_to_DATA.keys())
	img_ids = [ID for ID in img_ids if ID in included]
	# for ID in img_ids:
	# 	assert ID in included

	# The picture data 
	data_X = np.vstack([np.array(img_ID_to_DATA[ID])[None,...] for ID in tqdm(img_ids)])
	labels = [img_ID_to_LABELS[ID] for ID in img_ids]

	assert data_X.shape[0] == len(img_ids)
	assert len(labels) == len(img_ids)
	print "data loaded!"
	return data_X, labels, img_ids


def save_to_db(db, key, value):
	json.dump(value, open("/tmp/test.json", "wb"))
	db.set(key, value)




# def get_files_labels(load_small=False):

# 	ilsvrc_coco_overlap_categories = np.load("{}/{}".format(base_path, "ilsvrc_coco_overlap_categories.npy"))
# 	cat_to_ind = {c[-1]:i for i,c in enumerate(ilsvrc_coco_overlap_categories)}

# 	# The files
# 	data_files = np.load("data/file_names_small.npy") if load_small else np.load("data/file_names.npy")
# 	# The labels for the picture data
# 	label_inds = np.array([cat_to_ind[f.split("_")[0]] for f in data_files])
# 	label_onehots = np.zeros((len(data_files), len(ilsvrc_coco_overlap_categories)))
# 	label_onehots[np.arange(len(data_files)), label_inds] = 1
# 	return label_inds, label_onehots, data_files

# def get_image_data(load_small=False):
# 	print "loading data..."
# 	# Load the data files and label mappings
# 	label_inds, label_onehots, data_files = get_files_labels(load_small=load_small)

# 	# The picture data 
# 	loaded_data = np.load("data/data_X_dict.npy").item()
# 	data_X = np.vstack([np.array(loaded_data[f])[None,...] for f in tqdm(data_files)])

# 	assert data_X.shape[0] == len(label_inds)
# 	assert len(label_inds) == len(label_onehots)
# 	assert len(label_onehots) == len(data_files)
# 	print "data loaded!"
# 	return data_X, label_inds, label_onehots, data_files

# def get_image_data(load_small=False):
# 	# Load the data and label mappings

# 	ilsvrc_coco_overlap_categories = np.load("{}/{}".format(base_path, "ilsvrc_coco_overlap_categories.npy"))
# 	cat_to_ind = {c[-1]:i for i,c in enumerate(ilsvrc_coco_overlap_categories)}

# 	# The picture data and files
# 	print "loading data..."
# 	if load_small:
# 		data_files = np.load("data/file_names_small.npy")
# 		data_X = np.load("data/data_X_small.npy")
# 	else:
# 		data_files = np.load("data/file_names.npy")
# 		data_X = np.load("data/data_X.npy")
# 	print "data loaded!"


# 	# The labels for the picture data
# 	label_inds = np.array([cat_to_ind[f.split("_")[0]] for f in data_files])
# 	label_onehots = np.zeros((len(data_files), len(ilsvrc_coco_overlap_categories)))
# 	label_onehots[np.arange(len(data_files)), label_inds] = 1
# 	assert data_X.shape[0] == len(label_inds)
# 	assert len(label_inds) == len(label_onehots)
# 	assert len(label_onehots) == len(data_files)
# 	return data_X, label_inds, label_onehots, data_files

