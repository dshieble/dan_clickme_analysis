import sys
sys.path.append("clickme_modeling")
sys.path.append("clickme_modeling/models")
sys.path.append("/media/data_cifs/image_datasets/coco_2014/PythonAPI")

from pycocotools.coco import COCO
train_coco=COCO("/media/data_cifs/image_datasets/coco_2014/coco_images/annotations/instances_train2014.json")
val_coco=COCO("/media/data_cifs/image_datasets/coco_2014/coco_images/annotations/instances_val2014.json")

import h5py
import pandas as pd
import numpy as np
import helper_functions as hf
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from collections import Counter
from tqdm import tqdm
from scipy.misc import imread, imresize
import urllib2


path = "/media/data_cifs/image_datasets/coco_2014/coco_images"
ilsvrc_coco_overlap_categories = np.load("{}/ilsvrc_coco_overlap_categories.npy".format(path))
overlaps = set([int(c[2]) for c in ilsvrc_coco_overlap_categories])

for kind, coco in [('train', train_coco), ('val', val_coco)]:

    img_to_labels = {}
    overlap_names = set([k[0] for k in ilsvrc_coco_overlap_categories])
    overlap_ids = []
    for img in tqdm(coco.imgs.keys()):
        ann_ids = coco.getAnnIds(imgIds=[img])
        annotations = coco.loadAnns(ann_ids)
        annotation_ids = list(set([a['category_id'] for a in annotations]))
        img_to_labels[img] = annotation_ids
        cats = coco.loadCats(ids=annotation_ids)
        id_to_name = {c["id"]:c["name"] for c in cats}
        f_annotation_id_names = [(ID, id_to_name[ID]) for ID in annotation_ids if id_to_name[ID] in overlap_names]
        if len(f_annotation_id_names):
        	overlap_ids.append(img)

    print len(coco.imgs.keys()), len(img_to_labels), len(overlap_ids)

    np.save("{}/{}_img_IDs.npy".format(hf.data_path, kind), coco.imgs.keys())
    np.save("{}/{}_overlap_img_IDs.npy".format(hf.data_path, kind), overlap_ids)
    np.save("{}/{}_img_ID_to_LABELS.npy".format(hf.data_path, kind), img_to_labels)





for kind, coco in [('val', val_coco), ('train', train_coco)]:
    IDS = np.load("{}/{}_img_IDs.npy".format(hf.data_path, kind))
    img_objs = coco.loadImgs(ids=list(IDS))

    try:
        ID_to_img = np.load("{}/{}_img_ID_to_DATA.npy".format(hf.data_path, kind)).item()
        img_objs = [img_obj for img_obj in tqdm(img_objs) if not img_obj["id"] in ID_to_img.keys()]
    except Exception as e:
        print e
        ID_to_img = {}
    for i, img_obj in tqdm(zip(range(len(img_objs)), img_objs)):
        f = urllib2.urlopen(img_obj['flickr_url'])
        img = imread(f)
        img = imresize(img, (224,224,3))
        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        assert len(img.shape) == 3
        assert img.shape[2] == 3
        ID_to_img[img_obj['id']] = img
        if i % 1000 == 0:
    		np.save("{}/{}_img_ID_to_DATA.npy".format(hf.data_path, kind), ID_to_img)
    np.save("{}/{}_img_ID_to_DATA.npy".format(hf.data_path, kind), ID_to_img)


# loaded_data = {f:imread("{}/{}".format(path, f))[None,...] for f in tqdm(data_files)}
# resized_data = {f:imresize(loaded_data[f][0], (224,224,3)) for f in loaded_data}



# The labels for the datafiles
# base_path = "/media/data_cifs/image_datasets/coco_2014/coco_images"
# path = "{}/ilsvrc12_val_overlap".format(base_path)
# data_files = hf.ls_function(path)



# loaded_data = {f:imread("{}/{}".format(path, f))[None,...] for f in tqdm(data_files)}
# resized_data = {f:imresize(loaded_data[f][0], (224,224,3)) for f in loaded_data}


# np.save("data/data_X.npy", resized_data)

# data_X = np.vstack(resized_data)
# print data_X.shape