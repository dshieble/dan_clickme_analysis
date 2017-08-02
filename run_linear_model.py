import h5py
import pandas as pd
import numpy as np
import helper_functions as hf
import matplotlib.pyplot as plt
from scipy.misc import imread
from collections import Counter
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickledb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow_linear_model import TFLinearModel
import sys

def run_model(signature, clf, img_ID_to_DATA):
	features_db = pickledb.load('databases/fc7_features_signature.db', True)
	model_results_db = pickledb.load('databases/linear_model_results.db', True)
	saved_weights_path = features_db.get(signature)["saved_weights_path"]

	print "prediction for {}".format(saved_weights_path)

	features_db = pickledb.load('databases/signature_to_saved_data_path.db', True)


	print "loading..."
	file_to_features = np.load("{}/generated_feature_vectors/{}.npy".format(hf.data_path, signature)).item()
	train_keys, val_keys = img_ID_to_DATA["train_keys"], img_ID_to_DATA["val_keys"]

	Xtrain = np.vstack([file_to_features["train"][ID] for ID in tqdm(train_keys)])
	ytrain = img_ID_to_DATA["train_labels"][:]

	Xtest = np.vstack([file_to_features["val"][ID] for ID in tqdm(val_keys)])
	ytest = img_ID_to_DATA["val_labels"][:]

	scalar = StandardScaler()
	Xtrain_scaled = scalar.fit_transform(Xtrain)
	Xtest_scaled = scalar.transform(Xtest)
	print "loaded!"

	print Xtrain_scaled.shape, ytrain.shape, Xtest_scaled.shape, ytest.shape


	start = time.time()
	print "training..."
	clf.fit(Xtrain_scaled, ytrain)
	print "trained in {}!".format(time.time() - start)

	predictions = clf.predict(Xtest_scaled)
	probs = clf.predict_proba(Xtest_scaled)

	accuracy = accuracy_score(ytest, predictions)
	recall = recall_score(ytest, predictions, average="samples")
	precision = precision_score(ytest, predictions, average="samples")

	report = classification_report(ytest, predictions)
	print  "\n\n\n"
	print "prediction for {}".format(saved_weights_path)
	print "accuracy", accuracy
	print "recall", recall
	print "precision", precision
	print  "\n\n\n"
	data = {"accuracy":accuracy, "recall":recall, "precision":precision, "predictions":predictions.tolist(), 
			"probs":probs.tolist(), 
			"y":ytest.tolist(), "report": report}
	hf.save_to_db(model_results_db, signature, data)


if __name__ == "__main__":
 	prev = int(sys.argv[1])


	img_ID_to_DATA = h5py.File("{}/img_ID_to_DATA.h5py".format(hf.data_path))
	perf_list = [p.split(".")[0] for p in hf.ls_function("/media/data_cifs/danshiebler/data/generated_feature_vectors")]
	for signature in [str(s) for s in sorted([int(p) for p in perf_list])[-prev:]]:
		clf = TFLinearModel()
		run_model(signature, clf, img_ID_to_DATA)






	# clf = SGDClassifier(warm_start=True, n_jobs=-1)#LogisticRegression(solver='sag', multi_class="multinomial", verbose=1000, n_jobs=-1)
	# signature = "1500179298"
	# batch_size = 50000
	# num_epochs = 10
	# run_model(signature, clf, batch_size, num_epochs)
	# clf = OneVsRestClassifier(SGDClassifier(loss="hinge", warm_start=True, n_jobs=-1, verbose=1000))#LogisticRegression(solver='sag', multi_class="multinomial", verbose=1000, n_jobs=-1)
	# clf = OneVsRestClassifier(SGDClassifier(loss="log", warm_start=True, n_jobs=-1, verbose=1000))#LogisticRegression(solver='sag', multi_class="multinomial", verbose=1000, n_jobs=-1)
	# clf = MLPClassifier(hidden_layer_sizes=(),
	#                                           activation='logistic', solver='adam', 
	#                                           alpha=0.0001, batch_size='auto', 
	#                                           learning_rate='constant', 
	#                                           learning_rate_init=0.001, power_t=0.5, 
	#                                           max_iter=200, shuffle=True, random_state=None, 
	#                                           tol=0.00001, verbose=True, warm_start=False, 
	#                                           momentum=0.9, nesterovs_momentum=True,
	#                                           early_stopping=True, validation_fraction=0.1, 
	#                                           beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	# clf = TFLinearModel()
	# clf.fit(np.random.random((10000,4096)), np.random.randint(0,2, (10000, 80)))
	# probs = clf.predict_proba(np.random.random((1000,4096)))
	# print probs.shape
	# probs = clf.predict(np.random.random((1000,4096)))
	# print probs


