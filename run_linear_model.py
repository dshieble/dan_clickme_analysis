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


def run_model(signature, clf):
	features_db = pickledb.load('databases/signature_to_saved_data_path.db', True)
	model_results_db = pickledb.load('databases/signature_to_linear_model_results.db', True)


	print "prediction for {}".format(features_db.get(signature)["saved_data_path"])

	features_db = pickledb.load('databases/signature_to_saved_data_path.db', True)

	img_ids, img_ID_to_LABELS = hf.get_ids_labels()
	# img_ids = [ID for ID in img_ids if not img_ID_to_LABELS[ID] in [39, 45]]
	labels = [img_ID_to_LABELS[ID] for ID in img_ids]

	print "loading..."
	file_to_features = np.load("data/{}.npy".format(signature)).item()
	print "loaded!"

	X = np.vstack([file_to_features[ID] for ID in img_ids])
	y = MultiLabelBinarizer().fit_transform([img_ID_to_LABELS[ID] for ID in img_ids])

	indices = np.random.permutation(np.arange(X.shape[0]))
	train, test = indices[:int(0.75*len(indices))], indices[int(0.75*len(indices)):]
	Xtrain, ytrain, Xtest, ytest = X[train], y[train],  X[test], y[test]

	scalar = StandardScaler()
	Xtrain_scaled = scalar.fit_transform(Xtrain)
	Xtest_scaled = scalar.transform(Xtest)
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
	print "accuracy", accuracy
	print "recall", recall
	print "precision", precision
	print 
	data = {"accuracy":accuracy, "recall":recall, "precision":precision, "predictions":predictions.tolist(), 
			"probs":probs.tolist(), "y":ytest.tolist(), "report": report}
	hf.save_to_db(model_results_db, signature, data)


if __name__ == "__main__":
	# clf = SGDClassifier(warm_start=True, n_jobs=-1)#LogisticRegression(solver='sag', multi_class="multinomial", verbose=1000, n_jobs=-1)
	# signature = "1500179298"
	# batch_size = 50000
	# num_epochs = 10
	# run_model(signature, clf, batch_size, num_epochs)


	for signature in ["1500301581", "1500301444", "1500301269"]:
		clf = OneVsRestClassifier(SGDClassifier(loss="log", warm_start=True, n_jobs=-1))#LogisticRegression(solver='sag', multi_class="multinomial", verbose=1000, n_jobs=-1)
		# clf = MLPClassifier(hidden_layer_sizes=(),
	 #                                           activation='logistic', solver='adam', 
	 #                                           alpha=0.0001, batch_size='auto', 
	 #                                           learning_rate='constant', 
	 #                                           learning_rate_init=0.001, power_t=0.5, 
	 #                                           max_iter=200, shuffle=True, random_state=None, 
	 #                                           tol=0.001, verbose=True, warm_start=False, 
	 #                                           momentum=0.9, nesterovs_momentum=True,
	 #                                           early_stopping=True, validation_fraction=0.1, 
	 #                                           beta_1=0.9, beta_2=0.999, epsilon=1e-08)


		run_model(signature, clf)

