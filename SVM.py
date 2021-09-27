from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn import preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from metrics import Metrics



ids = [2]
plts = ["Best", "Worst"]
# iters = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
# 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
# 3000, 3100, 3200, 3300, 3400]
iters = 100*np.linspace(200,1000,100)

for i in ids:

	data_set_id = i

	if data_set_id == 1:

		# import data
		wine_data = pd.read_csv('winequality-red.csv', sep = ';', header = None)
		wine_df = pd.DataFrame(wine_data)
		wine_data = np.array(wine_data)
		wine_data = np.array(wine_data[1:][:],dtype='float')
		y_wine = np.array(wine_data[:,-1], dtype='int')
		x_wine = wine_data[:, :-1]

		x = x_wine
		y = y_wine

		# param_grid = {
		# 	'kernel': ['rbf', 'linear', 'sigmoid', 'poly'],
		# 	'C': [0.1, 0.5, 1, 2, 4],
		# 	'gamma': ['scale', 'auto']
		# 	}
		# name = "SVCMetrics_Wine_2"
		# met = Metrics()
		# model = SVC()
		# met.cross_validation_scores(model, param_grid, x, y, name)
		# model = SVC(kernel='linear', C=1, verbose=2)
		# model.fit(x, y)
		# print(sdfd)


		for j in range(len(plts)):

			train_scores_plt = []
			test_scores_plt = []
			# if j == 0:


				# for itr in iters:
				# 	skf = StratifiedKFold(n_splits=5)
				# 	model = SVC(kernel='linear', C=1, max_iter=itr)
				# 	print(np.unique(y))
				# 	scores  = cross_validate(model, x, y, cv=skf, scoring='accuracy', return_train_score=True)
				# 	print("Scores: ", scores)
				# 	train_scores = scores['train_score']
				# 	test_scores = scores['test_score']
				# 	train_scores_avg = np.mean(train_scores)
				# 	test_scores_avg = np.mean(test_scores)
				# 	train_scores_plt.append(train_scores_avg)
				# 	test_scores_plt.append(test_scores_avg)
				# name = "SVC_Wine_" + plts[j]
				# title = "SVC (Wine) - Best"
				# Metrics().plot_learning_curve_itr(name, title, iters, train_scores_plt, test_scores_plt)

				# 	# met = Metrics()
				# 	# met.learning_curve_data(model, x, y)
					# met.plot_learning_curve("SVM_LearningCurve", "SVM (Wine Data)")


			if j == 1:

				for itr in iters:
					skf = StratifiedKFold(n_splits=5)
					model = SVC(kernel='sigmoid', C=1, max_iter=itr)
					print(np.unique(y))
					scores  = cross_validate(model, x, y, cv=skf, scoring='accuracy', return_train_score=True)
					print("Scores: ", scores)
					train_scores = scores['train_score']
					test_scores = scores['test_score']
					train_scores_avg = np.mean(train_scores)
					test_scores_avg = np.mean(test_scores)
					train_scores_plt.append(train_scores_avg)
					test_scores_plt.append(test_scores_avg)
				name = "SVC_Wine_" + plts[j]
				title = "SVC (Wine) - Worst"
				Metrics().plot_learning_curve_itr(name, title, iters, train_scores_plt, test_scores_plt)

					# met = Metrics()
					# met.learning_curve_data(model, x, y)
					# met.plot_learning_curve("SVM_LearningCurve", "SVM (Wine Data)")

	if data_set_id == 2:

		# import data
		default_data = pd.read_csv('default.csv', sep = ',', header = None)
		default_df = pd.DataFrame(default_data)
		default_data = np.array(default_data)
		default_data = np.array(default_data[:][:])
		y_default = default_data[2:, -1]
		x_default = default_data[2:, 1:-1]

		print(y_default)
		print(x_default)

		x = np.array(x_default, dtype='int')
		y = np.array(y_default, dtype='int')

		# param_grid = {
		# 	'kernel': ['rbf', 'linear', 'sigmoid', 'poly'],
		# 	'C': [1, 2, 4]
		# 	}
		# name = "SVCMetrics_Default"
		# met = Metrics()
		# model = SVC()
		# met.cross_validation_scores(model, param_grid, x, y, name)
		# # model = SVC(kernel='linear', C=1, verbose=2)
		# # model.fit(x, y)
		# # print(sdfd)
		# print(skfhg)

		iters = 100*np.linspace(200,1000,5)


		for j in range(len(plts)):

			train_scores_plt = []
			test_scores_plt = []
			if j == 0:


				for itr in iters:
					skf = StratifiedKFold(n_splits=5)
					model = SVC(kernel='linear', C=1, max_iter=itr)
					print(np.unique(y))
					scores  = cross_validate(model, x, y, cv=skf, scoring='accuracy', return_train_score=True)
					print("Scores: ", scores)
					train_scores = scores['train_score']
					test_scores = scores['test_score']
					train_scores_avg = np.mean(train_scores)
					test_scores_avg = np.mean(test_scores)
					train_scores_plt.append(train_scores_avg)
					test_scores_plt.append(test_scores_avg)
				name = "SVC_Default_" + plts[j]
				title = "SVC (Default) - Best"
				Metrics().plot_learning_curve_itr(name, title, iters, train_scores_plt, test_scores_plt)

					# met = Metrics()
					# met.learning_curve_data(model, x, y)
					# met.plot_learning_curve("SVM_LearningCurve", "SVM (Wine Data)")



				for itr in iters:
					skf = StratifiedKFold(n_splits=5)
					model = SVC(kernel='sigmoid', C=1, max_iter=itr)
					print(np.unique(y))
					scores  = cross_validate(model, x, y, cv=skf, scoring='accuracy', return_train_score=True)
					print("Scores: ", scores)
					train_scores = scores['train_score']
					test_scores = scores['test_score']
					train_scores_avg = np.mean(train_scores)
					test_scores_avg = np.mean(test_scores)
					train_scores_plt.append(train_scores_avg)
					test_scores_plt.append(test_scores_avg)
				name = "SVC_Default_" + plts[j]
				title = "SVC (Default) - worst"
				Metrics().plot_learning_curve_itr(name, title, iters, train_scores_plt, test_scores_plt)





			# if j == 1:

			# 	model = SVC(kernel='linear', C=2)
			# 	met = Metrics()
			# 	met.learning_curve_data(model, x, y)
			# 	met.cross_validation_scores(model, param_grid, x, y, name)




# # split data between training and testing
# sss = StratifiedShuffleSplit(n_splits=1 , test_size=0.2)
# for train_idx, test_idx in sss.split(x,y):
# 	x_train, x_test = x[train_idx], x[test_idx]
# 	y_train, y_test = y[train_idx], y[test_idx]
# 	print(len(x_train))
# 	print(len(x_test))




# param_grid = {
# 	'kernel': ['rbf', 'linear'],
# 	'C': [1, 2, 4]
# }
# name = "SVCMetrics_Wine"
# met = Metrics()
# model = SVC()
# met.cross_validation_scores(model, param_grid, x, y, name)


# model = SVC(kernel='linear', C=2)

# met = Metrics()
# met.learning_curve_data(model, x, y)
# met.plot_learning_curve("SVM_LearningCurve", "SVM (Wine Data)")


# met.dsiplay_score(model, x_train, y_train, x_test, y_test)