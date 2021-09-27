from sklearn import datasets
# from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import learning_curve
from sklearn import preprocessing
from metrics import Metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv


ids = [2]
plts = ["Best", "Worst"]

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

		x = preprocessing.normalize(x_wine)
		y = y_wine



		# print("x.shape[0]", x.shape[0])
		# leafsize = int(0.01*x.shape[0])
		# model = GradientBoostingClassifier(n_estimators=100, max_depth=5, min_samples_leaf=leafsize)
		# lcurve = learning_curve(model, x, y, scoring='accuracy', n_jobs=1, train_sizes=np.linspace(0.01, 1, 50), verbose=1)
		# train_sizes, train_scores, test_scores = lcurve

		# # print(train_scores)
		# train_means = np.mean(train_scores, axis=1)
		# test_means = np.mean(test_scores, axis=1)

		# plt.plot(train_sizes, train_means)
		# plt.plot(train_sizes, test_means)
		# plt.show()


		# run CV Search Grid
		# dt = tree.DecisionTreeClassifier()
		# model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
		# param_grid = {
		# 	'base_estimator__max_depth':[2, 4, 6],
		# 	'base_estimator__min_samples_leaf':[5,10],
		# 	'n_estimators':[10,50,100],
		# 	'learning_rate':[0.01,0.1]}
		# name = "AdaBoostMetrics_Wine"
		# met = Metrics()
		# # model = tree.DecisionTreeClassifier()
		# met.cross_validation_scores(model, param_grid, x, y, name)

		# plot learning curves
		for j in range(len(plts)):

			if j == 0:
				
				dt = DecisionTreeClassifier(min_samples_leaf=10, max_depth=6)
				model = AdaBoostClassifier(n_estimators=100, learning_rate=0.01, base_estimator=dt)

				met = Metrics()
				met.learning_curve_data(model, x, y)
				met.plot_learning_curve("AdaBoost_LearningCurve_Wine_" + plts[j], "AdaBoost (Wine Data)")

			if j == 1:
				dt = DecisionTreeClassifier(min_samples_leaf=10, max_depth=6)
				model = AdaBoostClassifier(n_estimators=10, learning_rate=0.01, base_estimator=dt)

				met = Metrics()
				met.learning_curve_data(model, x, y)
				met.plot_learning_curve("AdaBoost_LearningCurve_Wine_" + plts[j], "AdaBoost (Wine Data)")


	if data_set_id == 2:

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


		# model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
		# param_grid = {
		# 	'base_estimator__max_depth':[2, 4, 6],
		# 	'base_estimator__min_samples_leaf':[3, 5],
		# 	'n_estimators':[10,50,100],
		# 	'learning_rate':[0.01,0.1]}
		# name = "AdaBoostMetrics_Default"
		# met = Metrics()
		# # model = tree.DecisionTreeClassifier()
		# met.cross_validation_scores(model, param_grid, x, y, name)


		for j in range(len(plts)):

			if j == 0:
				
				dt = DecisionTreeClassifier(min_samples_leaf=5, max_depth=4)
				model = AdaBoostClassifier(n_estimators=100, learning_rate=0.01, base_estimator=dt)

				met = Metrics()
				met.learning_curve_data(model, x, y)
				met.plot_learning_curve("AdaBoost_LearningCurve_Default_" + plts[j], "AdaBoost (Default)")

			if j == 1:
				dt = DecisionTreeClassifier(min_samples_leaf=3, max_depth=6)
				model = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, base_estimator=dt)

				met = Metrics()
				met.learning_curve_data(model, x, y)
				met.plot_learning_curve("AdaBoost_LearningCurve_Default_" + plts[j], "AdaBoost (Default)")