# Import files

from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from metrics import Metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

ids = [1, 2]
plts = ["Best", "Worst"]

for i in ids:

	data_set_id = i

	if data_set_id == 1:



		wine_data = pd.read_csv('winequality-red.csv', sep = ';', header = None)
		wine_df = pd.DataFrame(wine_data)
		wine_data = np.array(wine_data)
		wine_data = np.array(wine_data[1:][:],dtype='float')
		y_wine = np.array(wine_data[:,-1], dtype='int')
		x_wine = wine_data[:, :-1]

		x = preprocessing.normalize(x_wine)
		y = y_wine

		# param_grid = {
		# 	'min_samples_leaf': [2, 10, 25, 50],
		# 	'max_depth': [None, 2, 3, 9, 20],
		# 	'ccp_alpha': [0, 0.001, 0.003, 0.005]
		# }
		# name = "DTMetrics_Wine"
		# met = Metrics()
		# model = tree.DecisionTreeClassifier()
		# met.cross_validation_scores(model, param_grid, x, y, name)


		for j in range(len(plts)):

			if j == 0:
				model = tree.DecisionTreeClassifier(min_samples_leaf=2, max_depth=3, ccp_alpha=0.003)

				met = Metrics()
				met.learning_curve_data(model, x, y)
				met.plot_learning_curve("DT_Wine_" + plts[j],"Decision Trees (Wine)")

			if j == 1:
				model = tree.DecisionTreeClassifier(min_samples_leaf=2, max_depth=None, ccp_alpha=0.0)

				met = Metrics()
				met.learning_curve_data(model, x, y)
				met.plot_learning_curve("DT_Wine_" + plts[j],"Decision Trees (Wine)")




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


		# param_grid = {
		# 	'min_samples_leaf': [2, 10, 25, 50],
		# 	'max_depth': [None, 3, 9, 20],
		# 	'ccp_alpha': [0, 0.001, 0.003, 0.005]
		# }
		# name = "DTMetrics_Default"
		# met = Metrics()
		# model = tree.DecisionTreeClassifier()
		# met.cross_validation_scores(model, param_grid, x, y, name)


		for j in range(len(plts)):

			if j == 0:
				model = tree.DecisionTreeClassifier(min_samples_leaf=2, max_depth=3, ccp_alpha=0.0)

				met = Metrics()
				met.learning_curve_data(model, x, y)
				met.plot_learning_curve("DT_Default_" + plts[j],"Decision Trees (Default)")

			if j == 1:
				model = tree.DecisionTreeClassifier(min_samples_leaf=2, max_depth=None, ccp_alpha=0.0)

				met = Metrics()
				met.learning_curve_data(model, x, y)
				met.plot_learning_curve("DT_Default_" + plts[j],"Decision Trees (Default)")



		# model = tree.DecisionTreeClassifier(min_samples_leaf=leafsize, ccp_alpha=0.001)
		# model = tree.DecisionTreeClassifier(min_samples_leaf=10)

		# met = Metrics()
		# met.learning_curve_data(model, x, y)
		# met.plot_learning_curve("DT_Default","Decision Trees (Default)")






