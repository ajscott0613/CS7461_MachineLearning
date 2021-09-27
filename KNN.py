from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from metrics import Metrics


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

		# x = preprocessing.normalize(x_wine)
		x = x_wine
		y = y_wine


		# param_grid = {
		# 	'n_neighbors': [2, 5, 10, 20, 50],
		# 	'weights': ['uniform', 'distance']
		# }
		# name = "KNNMetrics_Wine"
		# met = Metrics()
		# model = KNeighborsClassifier()
		# met.cross_validation_scores(model, param_grid, x, y, name)

		for j in range(len(plts)):

			# plots learning curves
			if j == 0:
				neigh = KNeighborsClassifier(n_neighbors=50)
				met = Metrics(train_sizes=np.linspace(0.01, 1, 100))
				met.learning_curve_data(neigh, x, y)
				met.plot_learning_curve("KNN_Wine_" + plts[j], "KNN (Wine)")

			if j == 1:
				neigh = KNeighborsClassifier(n_neighbors=5)
				met = Metrics(train_sizes=np.linspace(0.01, 1, 100))
				met.learning_curve_data(neigh, x, y)
				met.plot_learning_curve("KNN_Wine_" + plts[j], "KNN (Wine)")



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
		# 	'n_neighbors': [2, 5, 10, 20, 50, 100, 150, 200],
		# 	'weights': ['uniform', 'distance']
		# }
		# name = "KNNMetrics_Default"
		# met = Metrics()
		# model = KNeighborsClassifier()
		# met.cross_validation_scores(model, param_grid, x, y, name)

		for j in range(len(plts)):

			# plot learning curve
			if j == 0:
				neigh = KNeighborsClassifier(n_neighbors=100)
				met = Metrics(train_sizes=np.linspace(0.01, 1, 100))
				met.learning_curve_data(neigh, x, y)
				met.plot_learning_curve("KNN_Default_" + plts[j], "KNN (Default)")

			if j == 1:
				neigh = KNeighborsClassifier(n_neighbors=2)
				met = Metrics(train_sizes=np.linspace(0.01, 1, 100))
				met.learning_curve_data(neigh, x, y)
				met.plot_learning_curve("KNN_Default_" + plts[j], "KNN (Default)")