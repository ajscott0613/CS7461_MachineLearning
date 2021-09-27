# Import files

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from metrics import Metrics

# from tensorflow import keras
# from keras.optimizers import Adam
# from keras.layers import Dense
# import tensorflow as tf



# def build_dqn(alpha, actions_n, inputs_n, h1_n, h2_n, env):
#     model = keras.models.Sequential()
#     model.add(Dense(units=h1_n, input_dim=env.observation_space.shape[0], activation='relu'))
#     model.add(Dense(units=h2_n, activation='relu'))
#     model.add(Dense(units=env.action_space.n, activation='linear'))
#     model.compile(loss='mean_squared_error', optimizer=Adam(lr=alpha))
#     model.summary()
#     return model


# dqn.fit(X_vals, q_targets, batch_size=batch_size, verbose=0)



ids = [2]
plts = ["Best", "Worst"]
iters = np.linspace(5, 300, 30)

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
		# 	'alpha': [0.00001, 0.0001, 0.001, 0.01],
		# 	'hidden_layer_sizes': [(15,15), (50,50), (75,75),(125,125)]
		# 	}
		# param_grid = {
		# 	'alpha': [0.00001, 0.0001],
		# 	'hidden_layer_sizes': [(200,200),(300,300),(400,400)]
		# 	}
		# name = "ANNMetrics_Wine_2"
		# met = Metrics()
		# model = MLPClassifier(max_iter=10000, n_iter_no_change=100)
		# met.cross_validation_scores(model, param_grid, x, y, name)
		# model = MLPClassifier(alpha=0.00001, hidden_layer_sizes=(200,200), max_iter=10000)
		# model.fit(x,y)
		# print("iters: ", model.n_iter_)



		for j in range(len(plts)):

			train_scores_plt = []
			test_scores_plt = []
			if j == 0:


				for itr in iters:

					skf = StratifiedKFold(n_splits=5)
					model = MLPClassifier(alpha=0.00001, hidden_layer_sizes=(200,200), max_iter=int(itr))
					print(np.unique(y))
					scores  = cross_validate(model, x, y, cv=skf, scoring='accuracy', return_train_score=True)
					print("Best -- Iter Num: ", itr)
					train_scores = scores['train_score']
					test_scores = scores['test_score']
					train_scores_avg = np.mean(train_scores)
					test_scores_avg = np.mean(test_scores)
					train_scores_plt.append(train_scores_avg)
					test_scores_plt.append(test_scores_avg)
				name = "ANN_Wine_" + plts[j]
				title = "ANN (Wine) - Best"
				Metrics().plot_learning_curve_itr(name, title, iters, train_scores_plt, test_scores_plt)


			train_scores_plt = []
			test_scores_plt = []
			if j == 1:


				for itr in iters:
					skf = StratifiedKFold(n_splits=5)
					model = MLPClassifier(alpha=0.0001, hidden_layer_sizes=(125,125), max_iter=int(itr))
					print(np.unique(y))
					scores  = cross_validate(model, x, y, cv=skf, scoring='accuracy', return_train_score=True)
					print("Worst -- Iter Num: ", itr)
					train_scores = scores['train_score']
					test_scores = scores['test_score']
					train_scores_avg = np.mean(train_scores)
					test_scores_avg = np.mean(test_scores)
					train_scores_plt.append(train_scores_avg)
					test_scores_plt.append(test_scores_avg)
				name = "ANN_Wine_" + plts[j]
				title = "ANN (Wine) - Worst"
				Metrics().plot_learning_curve_itr(name, title, iters, train_scores_plt, test_scores_plt)


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
		# 	'alpha': [0.00001, 0.0001, 0.001, 0.01],
		# 	'hidden_layer_sizes': [(15,15), (50,50), (75,75),(125,125)]
		# 	}
		# param_grid = {
		# 	'alpha': [0.00001, 0.0001, 0.001],
		# 	'hidden_layer_sizes': [(200,200),(125, 125)]
		# 	}
		# name = "ANNMetrics_Default"
		# met = Metrics()
		# model = MLPClassifier()
		# met.cross_validation_scores(model, param_grid, x, y, name)
		# model = MLPClassifier(alpha=0.00001, hidden_layer_sizes=(200,200), max_iter=10000)
		# model.fit(x,y)
		# print("iters: ", model.n_iter_)



		for j in range(len(plts)):

			train_scores_plt = []
			test_scores_plt = []
			if j == 0:


				for itr in iters:

					skf = StratifiedKFold(n_splits=5)
					model = MLPClassifier(alpha=0.0001, hidden_layer_sizes=(200,200), max_iter=int(itr))
					print(np.unique(y))
					scores  = cross_validate(model, x, y, cv=skf, scoring='accuracy', return_train_score=True)
					print("Best -- Iter Num: ", itr)
					train_scores = scores['train_score']
					test_scores = scores['test_score']
					train_scores_avg = np.mean(train_scores)
					test_scores_avg = np.mean(test_scores)
					train_scores_plt.append(train_scores_avg)
					test_scores_plt.append(test_scores_avg)
				name = "ANN_Default_" + plts[j]
				title = "ANN (Default) - Best"
				Metrics().plot_learning_curve_itr(name, title, iters, train_scores_plt, test_scores_plt)


			train_scores_plt = []
			test_scores_plt = []
			if j == 1:


				for itr in iters:
					skf = StratifiedKFold(n_splits=5)
					model = MLPClassifier(alpha=0.001, hidden_layer_sizes=(125,125), max_iter=int(itr))
					print(np.unique(y))
					scores  = cross_validate(model, x, y, cv=skf, scoring='accuracy', return_train_score=True)
					print("Worst -- Iter Num: ", itr)
					train_scores = scores['train_score']
					test_scores = scores['test_score']
					train_scores_avg = np.mean(train_scores)
					test_scores_avg = np.mean(test_scores)
					train_scores_plt.append(train_scores_avg)
					test_scores_plt.append(test_scores_avg)
				name = "ANN_Default_" + plts[j]
				title = "ANN (Default) - Worst"
				Metrics().plot_learning_curve_itr(name, title, iters, train_scores_plt, test_scores_plt)








		# for j in range(len(plts)):

		# 	if j == 0:

		# 		model = MLPClassifier(alpha=lr, activation='identity', hidden_layer_sizes=(25,50), max_iter=10000, n_iter_no_change=100)
		# 		met = Metrics()
		# 		met.learning_curve_data(model, x, y)
		# 		met.plot_learning_curve("DT_Wine_" + plts[j],"Decision Trees (Wine)")


		# lr = 0.001
		# model = MLPClassifier(alpha=lr, activation='identity', hidden_layer_sizes=(25,50), max_iter=1000, n_iter_no_change=100)
		# model.fit(x, y)

		# plt.plot(model.loss_curve_)
		# plt.show()