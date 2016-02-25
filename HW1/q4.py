# Title: CSE5523 Machine Learning HW1 Question 4
# Author: Yiran Lawrence Luo
# Date: Feb 24, 2016
# Abstract: A simulator for testing classification accuracy using 1-NN and 3-NN. 
#	The classifiers are trained and tested using two labeled Gaussian clusters. 
# 	The goal is to find out the relationship between data's dimensionality and classification's accuracy.

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def dataGenerator(num, p, mean_distance=3):
	# Generating the two Gaussian clusters of labeled random data points
	mean = np.zeros(p)
	mean2 = np.zeros(p)
	mean2[0] = mean_distance # which is 3 in question

	cov = np.identity(p)

	# Coin toss simulator
	s = np.random.binomial(1, 0.5, num)

	c1_num = s.tolist().count(0)
	c2_num = num - c1_num
	point_c1 = np.random.multivariate_normal(mean, cov, c1_num)
	point_c2 = np.random.multivariate_normal(mean2, cov, c2_num)
	
	X = np.concatenate((point_c1, point_c2), axis=0)
	y = [0] * c1_num + [1] * c2_num
	
	return X, y

def run():
	p_list = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]

	# Getting the error rate vs p by training 1-NN
	error_list1 = []
	for p in p_list:
		X_train, y_train = dataGenerator(200, p)
		X_test, y_test = dataGenerator(1000, p)

		knn = KNeighborsClassifier(n_neighbors=1)
		knn.fit(X_train, y_train)
		error_list1.append( 1.0 - knn.score(X_test, y_test) )
		
	# Same process for 3-NN
	error_list2 = []
	for p in p_list:
		X_train, y_train = dataGenerator(200, p)
		X_test, y_test = dataGenerator(1000, p)

		knn = KNeighborsClassifier(n_neighbors=3)
		knn.fit(X_train, y_train)
		error_list2.append( 1.0 - knn.score(X_test, y_test) )

	# Plotting the two lines into one figure
	line1, = plt.plot(p_list, error_list1, 'ro-', label='1-NN')
	line2, = plt.plot(p_list, error_list2, 'bx-', label='3-NN')
	plt.xlabel('p')
	plt.ylabel('Test error rate %')
	plt.legend(handles=[line1, line2], loc=4)
	plt.savefig('Q4.png')

run()