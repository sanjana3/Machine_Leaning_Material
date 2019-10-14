import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.datasets import load_iris

import matplotlib.pyplot as plt

def main():
	iris_df = load_iris()
	#print(iris_df.feature_names)
	#print(iris_df.target[0:3,:])
	print(iris_df.target_names)

	x_train = iris_df.data[0:126]
	y_train = iris_df.target[0:126]

	x_test = iris_df.data[126:]
	y_test = iris_df.target[126:]

	#print("Shape of x:")
	
	#KNN Classifier
	knn1 = KNeighborsClassifier(n_neighbors = 1)
	knn5 = KNeighborsClassifier(n_neighbors = 5)

	#Train our KNNs
	knn1.fit(x_train,y_train)
	knn5.fit(x_train,y_train)

	print("n-neighbour = 1: Predicts: ",knn1.predict(x_test))
	print("n-neighbour = 5: Predicts: ",knn5.predict(x_test))
	print("Actual labels: ",y_test)

	#Logistic Regression
	logreg = LogisticRegression()
	
	logreg.fit(x_train,y_train)

	print("Predicted labels for logreg: ",logreg.predict(x_test))
	print("Actual labels for logreg: ",y_test)

	#Plotting the accuracy vs number of neighbours
	k_range = list(range(1,26,2))
	scores = []
	for k in k_range:
		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(x_train,y_train)
		y_pred = knn.predict(x_test)
		scores.append(metrics.accuracy_score(y_test,y_pred))

	plt.rcParams.update({'font.size':18})
	plt.plot(k_range,scores,'ro',linewidth=2.0,linestyle='-')
	plt.xlabel('k')
	plt.ylabel('Accuracy')
	plt.show()
	pass

if __name__ == '__main__':
	main()

