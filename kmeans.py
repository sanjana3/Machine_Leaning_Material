import csv
import pandas as pd
import numpy as np
import plotly.plotly as py
import matplotlib.pyplot as plt
import plotly.tools as tls
from sklearn.preprocessing import MinMaxScaler


#Question 3-1 : Reading the seeds dataset 
seeds_data = pd.read_csv("D:/FALL2018/Data Mining/HW4/seeds_dataset.csv",encoding="ISO-8859-1",engine = "python",header=None)

#Question 3-2 : Removes the class attribute 
seed_data_new = seeds_data.drop(seeds_data.columns[7],axis=1)

#Preprocessing activities :
# 1) Rounding all the columns data upto 2 places
seed_data_new1 = seed_data_new.round(2)

# 2) Simple boxplot of the data to find outliers
for i in range(1,6):
	plt.boxplot(seed_data_new1[i],positions=[2])
	plt.show()

#From the boxplots, columns 2 and 5 have outliers.

#3) Outlier Elimination

for j in range(1,6):
	arr1 = seed_data_new1[j]
	elements = np.array(arr1)

	mean = np.mean(elements)
	sd = np.std(elements)

	seed_data_new1 = [[x] for x in arr1 if (x > mean - 2 * sd)]
	seed_data_new1 = [[x] for x in seed_data_new1 if (x < mean + 2 * sd)]
	print(seed_data_new1)

#3)K-Means clustering

def kmeans(x,k):
	idx = np.arange(2,4)
	newcentroids = np.asarray(x[np.random.choice(len(x),k,replace=False),:])
	sse = np.zeros((k,1))
	oldcentroids = np.zeros((k,13))
	i = 0
	while (oldcentroids.any(newcentroids)):
		oldcentroids = newcentroids.copy()
	cluster = np.zeros((720,3))
	#
	for i in range(len(x)):
		min_dist = np.Inf
		for index,y_k in enumerate(newcentroids):
			distance = np.sqrt(np.sum(x[i,:]-y_k**2))
		if distance < min_dist:
			min_dist = distance
		idx = index
		cluster[i] = idx,min_dist,min_dist**2
		i = i+1

	for k in range(k):
		newcentroids[k] = x[np.where(cluster[:,0]==k),:].mean(axis=1)
		sse[k] = np.sum(cluster[np.where(cluster[:,0]==k),2])
		total_sse = np.sum(sse)
	return newcentroids,oldcentroids,cluster,sse,total_sse,i

k_vals = [2,3,4,5,6]
tsse_values_list = []
seed_data_new1 = seed_data_new1.as_matrix()
for k_val in k_vals:
	print(k_val)
	newcentroids,oldcentroids,cluster,sse,total_sse,num_iter = kmeans(seed_data_new,k_val)
	print('K-Value: ',k_val)
	print('Number of iterations: ',num_iter)
	print('Total SSE: ',total_sse)
	print('Centroids: ',newcentroids)