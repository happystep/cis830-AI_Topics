import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics 
from scipy.spatial.distance import cdist 
import pandas as pd
import re
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

data = pd.read_csv("/Users/luis/Documents/Spring 2020/Current_Topics_AI/Code/Results/results.csv")
dataT = data.T
X = dataT.to_numpy()
Y = [0, 1, 2, 3]
#Visualizing the data 
plt.plot() 
plt.title('Dataset') 
plt.xlabel('Roles')
plt.ylabel('In Degree') 

plt.scatter(Y, X[:, 1]) 
plt.show()

distortions = [] 
inertias = [] 
mapping1 = {} 
mapping2 = {} 
K = range(1,5) 
  
for k in K: 
    #Building and fitting the model 
    kmeanModel = KMeans(n_clusters=k).fit(X) 
    kmeanModel.fit(X)     
      
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                      'euclidean'),axis=1)) / X.shape[0]) 
    inertias.append(kmeanModel.inertia_) 
  
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                 'euclidean'),axis=1)) / X.shape[0] 
    mapping2[k] = kmeanModel.inertia_   

plt.plot(K, distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method using Distortion') 
plt.show()

# 3 number of clusters is best
y_pred = KMeans(n_clusters=3).fit_predict(X)
plt.scatter(Y, X[:, 1], c=y_pred)
plt.title("K-Means Clusters")
plt.xlabel('Roles')
plt.ylabel('In Degree') 
plt.show()  
