#PCA
#Perform Principal component analysis and perform clustering using first 
#3 principal component scores (both heirarchial and k mean clustering(scree plot or elbow curve) and obtain 
#optimum number of clusters and check whether we have obtained same number of clusters with the original data 
#(class column we have ignored at the begining who shows it has 3 clusters)df

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('wine.csv')
dataset.describe()
dataset.head()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset_scaled = sc.fit_transform(dataset.iloc[:, 1:])

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
pca_values= pca.fit_transform(dataset_scaled)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]

# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1,color="red")

# Variance plot for PCA components obtained 
plt.plot(var1,color="red")

# plot between PCA1 and PCA2 
x = pca_values[:,[0]]
y = pca_values[:,[1]]
z = pca_values[:,2:3]
plt.scatter(x,y,color="red")

################### Clustering  ##########################
new_df = pd.DataFrame(pca_values[:,0:3])  #selecting first 3 PCA

from	sklearn.cluster	import	KMeans
###### screw plot or elbow curve ############
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters= i, init= 'k-means++',n_init=10, max_iter=300,random_state=0)
    kmeans.fit(new_df)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)    
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=3, init = 'k-means++', random_state = 0) 
model.fit(new_df)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
dataset['clust']=model.labels_ # creating a  new column and assigning it to new column 


dataset.iloc[:,1:14].groupby(dataset.clust).mean()

## Visualising the clusters
dataset.plot(x="Type",y = "Alcohol",c=model.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)


