# classification-algorithm
# Did some classification and clustering of iris_data using k-means clustering, K-nearest neighbor and support vector machine classification algorithm

import scipy
from sklearn import datasets, neighbors, svm, cluster

iris = datasets.load_iris()
#diabetes = datasets.load_diabetes()

#print (iris.data)
#print (iris.target)
#print(iris.data.shape)

#print(diabetes.data)

# Use KNN classification algorithm for classification
knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data,iris.target)                                 # learn from the existing data
knn.predict([[5.0, 3.0, 5.0, 2.0]])                            # predict the unknown data using classification model


#Use kmeans clustering algorithm to cluster
kmeans = cluster.KMeans(n_clusters = 3).fit(iris.data)
pred = kmeans.predict(iris.data)

for label in pred:
    print (label, end = ' ')                                  # print label of each data entry that has been predicted

for label in iris.target:                                     # print label of each data entry with original marked
    print (label, end = '')


# Use support vector machine algorihtm for classification
svc = svm.LinearSVC()
svc.fit(iris.data, iris.target)
svc.predict([[5.0, 3.0, 5.0, 2.0]])
