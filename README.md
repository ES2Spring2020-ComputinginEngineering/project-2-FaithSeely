This project is based on an example and dataset from Data Science course developed at Berkeley (Data8.org).

NearestNeighborClassification.py contains the functions needed to use the Nearest Neighbor and K-Nearest Neighbor classification methods on a set of data. It contains the functions:
openckdfile - retrieves the data set on chronic kidney disease diagnonses.
normalizeData - takes arrays of data and creates new arrays of the normalized values of the data.
graphData - generates a scatter plot of the inputted data that is colorcoded based on points' classifications.
createTestCase - randomly generates two floats between 0 and 1 that serve as a random set of normalized glucose and hemoglobin levels.
calculateDistanceArray - generates an array that contains the distances between an inputted point and all points in an inputted data set.
nearestNeighborClassifier - uses the calculateDistanceArray function then finds the index of the minimum distance in the array and uses it to find the classification of the point in the data set with that index.
graphTestCase - similar to graphData but it also graphs the inputted test case as a magenta star.
kNearestNeighborClassifier - uses the calculateDistanceArray function, sorts the distance array's indices from minimum to maximum, selects the first k indices, finds and sorts the classifications associated with those indices then finds the median classification which is also the mode.

KMeansClustering_functions.py contains all the functions necessary for running KMeansClustering_driver.py. KMeansClustering_functions.py contains the functions:

openckdfile()
- Retrieves the data set on chronic kidney disease diagnonses.
  Parameters: None.
  Returns: This is a void function
  
normalizeData(glucose, hemoglobin, classification)
- Creates new arrays of the normalized values of the inputted data. 
  Parameters: glucose - array of glucose values to be normalized.
              hemoglobin -  array of hemoglobin values to be normalized.
              classification - array of classifications for the points in the data set.
  Returns: an array of the scaled glucose values, an array of the scaled hemoglobin values, and the classification array, which was not            altered.

calculateDistanceArray(newglucose, newhemoglobin, glucose, hemoglobin)
- Generates an array of the distances between a specific point and all points in the inputted data set.
  Parameters: newglucose - the normalized glucose value of the point being compared to the points in the data set.
              newhemoglobin - the normalized hemoglobin value of the point being compared to the points in the data set.
              glucose - an array of normalized glucose values from a data set to compare the inputted point to.
              hemoglobin - an array of normalized hemoglobin values from a data set to compare the inputted point to.
  Returns: an array of the distances that were calculated between the specific point and the points in the inputted data set.

nearestCentroidClassifier(pointGlucose, pointHemoglobin, centroidGlucose, centroidHemoglobin, centroidClass)
- Uses the calculateDistanceArray function to calculate the distance between an inputted point and a set of centroids then finds the index of the minimum distance in the array and uses it to find the classification of the centroid that corresponds to that index.
  Parameters: pointGlucose - the normalized glucose level of a specific point.
              pointHemoglobin - the normalized hemoglobin level of a specifc point.
              centroidGlucose - an array of normalized glucose values for a set of centroids.
              centroidHemoglobin - an array of normalized hemoglobin values for a set of centroids.
              centroidClass - an array of the classifications for the centroids.
  Returns: the class of the closest centroid to the specific point.

generateInitialCentroids(k)
-Generates two arrays of k zeros. These arrays are used when checking the condition of the while loop in the K-Means Clustering program.
  Parameters: k - an integer that represents the number of centroids to be generated.
  Returns: two arrays of k zeros.

generateNormCentroids(k)
- Randomly generates two arrays of k values between 0 and 1 to be used as normalized centroid values.
  Parameters: k - an integer that represents the number of centroids to be generated.
  Returns: an array of the normalized glucose values of the centroids, an array of the normalized hemoglobin values of the centroids,              and an array of the centroid classifications.

calculateClusterMean(clusterclass, glucose, hemoglobin, newclassification)
- Calculates the mean values of glucose and hemoglobin for a cluster.
  Parameters: clusterclass - the classification of the cluster that the mean is to be calculated for.
              glucose - the array of glucose values of the data set
              hemoglobin - the array of hemoglobin values for the data set.
              newclassification - the array of classifications for the points in the data set determined by which centroid they're                                         closest to.
  Returns: the mean glucose value and the mean hemoglobin value for the cluster.

unscaleCentroids(normGlucose, normHemoglobin, glucose, hemoglobin)
- Unscales the normalized values of given set of centroids by finding the minimum and maximum of the data set and applying reversing the process used in the normalizeData function.
  Parameters: normGlucose - the array of normalized glucose values for the centroids.
              normHemoglobin - the array of normalized hemoglobin values for the centroids.
              glucose - the array of the glucose values for the data set.
              hemoglobin - the array of hemoglobin values for the data set.
  Returns: an array of the unscaled glucose values for the centroids and an array of the unscaled hemoglobin values for the centroids.

graphKMeans(glucose, hemoglobin, assignment, centroidHemoglobin, centroidGlucose)
- Graphs a data set and its clusters' centroids with different colors for the clusters and black diamonds for the centroids.
  Parameters: glucose - the array of the glucose values for the data set.
              hemoglobin - the array of hemoglobin values for the data set.
              assignment - an array of classifications for the points in the data set.
              centroidHemoglobin - the array for the glucose values of the centroids.
              centroidGlucose - the array for the hemoglobin values of the centroids.
  Returns: This is a void function
  
  calculateAccuracy(classification, newclassification)
  - Calculates and prints the True Positives Rate, False Positives Rate, True Negatives Rate, and False Negatives Rate.
    Parameters: classification - an array of the actual classifications of the data set.
                newclassification - an array of the classification determined by K-Means Clustering.
    Returns: This is a void function.

KMeansClustering_driver.py is the script for using the K-Means Clustering classification method on a data set with two features. This script calls functions from KMeansClustering_functions.py and uses a while loop that ends when the centroids do not change after the most recent Update step. At the beginning of the while loop, it compares "initial centroid values" to the "normalized centroid values". If the condition is true, the "initial centroid values" are updated to have the same values as the "normalized centroid values", the mean values are calculated for each cluster, and the "normalized centroid values" are updated to be the calculated mean values. Once the mean values for the clusters stop changing the "initial centroid values" and "normalized centroid values" will be equal and the loop will end.
To run the KMeansClustering_driver.py, make k in line 11 equal the number of clusters you want to be found. If you choose a number other than 2, I suggest commenting out the kmc.calculateAccuracy() function in line 41 or else it will print out the error messages I wrote into the function. When the program runs, it will print out the random normalized values that the generated centroids start as, the final normalized values of the centroids, the final unscaled values of the centroids, the classifications of the points in the data set based on the generated clusters, a normalized graph of the clusters and centroids, an unscaled graph of the clusters and centroids, and the accuracy rates. Sometimes the resulting classification numbers assigned to the clusters will be flipped compared to the actual classfications which will cause the calculated accuracy rates to switch with each other (ex. True Positives Rate switches with False Negatives Rate).
