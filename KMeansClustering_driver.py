#Faith Seely
#I worked alone on this assignment
#Script for K-Means Clustering
#Please place your FUNCTION code for step 4 here.
import KMeansClustering_functions as kmc #Use kmc to call your functions
import numpy as np

glucose, hemoglobin, classification = kmc.openckdfile()
normGlucose, normHemoglobin, classification = kmc.normalizeData(glucose, hemoglobin, classification)

k = 2
initialCentroidGlucose, initialCentroidHemoglobin = kmc.generateInitialCentroids(k)
normCentroidGlucose, normCentroidHemoglobin, centroidClass = kmc.generateNormCentroids(k)
print('Starting Centroid Glucose Values:', normCentroidGlucose)
print('Starting Centroid Hemoglobin Values:', normCentroidHemoglobin)

newClassifications = np.zeros(glucose.size)
while not np.array_equal(initialCentroidGlucose, normCentroidGlucose) and not np.array_equal(initialCentroidHemoglobin, normCentroidHemoglobin):
    for i in range(len(normCentroidGlucose)):
        initialCentroidGlucose[i] = normCentroidGlucose[i]
        initialCentroidHemoglobin[i] = normCentroidHemoglobin[i]
    for i in range(len(normGlucose)):
        pointClass = kmc.nearestCentroidClassifier(normGlucose[i], normHemoglobin[i], normCentroidGlucose, normCentroidHemoglobin, centroidClass)
        newClassifications[i] = pointClass
    for i in range(k):
        meanGlucose, meanHemoglobin = kmc.calculateClusterMean(i, normGlucose, normHemoglobin, newClassifications)
        normCentroidGlucose[i] = meanGlucose
        normCentroidHemoglobin[i] = meanHemoglobin

unscaled_glucose, unscaled_hemoglobin = kmc.unscaleCentroids(normCentroidGlucose, normCentroidHemoglobin, glucose, hemoglobin)

print('Normalized Centroid Glucose:', normCentroidGlucose)
print('Normalized Centroid Hemoglobin:', normCentroidHemoglobin)
print('Unscaled Centroid Glucose:', unscaled_glucose)
print('Unscaled Centroid Hemoglobin:', unscaled_hemoglobin)
print(newClassifications)

kmc.graphKMeans(normGlucose, normHemoglobin, newClassifications, normCentroidHemoglobin, normCentroidGlucose)
kmc.graphKMeans(glucose, hemoglobin, newClassifications, unscaled_hemoglobin, unscaled_glucose)

kmc.calculateAccuracy(classification, newClassifications)
