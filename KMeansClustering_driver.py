#Please place your FUNCTION code for step 4 here.
import KMeansClustering_functions as kmc #Use kmc to call your functions
import numpy as np
import matplotlib.pyplot as plt

glucose, hemoglobin, classification = kmc.openckdfile()
normGlucose, normHemoglobin, classification = kmc.normalizeData(glucose, hemoglobin, classification)

initialCentroidGlucose, initialCentroidHemoglobin = kmc.generateInitialCentroids(2)
normCentroidGlucose, normCentroidHemoglobin, centroidClass = kmc.generateNormCentroids(2)

newClassifications = np.zeros(glucose.size)
while not np.array_equal(initialCentroidGlucose, normCentroidGlucose) and not np.array_equal(initialCentroidHemoglobin, normCentroidHemoglobin):
    for i in range(len(normCentroidGlucose)):
        initialCentroidGlucose[i] = normCentroidGlucose[i]
        initialCentroidHemoglobin[i] = normCentroidHemoglobin[i]
    for i in range(len(normGlucose)):
        pointClass = kmc.nearestCentroidClassifier(normGlucose[i], normHemoglobin[i], normCentroidGlucose, normCentroidHemoglobin, centroidClass)
        newClassifications[i] = pointClass
    for i in range(2):
        meanGlucose, meanHemoglobin = kmc.calculateClusterMean(i, normGlucose, normHemoglobin, newClassifications)
        normCentroidGlucose[i] = meanGlucose
        normCentroidHemoglobin[i] = meanHemoglobin

unscaled_glucose, unscaled_hemoglobin = kmc.unscaleCentroids(normCentroidGlucose, normCentroidHemoglobin, glucose, hemoglobin)

print('Normalized Centroid Glucose:', normCentroidGlucose)
print('Normalized Centroid Hemoglobin:', normCentroidHemoglobin)
print('Unscaled Centroid Glucose:', unscaled_glucose)
print('Unscaled Centroid Hemoglobin:', unscaled_hemoglobin)
print(newClassifications)

plt.figure()
plt.plot(normHemoglobin[newClassifications==1], normGlucose[newClassifications==1],'b.')
plt.plot(normHemoglobin[newClassifications==0], normGlucose[newClassifications==0],'r.')
plt.plot(normCentroidHemoglobin, normCentroidGlucose, 'k*', label = 'Centroid')
plt.xlabel('Hemoglobin')
plt.ylabel('Glucose')
plt.legend()

plt.figure()
plt.plot(hemoglobin[newClassifications==1], glucose[newClassifications==1],'b.', label = 'Not CKD')
plt.plot(hemoglobin[newClassifications==0], glucose[newClassifications==0],'r.', label = 'CKD')
plt.plot(unscaled_hemoglobin, unscaled_glucose, 'k*', label = 'Centroid')
plt.xlabel('Hemoglobin')
plt.ylabel('Glucose')
plt.legend()
plt.show()