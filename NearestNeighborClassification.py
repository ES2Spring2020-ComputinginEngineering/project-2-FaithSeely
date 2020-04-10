#Please put your code for Step 2 and Step 3 in this file.
#Faith Seely
#I worked alone on this assignment
#Functions and Graphs for Nearest Neighbor and k-Nearest Neighbor

#Import Statements
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# FUNCTIONS

def openckdfile():
    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)
    return glucose, hemoglobin, classification

def normalizeData(glucose, hemoglobin, classification):
#Normalizes the given data by finding minimums and maxiumums of the data and
#scaling it so the new values are between 0 and 1.
#Takes parameters glucose, hemoglobin, and classification, which are arrays
#of levels of glucose and hemoglobin in a person and their CKD diagnosis.
#This function returns an array of the scaled glucose values, an array of the
#scaled hemoglobin values, and the classification array, which was not altered.
    normGlucose = np.zeros(glucose.size)
    a = glucose.min()
    b = glucose.max()
    normHemoglobin = np.zeros(hemoglobin.size)
    c = hemoglobin.min()
    d = hemoglobin.max()
    for i in range(len(glucose)):
        normPoint = (glucose[i] - a)/(b-a)
        normGlucose[i] = normPoint
    for i in range(len(hemoglobin)):
        normPoint = (hemoglobin[i]-c)/(d-c)
        normHemoglobin[i] = normPoint
    return normGlucose, normHemoglobin, classification

def graphData(glucose, hemoglobin, classification):
#Graphs the given data as a scatter plot with points that correspond to CKD as
#red and points that don't correspond to CKD as blue.
#Takes parameters glucose, hemoglobin, and classification, which are arrays
#of levels of glucose and hemoglobin in a person and their CKD diagnosis.
#This is a void function
    plt.figure()
    plt.plot(hemoglobin[classification==1], glucose[classification==1],'b.', label = 'Not CKD')
    plt.plot(hemoglobin[classification==0], glucose[classification==0],'r.', label = 'CKD')
    plt.xlabel('Hemoglobin')
    plt.ylabel('Glucose')
    plt.legend()
    plt.show()

def createTestCase():
#Generates two random floats between 0 and 1 that corresponds to scaled values
#of a person's hemoglobin and glucose levels. These random values will fall
#within the minimum and maximum values of the training scale when unscaled.
#This function takes no parameters.
#This function returns the randomly generated,scaled hemoglobin and glucose values.
    newhemoglobin = random.random()
    newglucose = random.random()
    return newhemoglobin, newglucose

def calculateDistanceArray(newglucose, newhemoglobin, glucose, hemoglobin):
#Calculates the distance between the given test case point and all points in
#the given hemoglobin and glucose data arrays and generates an array of all the
#calculated distances.
#Parameters newglucose and newhemoglobin are the scaled glucose level and the
#scaled hemoglobin level of the test case, respectively. Parameters glucose and
#hemoglobin are arrays of normalized glucose and hemoglobin data, respectively.
#This function returns an array of all the calculated distances between points.
    distanceArray = np.zeros(hemoglobin.size)
    for i in range(len(hemoglobin)):
        a = (hemoglobin[i] - newhemoglobin)**2
        b = (glucose[i] - newglucose)**2
        distance = math.sqrt(a+b)
        distanceArray[i] = distance
    return distanceArray

def nearestNeighborClassifier(newglucose, newhemoglobin, glucose, hemoglobin, classification):
#Finds the index of the minimum distance in the given distance array and
#identifies the class of the point with that index (the closest point).
#Parameters newglucose and newhemoglobin are the scaled glucose level and the
#scaled hemoglobin level of the test case, respectively. Parameters glucose and
#hemoglobin are arrays of normalized glucose and hemoglobin data, respectively.
#This function returns the class of the closest point (1 or 0).
    distanceArray = calculateDistanceArray(newglucose, newhemoglobin, glucose, hemoglobin)
    min_index = np.argmin(distanceArray)
    nearest_class = classification[min_index]
    return nearest_class

def graphTestCase(newglucose, newhemoglobin, glucose, hemoglobin, classification):
#Graphs the arrays of training data and the test case. Test case is graphed as
#a magenta star. CKD point are red and not CKD points are blue. The function
#can use raw or normalized data.
#Parameters newglucose and newhemoglobin are the glucose level and the
#hemoglobin level of the test case, respectively. Parameters glucose and
#hemoglobin are arrays of the training set glucose and hemoglobin data,
#respectively. This is a void function.
    plt.figure()
    plt.plot(hemoglobin[classification==1], glucose[classification==1],'b.', label = 'Not CKD')
    plt.plot(hemoglobin[classification==0], glucose[classification==0],'r.', label = 'CKD')
    plt.plot(newglucose, newhemoglobin,'m*')
    plt.xlabel('Hemoglobin')
    plt.ylabel('Glucose')
    plt.legend()

def kNearestNeighborClassifier(k, newglucose, newhemoglobin, glucose, hemoglobin, classification):
#Finds the indeces of the k points in the given training set that are the
#closest to the given test case and determines the mean classification of those
#points, which will be what the classification of the test case is.
#Parameters newglucose and newhemoglobin are the scaled glucose level and the
#scaled hemoglobin level of the test case, respectively. Parameters glucose and
#hemoglobin are arrays of normalized glucose and hemoglobin data, respectively.
#Parameter k is an odd integer that is the number of closest points to be
#considered in the determination of the classification of the test case.
    distanceArray = calculateDistanceArray(newglucose, newhemoglobin, glucose, hemoglobin)
    sorted_indices = np.argsort(distanceArray)
    k_indices = sorted_indices[:k]
    k_classifications = classification[k_indices]
    sorted_k_classifications = np.sort(k_classifications)
    return sorted_k_classifications.median()
    
# MAIN SCRIPT
glucose, hemoglobin, classification = openckdfile()

#plt.figure()
#plt.plot(hemoglobin[classification==1],glucose[classification==1], "k.", label = "Class 1")
#plt.plot(hemoglobin[classification==0],glucose[classification==0], "r.", label = "Class 0")
#plt.xlabel("Hemoglobin")
#plt.ylabel("Glucose")
#plt.legend()
#plt.show()

normGlucose, normHemoglobin, classification = normalizeData(glucose, hemoglobin, classification)
graphData(normGlucose, normHemoglobin, classification)

newhemoglobin, newglucose = createTestCase()
print(newhemoglobin, newglucose)
graphTestCase(newglucose, newhemoglobin, normGlucose, normHemoglobin, classification)

print(nearestNeighborClassifier(newglucose, newhemoglobin, normGlucose, normHemoglobin, classification))
print(kNearestNeighborClassifier(3, newglucose, newhemoglobin, normGlucose, normHemoglobin, classification))
