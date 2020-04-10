#Please place your FUNCTION code for step 4 here.
#Faith Seely
#I worked alone on this assignment
#Functions for K-Means Clustering

import numpy as np
import matplotlib.pyplot as plt
import random
import math


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

def calculateDistanceArray(newglucose, newhemoglobin, glucose, hemoglobin):
#Calculates the distance between a given point and all points in a given set
#of hemoglobin and glucose data arrays for the centroids then generates an
#array of all the calculated distances.
#Parameters newglucose and newhemoglobin are the scaled glucose level and the
#scaled hemoglobin level of the point, respectively. Parameters glucose and
#hemoglobin are arrays of normalized glucose and hemoglobin data for the
#centroids, respectively. This function returns an array of all the calculated
#distances between the point and the centroids.
    distanceArray = np.zeros(hemoglobin.size)
    for i in range(len(hemoglobin)):
        a = (hemoglobin[i] - newhemoglobin)**2
        b = (glucose[i] - newglucose)**2
        distance = math.sqrt(a+b)
        distanceArray[i] = distance
    return distanceArray

def nearestCentroidClassifier(pointGlucose, pointHemoglobin, centroidGlucose, centroidHemoglobin, centroidClass):
#Finds the index of the minimum distance in the calculated distance array and
#identifies the class of the centroid with that index (the closest centroid).
#Parameters pointGlucose and pointHemoglobin are the scaled glucose level and the
#scaled hemoglobin level of a point, respectively. Parameters centroidGlucose
#and centroidHemoglobin are arrays of normalized glucose and hemoglobin data
#for all of the centroids, respectively. centroidClass is an array of the
#classifications for the centroids.
#This function returns the class of the closest centroid.
    distanceArray = calculateDistanceArray(pointGlucose, pointHemoglobin, centroidGlucose, centroidHemoglobin)
    min_index = np.argmin(distanceArray)
    nearest_class = centroidClass[min_index]
    return nearest_class

def generateInitialCentroids(k):
#Generates two arrays of k zeros to be used to compare the randomly generated
#centroids to for the while loop condition in the K-Means Clustering program.
#Parameter k is an integer that represents the number of centroids generated.
#This function returns two arrays of k zeros.
    initialCentroidGlucose = np.zeros(k)
    initialCentroidHemoglobin = np.zeros(k)
    return initialCentroidGlucose, initialCentroidHemoglobin

def generateNormCentroids(k):
#Randomly generates two arrays of values between 0 and 1 to be used as
#normalized centroid values. Parameter k is an integer that represents the
#number of centroids to be generated. This function returns an array of the
#normalized glucose values of the centroids, an array of the normalized
#hemoglobin values of the centroids, and an array of the centroid classifications.
    normCentroidGlucose = np.zeros(k)
    normCentroidHemoglobin = np.zeros(k)
    centroidClass = np.linspace(0,k-1,k)
    for i in range(k):
        glucoseValue = random.random()
        hemoglobinValue = random.random()
        normCentroidGlucose[i] = glucoseValue
        normCentroidHemoglobin[i] = hemoglobinValue
    return normCentroidGlucose, normCentroidHemoglobin, centroidClass

def calculateClusterMean(clusterclass, glucose, hemoglobin, newclassification):
#Calculates the mean values of glucose and hemoglobin for a cluster. Parameter
#clusterclass is the classification of the cluster that the mean is to be
#calculated for. glucose is the array of glucose values of the data set.
#hemoglobin is the array of hemoglobin values for the data set.
#newclassification is the array of classifications for the points in
#the data set based on which centroid they're closest to. This function returns
#the mean glucose value and the mean hemoglobin value for the cluster.
    counter = 0
    for i in range(glucose.size):
        if newclassification[i] == clusterclass:
            counter = counter + 1
        else:
            counter = counter
    clusterGlucose = np.zeros(counter)
    clusterHemoglobin = np.zeros(counter)
    counter = 0
    for i in range(glucose.size):
        if newclassification[i] == clusterclass:
            clusterGlucose[counter] = glucose[i]
            clusterHemoglobin[counter] = hemoglobin[i]
            counter = counter + 1
    meanGlucose = (clusterGlucose.sum())/len(clusterGlucose)
    meanHemoglobin = (clusterHemoglobin.sum())/len(clusterHemoglobin)
    return meanGlucose, meanHemoglobin

def unscaleCentroids(normGlucose, normHemoglobin, glucose, hemoglobin):
#Unscales the normalized values of the centroids by finding the minimum and
#maximum of the data set and applying reversing the process used in the
#normalizeData function. Parameter normGlucose is the array of normalized
#glucose values for the centroids. normHemoglobin is the array of normalized
#hemoglobin values for the centroids. glucose is the array of the glucose
#values for the data set. hemoglobin is the array of the hemoglobin values for
#the data set. This function returns an array of the unscaled glucose values
#for the centroids and an array of the unscaled hemoglobin values for the
#centroids.
    unscaled_glucose = np.zeros(normGlucose.size)
    a = glucose.min()
    b = glucose.max()
    unscaled_hemoglobin = np.zeros(normHemoglobin.size)
    c = hemoglobin.min()
    d = hemoglobin.max()
    for i in range(len(normGlucose)):
        unscaled_point = ((b-a)*normGlucose[i])+a
        unscaled_glucose[i] = unscaled_point
    for i in range(len(normHemoglobin)):
        unscaled_point = ((d-c)*normHemoglobin[i]+c)
        unscaled_hemoglobin[i] = unscaled_point
    return unscaled_glucose, unscaled_hemoglobin

