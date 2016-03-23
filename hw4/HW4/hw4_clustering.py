import matplotlib.pyplot as plot
import matplotlib.legend_handler as legend_handler
import pandas as pd
import random
import numpy as np
import math

# Suppress warnings we want to ignore
pd.options.mode.chained_assignment = None  # default='warn'

data = pd.read_csv('2015.csv')
actual = pd.read_csv('2016.csv')

# Create a date column from the month and day
data["DATE"] = data["MONTH"].map(str) + "/" + data["DAY"].map(str)
actual["DATE"] = actual["MONTH"].map(str) + "/" + actual["DAY"].map(str)

# Number days sequentially in a column
data["DAYNUM"] = range(len(data["DATE"].map(str)))
actual["DAYNUM"] = range(len(actual["DATE"].map(str)))

# Create a Label column for each observation that we can later set
data["LABEL"] = np.arange(float(data.shape[0]))
actual["LABEL"] = np.arange(float(actual.shape[0]))

# Create a column to store the euclidean distance of each observation
data["EUCL"] = np.arange(float(data.shape[0]))
actual["EUCL"] = np.arange(float(actual.shape[0]))

# 
centroid_labels = {}

# Calculate euclidean distance
def euclidean_distance(daynum, tmin, tavg):
  return math.sqrt( daynum**2 + tmin**2 + tavg**2 )

# Set the EUCL column of all observations
for obs in range(0, data.shape[0]): 
  data["EUCL"][obs] = euclidean_distance(data["DAYNUM"][obs], data["TMIN"][obs], data["TAVG"][obs])

# Set the EUCL column of all observations
for obs in range(0, actual.shape[0]): 
  actual["EUCL"][obs] = euclidean_distance(actual["DAYNUM"][obs], actual["TMIN"][obs], actual["TAVG"][obs])

# Hold error over learning iterations
err_over_time = []

# Create and return an array of k cluster centroids
def pick_centroids(k):
  # Forgy Method - randomly select k observations to use as the initial centroids
  centroids = []
  rand_used = []
  for _ in range(0, k):
    obs = int(random.uniform(0, data.shape[0]))
    while obs in rand_used:
      obs = int(random.uniform(0, data.shape[0]))
    
    rand_used.append(obs)
    centroids.append( data["EUCL"][obs] )

  return centroids

# Determine, for a value, which centroid is closest
def closest_centroid(centroids, value):
  closest = centroids[0]
  for centroid in centroids:
    if abs(centroid - value) < abs(closest - value):
      closest = centroid

  return closest

# Assign the observations to the appropriate centroids
def assign_data_to_centroids(centroids):
  for obs in range(0, data.shape[0]):
    closest = closest_centroid(centroids, data["EUCL"][obs])
    data["LABEL"][obs] = closest

# Assign the observations to the appropriate centroids
def assign_actual_to_centroids(centroids):
  for obs in range(0, actual.shape[0]):
    closest = closest_centroid(centroids, actual["EUCL"][obs])
    actual["LABEL"][obs] = centroid_labels[str(closest)]

# Update k existing centroids
def update_centroids(centroids, shouldStoreError):
  sums = {}
  totals = {}
  errsum = 0  	

  for centroid in centroids:
    sums[str(centroid)] = 0
    totals[str(centroid)] = 0
    avg = 0
    num = 0

    for obs in range(0, data.shape[0]):
      if data["LABEL"][obs] == centroid:
        avg += data["TMAX"][obs]
        num += 1
    
    if (num != 0):
      avg = avg/num

    centroid_labels[str(centroid)] = avg

  for obs in range(0, data.shape[0]):
    centroid = data["LABEL"][obs]
    sums[str(centroid)] += data["EUCL"][obs]
    totals[str(centroid)] += 1
    errsum += compute_error(centroid_labels[str(centroid)], data["TMAX"][obs])

  newCentroids = []
  for centroid in centroids:
  	newCentroids.append(sums[str(centroid)] / totals[str(centroid)])

  error = errsum / data.shape[0]

  if (shouldStoreError == True):
    err_over_time.append(error)

  return newCentroids

def compute_error(predicted, target):
  return abs(predicted - target)

# Cluster with k centroids
def cluster(k):
  del err_over_time[:]
  centroids = pick_centroids(k)
  assign_data_to_centroids(centroids)

  for _ in range(0, 10):
    centroids = update_centroids(centroids, True if k==7 else False)
    assign_data_to_centroids(centroids)

  # Assign labels
  for centroid in centroids:
    avg = 0
    num = 0

    for obs in range(0, data.shape[0]):
      if data["LABEL"][obs] == centroid:
        avg += data["TMAX"][obs]
        num += 1
    
    if (num != 0):
      avg = avg/num

    centroid_labels[str(centroid)] = avg

  return centroids

# Test k values from 2 to 7
for k in range(2, 8):
  centroids = cluster(k)
  assign_actual_to_centroids(centroids)
  centroid_labels = {}

  # Predictions for 2016
  fig, axes = plot.subplots(nrows=1, ncols=1, sharey=False, sharex=False)
  axes.plot(actual["TMAX"], label='Actual Highs')
  axes.plot(actual["LABEL"], label='Predicted Highs')
  axes.set_title('K-Means Clustering Predictions For k = ' + str(k))
  axes.legend()

# Error over time
fig, axes = plot.subplots(nrows=1, ncols=1, sharey=False, sharex=False)
axes.plot(range(len(err_over_time)), err_over_time)
axes.set_title("Sum Squared Error vs Number of Iterations")

plot.show()






