import matplotlib.pyplot as plot
import matplotlib.legend_handler as legend_handler
import pandas as pd
import random
import numpy as np
import math

data = pd.read_csv('2015.csv')
actual = pd.read_csv('2016.csv')

# Create a date column from the month and day
data["DATE"] = data["MONTH"].map(str) + "/" + data["DAY"].map(str)
actual["DATE"] = actual["MONTH"].map(str) + "/" + actual["DAY"].map(str)

# Number days sequentially in a column
data["DAYNUM"] = range(len(data["DATE"].map(str)))
actual["DAYNUM"] = range(len(actual["DATE"].map(str)))

# Create a Label column for each observation that we can later set
data["LABEL"] = range(data.shape[0])
actual["LABEL"] = range(actual.shape[0])

# Create a column to store the euclidean distance of each observation
data["EUCL"] = range(data.shape[0])
actual["EUCL"] = range(actual.shape[0])

# Calculate euclidean distance
def euclidean_distance(daynum, tmin, tavg):
  return math.sqrt( daynum**2 + tmin**2 + tavg**2 )

# Set the EUCL column of all observations
for obs in range(0, data.shape[0]): 
  data["EUCL"][obs] = euclidean_distance(data["DAYNUM"][obs], data["TMIN"][obs], data["TAVG"][obs])

# Set the EUCL column of all observations
for obs in range(0, actual.shape[0]): 
  actual["EUCL"][obs] = euclidean_distance(actual["DAYNUM"][obs], actual["TMIN"][obs], actual["TAVG"][obs])

# Create and return an array of k cluster centroids
def pick_centroids(k):
  # Forgy Method - randomly select k observations to use as the initial centroids
  centroids = [] # np.array([])
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
    centroid = closest_centroid(centroids, data["EUCL"][obs])
    data["LABEL"][obs] = centroid

# Assign the observations to the appropriate centroids
def assign_actual_to_centroids(centroids, centroid_labels):
  for obs in range(0, actual.shape[0]):
    centroid = closest_centroid(centroids, actual["EUCL"][obs])
    actual["LABEL"][obs] = centroid_labels[str(centroid)]

# Update k existing centroids
def update_centroids(centroids):
  sums = {}
  totals = {}
  averages = []

  for centroid in centroids:
  	sums[str(centroid)] = 0
  	totals[str(centroid)] = 0

  for obs in range(0, data.shape[0]):
    centroid = data["LABEL"][obs]
    sums[str(centroid)] = sums[str(centroid)] + data["EUCL"][obs]
    totals[str(centroid)] = totals[str(centroid)] + 1

  for centroid in centroids:
  	temp = sums[str(centroid)] / totals[str(centroid)]
  	centroid = temp

  return centroids

# Cluster with k centroids
def cluster(k):
  centroids = pick_centroids(k)
  assign_data_to_centroids(centroids)
  for _ in range(0, 10):
    update_centroids(centroids)
  return centroids

# Test k values from 2 to 7
for k in range(2, 8):
  centroids = cluster(k)
  centroid_labels = {}
  for centroid in centroids:
    avg = 0
    num = 0
    for obs in range(0, data.shape[0]):
      if data["LABEL"][obs] == centroid:
        avg += data["TMAX"][obs]
        num += 1
    avg = avg/num
    centroid_labels[str(centroid)] = avg

  assign_actual_to_centroids(centroids, centroid_labels)

  fig, axes = plot.subplots(nrows=1, ncols=1, sharey=False, sharex=False)

  # Predictions for 2016
  axes.plot(actual["TMAX"], label='Actual Highs')
  axes.plot(actual["LABEL"], label='Predicted Highs')
  axes.set_title('K-Means Clustering Predictions For K=' + str(k))
  axes.legend()

plot.show()






