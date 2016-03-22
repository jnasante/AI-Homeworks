import matplotlib.pyplot as plot
import pandas as pd
import random
import numpy as np
import math

data = pd.read_csv('2015.csv')
actual = pd.read_csv('2016.csv')

# 
data["DATE"] = data["MONTH"].map(str) + "/" + data["DAY"].map(str)
actual["DATE"] = actual["MONTH"].map(str) + "/" + actual["DAY"].map(str)

# 
data["DAYNUM"] = range(len(data["DATE"].map(str)))
actual["DAYNUM"] = range(len(actual["DATE"].map(str)))

# 
data["LABEL"] = range(data.shape[0])

# 
data["EUCL"] = range(data.shape[0])

# Calculate euclidean distance
def euclidean_distance(daynum, tmin, tavg):
  return math.sqrt( daynum**2 + tmin**2 + tavg**2 )

# 
for obs in range(0, data.shape[0]): 
  data["EUCL"][obs] = euclidean_distance(data["DAYNUM"][obs], data["TMIN"][obs], data["TAVG"][obs])

# Create and return an array of k cluster centroids
def pick_centroids(k):
  # Forgy Method - randomly select k observations to use as the initial centroids
  centroids = [] # np.array([])
  rand_used = []
  for _ in range(0, k):
    obs = random.uniform(0, data.shape[0])
    while obs in rand_used:
      obs = random.uniform(0, data.shape[0])
    
    rand_used.append(obs)
    centroids.append( data["EUCL"][int(obs)] )

  return centroids

# 
def closest_centroid(centroids, value):
  closest = centroids[0]
  for centroid in centroids:
    if abs(centroid - value) < abs(closest - value):
      closest = centroid

  return closest

# Assign the observations to the appropriate centroids
def assign_data_to_centroids(centroids, k):
  for obs in range(0, data.shape[0]):
    label = closest_centroid(centroids, data["EUCL"][obs])
    data["LABEL"][obs] = label

# Update k existing centroids
def update_centroids(centroids, k):
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
  assign_data_to_centroids(centroids, k)
  for _ in range(0, 10):
    update_centroids(centroids, k)
  return centroids

# Test k values from 2 to 7
for k in range(2, 8):
	cluster(k)
