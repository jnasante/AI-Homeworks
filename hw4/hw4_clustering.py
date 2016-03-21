import matplotlib.pyplot as plot
import pandas as pd
import random

data = pd.read_csv('2015.csv')
actual = pd.read_csv('2016.csv')

YEAR = data.YEAR
MONTH = data.MONTH
DAY = data.DAY
TMIN = data.TMIN
TAVG = data.TAVG
TMAX = data.TMAX

data["DATE"] = data["MONTH"].map(str) + "/" + data["DAY"].map(str)
actual["DATE"] = actual["MONTH"].map(str) + "/" + actual["DAY"].map(str)

data["DAYNUM"] = range(len(data["DATE"].map(str)))
actual["DAYNUM"] = range(len(actual["DATE"].map(str)))


for k in range(2, 8):
	