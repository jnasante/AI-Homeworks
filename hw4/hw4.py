import matplotlib.pyplot as plot
import pandas as pd
import random

data = pd.read_csv('2015.csv')
actual = pd.read_csv('2016.csv')

# print(data.head())
# print(data.shape[0])

YEAR = data.YEAR
MONTH = data.MONTH
DAY = data.DAY
TMIN = data.TMIN
TAVG = data.TAVG
TMAX = data.TMAX

data["DATE"] = data["MONTH"].map(str) + "-" + data["DAY"].map(str)
actual["DATE"] = actual["MONTH"].map(str) + "-" + actual["DAY"].map(str)

# print(data["DATE"][0])

# Generate beginning w array
w = []
for i in range(0, 4):
	w.append(random.uniform(-1, 1))

# Learn new w's
def learn_w(alpha):
	for _ in range(0, 10):
		for row in range(0, data.shape[0]):
			x = [data["DATE"][row], data["TMIN"][row], data["TAVG"][row]]
			actual_y = actual["TMAX"][row]
			err = compute_error(w, x, actual_y)
			for i in range(0, len(w)):
				w[i] = lms(w[i], err, alpha, xij)

def lms(wi, error, alpha, xij):
	return wi - alpha*error * xij

def predict(w, x):
	return w[0] + w[1]*x[0] + w[2]*x[1] + w[3]*x[2]

def predict_feature(w, xi):
	return 

def compute_error(w, x, actual_y):
	return predict(w, x) - actual_y

# Define alpha
alpha = 0.00001

# f = predict(w, x)



# fig, axs = plot.subplots(1, 3, sharey=True)
# data.plot(kind='line', x='DATE', y='TMAX', ax=axs[0])
# data.plot(kind='scatter', x='TMIN', y='TMAX', ax=axs[1])
# data.plot(kind='scatter', x='TAVG', y='TMAX', ax=axs[2])
# plot.show()


