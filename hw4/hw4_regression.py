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

data["DATE"] = data["MONTH"].map(str) + "/" + data["DAY"].map(str)
actual["DATE"] = actual["MONTH"].map(str) + "/" + actual["DAY"].map(str)

data["DAYNUM"] = range(len(data["DATE"].map(str)))
actual["DAYNUM"] = range(len(actual["DATE"].map(str)))

# Generate beginning w array
w = []
for i in range(0, 4):
	w.append(random.uniform(-1, 1))

# Hold error over learning iterations
err_over_time = []

# Learn new w's
def learn_w(alpha):
	for _ in range(0, 100):
		for row in range(0, data.shape[0]):
			x = [1, float(data["DAYNUM"][row]), float(data["TMIN"][row]), float(data["TAVG"][row])]
			actual_y = actual["TMAX"][row]
			err = compute_error(w, x, actual_y)
			if row == 0:
				err_over_time.append( abs(err) )
			for i in range(0, len(w)):
				w[i] = lms(w[i], err, alpha, x[i])

def lms(wi, error, alpha, xij):
	return wi - alpha * error * xij

def predict(w, x):
	return w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + w[3]*x[3]

def compute_error(w, x, actual_y):
	return predict(w, x) - actual_y

# Define alpha
alpha = 0.00001

# Learn
learn_w(alpha)

# def test_predict(w, x):
# 	for i in range(0, 4):
# 		print("w[" + str(i) + "]: " + str(w[i]))
# 		print("x[" + str(i) + "]: " + str(x[i]))
# 	return w[0] + w[1]*x[1] + w[2]*x[2] + w[3]*x[3]


# Create predictions for every day

f = []
for day in range(0, data.shape[0]):
	x = [1, float(data["DAYNUM"][day]), float(data["TMIN"][day]), float(data["TAVG"][day])]
	f.append(predict(w, x))

graph_x = actual["DATE"]
y1 = actual["TMAX"]

fig, axes = plot.subplots(nrows=2, ncols=1, sharey=False, sharex=False)
axes[0].plot(y1)
axes[0].plot(f)
plot.sca(axes[0])
plot.xticks(range(len(graph_x)), graph_x)

axes[1].plot(range(len(err_over_time)), err_over_time)

plot.show()


# fig, axs = plot.subplots(1, 1, sharey=True)
# actual.plot(kind='line', x='DATE', y='TMAX', ax=axs[0])
# data.plot(kind='scatter', x='TMIN', y='TMAX', ax=axs[1])
# data.plot(kind='scatter', x='TAVG', y='TMAX', ax=axs[2])
# plot.show()

