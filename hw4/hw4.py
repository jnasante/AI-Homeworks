import matplotlib.pyplot as plot
import pandas as pd
import random

data = pd.read_csv('2015.csv')
actual = pd.read_csv('2016.csv')

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
    sum = 0.0
    for row in range(0, data.shape[0]):
      x = [1, data["DAYNUM"][row], data["TMIN"][row], data["TAVG"][row]]
      y = data["TMAX"][row]
      err = compute_error(w, x, y)
      sum = sum + abs(err)
        
      for i in range(0, len(w)):
        w[i] = lms(w[i], err, alpha, x[i])

    err_over_time.append( sum/data.shape[0] )

def lms(wi, error, alpha, xij):
  return wi - alpha * error * xij

def predict(w, x):
  return w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + w[3]*x[3]

def compute_error(w, x, y):
  return predict(w, x) - y

# Define alpha
alpha = 0.00001

# Learn w's
learn_w(alpha)

# Create predictions for every day
f = []
for day in range(0, data.shape[0]):
  x = [1, actual["DAYNUM"][day], actual["TMIN"][day], actual["TAVG"][day]]
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