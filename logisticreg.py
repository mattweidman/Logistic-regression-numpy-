import csv
import matplotlib.pyplot as plt
import numpy as np

inputs_arr = []
with open('data.csv', newline='') as file:
    csvreader = csv.DictReader(file)
    for row in csvreader:
        inputs_arr.append([
            float(row['x1']),
            float(row['x2']),
            float(row['label'])
        ])

inputs = np.array(inputs_arr) # Each row is (x1, x2, label)

# Each row of x is (x1, x2, 1)
x = np.concatenate([inputs[:,:2], np.ones((len(inputs), 1))], axis=1)
y = inputs[:,2]

# Choose initial weights that make predictions somewhat varied
weights = np.ones(3) * 0.01

# Hyperparameters
learning_rate = 0.001
iterations = 200000

for i in range(iterations):
    predictions = 1 / (1 + np.exp(-x.dot(weights)))
    percent_correct = np.average((predictions > 0.5) == y)
    loss = np.average(- y * np.log(predictions) - (1 - y) * np.log(1 - predictions))
    print("loss", loss, "percent_correct", percent_correct)

    gradient = np.average(x * (predictions - y)[:,np.newaxis], axis=0)
    weights -= learning_rate * gradient

print(weights)

# Plot
correct_falses = inputs[np.logical_and((predictions > 0.5) == y, y == 0)]
wrong_falses = inputs[np.logical_and((predictions > 0.5) != y, y == 0)]
correct_trues = inputs[np.logical_and((predictions > 0.5) == y, y == 1)]
wrong_trues = inputs[np.logical_and((predictions > 0.5) != y, y == 1)]

def plot_points(points, color):
    plt.plot(points[:,0], points[:,1],
             marker='o', markersize=2, linestyle='', color=color)

LIGHT_GREEN = '#04f000'
LIGHT_RED = '#f00000'
DARK_GREEN = '#007d02'
DARK_RED = '#8f0000'

plot_points(correct_falses, LIGHT_GREEN)
plot_points(wrong_falses, LIGHT_RED)
plot_points(correct_trues, DARK_GREEN)
plot_points(wrong_trues, DARK_RED)
plt.show()