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

inputs = np.array(inputs_arr)
false_data = inputs[inputs[:,2] == 0]
true_data = inputs[inputs[:,2] == 1]

# Plot
plt.plot(false_data[:,0], false_data[:,1], marker='o', markersize=2, linestyle='', color='red')
plt.plot(true_data[:,0], true_data[:,1], marker='o', markersize=2, linestyle='', color='blue')
plt.show()