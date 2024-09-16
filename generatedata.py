import csv
import math
import numpy as np

ARRAY_LEN = 1000
X_SPREAD = 100

# Separation line parameters
SEP_SLOPE = -0.5
SEP_YINTERCEPT = 75
SEP_BIAS = 10
SEP_STDEV = 10

# Data labeled false are on average below the separation line
x1_false = np.random.rand(math.ceil(ARRAY_LEN / 2)) * X_SPREAD
x2_false = x1_false * SEP_SLOPE + SEP_YINTERCEPT \
    - SEP_BIAS + np.random.normal(size=x1_false.shape) * SEP_STDEV

# Data labeled true are on average above the separation line
x1_true = np.random.rand(math.floor(ARRAY_LEN // 2)) * X_SPREAD
x2_true = x1_true * SEP_SLOPE + SEP_YINTERCEPT \
    + SEP_BIAS + np.random.normal(size=x1_true.shape) * SEP_STDEV

def combine_data(x1, x2, labels):
    ''' Combines x1, x2, and labels into one n x 3 matrix. '''
    return np.concatenate([x1, x2, labels]).reshape((3, len(x1))).transpose()

# Combine all data
false_data = combine_data(x1_false, x2_false, labels=np.zeros(x1_false.shape))
true_data = combine_data(x1_true, x2_true, labels=np.ones(x1_true.shape))
all_data = np.concatenate([false_data, true_data])
np.random.shuffle(all_data)

# Write all to file
with open('data.csv', 'w', newline='') as file:
    csvwriter = csv.DictWriter(file, fieldnames=['x1', 'x2', 'label'])
    csvwriter.writeheader()
    for (x1, x2, label) in all_data:
        csvwriter.writerow({'x1':x1, 'x2':x2, 'label':label})