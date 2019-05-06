import seaborn as sns
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas
sns.set(style="darkgrid")

# training set must contain tuples: (vector, class)
  training_set = [
      ([0, 1], 0), # class 0 training vector
      ([0.78861006, 0.61489363], 1) # class 1 training vector
  ]

# Load test vectors into testvectors list. Each test vector is a two dimensional
# list at this point.
testvectors = []

# Obtain the label/classificaiton for the ith test vector and make the two dimensional vector for
# the ith test vector a three dimensional one, where the 3rd element is the label.

for i in range(len(testvectors)):
  class_result = PQ_classifier.classify(test_vector= testvectors[i], training_set=training_set)
  testvectors[i].append(class_result)

with open('results.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(('x', 'y', 'label'))
    for i in range(len(toCSV)):
        writer.writerow([testvectors[i].get('x'), testvectors[i].get('y'), testvectors[i].get('label')])

rst = pandas.read_csv("results.csv")
sns.relplot(x="x", y="y", hue="label", data=rst);
plt.show()
