import csv
import seaborn as sns; sns.set()

toCSV = [{'x':1.0,'y':2.0,'label':1.0}, {'x':2.0,'y':3.0,'label':0.0}, {'x':4.0, 'y':6.0, 'label':1.0}]

with open('results.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(('x', 'y', 'label'))
    for i in range(len(toCSV)):
        writer.writerow([toCSV[i].get('x'), toCSV[i].get('y'), toCSV[i].get('label')])

import pandas as pd
import matplotlib.pyplot as plt
df  = pd.read_csv("results.csv")
# df.plot()  # plots all columns against index
sns.relplot(x="x", y="y", hue="label", data=df);
# df.plot(kind='density')  # estimate density function
plt.show()

# Iris visualization before any of our work is done

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# plot the first three dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three dimensions ")
ax.set_xlabel("1st dimension")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd dimension")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd dimension")
ax.w_zaxis.set_ticklabels([])

plt.show()
