from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

class Visual:

  def coloured_scatter(self, results):
    # take in results of type np array
    # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    pal = {"class1" : "#3498db", "class2" : "#e74c3c", "noclass" : "#2ecc71"}
    res12 = results[:,0:2]
    res3 = results[:,2]

    def to_colour(a):
      if a == 0:
        return "class1"
      elif a==1:
        return "class2"
      else: return "noclass"

    #convert to panda
    df  = pd.DataFrame(res12, columns = ['x', 'y'])
    df["label"] = pd.Series(res3).apply(to_colour)

    sns.scatterplot(df["x"], df["y"], hue=df["label"], palette = pal)
    plt.show()
    return ()

  def to_file(self, array):
    c = np.savetxt("results.txt", array)
    return ()

  def from_file(self):
    return np.loadtxt("results.txt")

  def dddplot(self, data, target):
    def make_third(b):
      if b == 0:
        return -10.0
      elif b == 1:
        return 10.0
      else: return 0.0

    xs = data[:,0]
    ys = data[:,1]
    zs = data[:,2].tolist()

    c = list(map(make_third, zs))
    zs = np.asarray(c)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs, c=target, cmap = plt.cm.tab10, marker='o')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Class')

    plt.show()
    return ()


'''
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
'''
