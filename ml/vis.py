from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

class Visual:

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

    return ()
