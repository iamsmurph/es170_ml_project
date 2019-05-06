from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

x_values = []
y_values = []
z_values = []
with open('results.csv') as csv_file:
       lis = [line.split() for line in csv_file]
       for i, value in enumerate(lis):
              if i != 0:
                    z_values.append(float(value[0].split(",")[2])), y_values.append(float(value[0].split(",")[1])), x_values.append(float(value[0].split(",")[0]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_val = 0
with open('results.csv') as csv_file:
       lis = [line.split() for line in csv_file]
       for i in range(len(lis)):
              x_val = 1 + x_val
n = x_val

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
xs = np.asarray(x_values)
ys = np.asarray(y_values)
zs = np.asarray(z_values)
ax.scatter(xs, ys, zs, c='r', marker='o')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Label')

plt.show()
