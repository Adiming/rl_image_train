import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

# file_name = "DDPG_test_1e6_gamma09_n" + ".csv"
file_name = "dqn_img_1e6" + ".csv"
plt.rcParams["figure.figsize"] = [7.00, 5.50]
plt.rcParams["figure.autolayout"] = True
columns = ["x", "y","i"]

df = pd.read_csv(file_name, usecols=columns)

# draw the trajectory
plt.plot(df.x, df.y, color = 'g', linestyle = 'dashed',
         marker = 'o',label = "Gear trajectory")

# draw points number
x = df['x']
y = df['y']
for i in range(len(x)):
    plt.text(x[i]*1.001, y[i]*1.001, i, color = "b")

# draw the center point
plt.plot(300, 300, color = 'r',marker = 'o')

# draw the peg circle
r=5.0
a,b=300,300
theta = np.arange(0, 2*np.pi, 0.01)
x = a + r * np.cos(theta)
y = b + r * np.sin(theta)
plt.plot(x, y, 'r')
plt.axis('equal')
# plt.axis('scaled')

plt.show()