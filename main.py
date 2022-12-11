import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cluster import KMeans


def optimise(x, y, m1, b1, m2, b2):
    loss = 0
    lr = 0.0001
    grad_m1 = 0
    grad_b1 = 0
    grad_m2 = 0
    grad_b2 = 0

    for i in range(len(all_x)):
        dist1 = m1*x[i] - y[i] + b1
        dist2 = m2*x[i] - y[i] + b2

        loss_dist1 = dist1 ** 2
        loss_dist2 = dist2 ** 2

        impact1 = 0.8
        impact2 = 0.2

        if loss_dist1 > loss_dist2:
            impact1 = 0.2
            impact2 = 0.8

        grad_dist1 = 2 * dist1 * impact1
        grad_m1_for_this = grad_dist1 * x[i]
        grad_b1_for_this = grad_dist1
        grad_m1 = grad_m1 + grad_m1_for_this
        grad_b1 = grad_b1 + grad_b1_for_this

        grad_dist2 = 2 * dist2 * impact2
        grad_m2_for_this = grad_dist2 * x[i]
        grad_b2_for_this = grad_dist2
        grad_m2 = grad_m2 + grad_m2_for_this
        grad_b2 = grad_b2 + grad_b2_for_this

        loss = loss + loss_dist1 + loss_dist2

    descent_grad_m1 = lr * grad_m1
    descent_grad_b1 = lr * grad_b1
    descent_grad_m2 = lr * grad_m2
    descent_grad_b2 = lr * grad_b2

    m1 = m1 - descent_grad_m1
    b1 = b1 - descent_grad_b1
    m2 = m2 - descent_grad_m2
    b2 = b2 - descent_grad_b2

    return loss, m1, b1, m2, b2


def fit_lines(all_x, all_y, ax1, ax2):
    m1 = np.random.random()
    b1 = np.random.random()
    m2 = -m1
    b2 = np.random.random()

    x = np.linspace(0, 1, 100)
    y1 = m1 * x + b1
    y2 = m2 * x + b2

    ax1.scatter(all_x,all_y, s=5)
    ax1.scatter(x,y1, s=5, c="purple")
    ax1.scatter(x,y2, s=5, c="red")

    for i in range(10000):
        loss, m1, b1, m2, b2 = optimise(all_x, all_y,m1, b1, m2, b2)
        print(i,":",loss)

    x = np.linspace(0, 1, 100)
    y1 = m1 * x + b1
    y2 = m2 * x + b2

    ax2.scatter(all_x,all_y, s=5)
    ax2.scatter(x,y1, s=5, c="purple")
    ax2.scatter(x,y2, s=5, c="red")

np.random.seed(2)

m1 = np.random.random()
b1 = np.random.random()

x = np.linspace(0,1,100)
y = m1 * x + b1

ax = plt.subplot(2,2,1)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(True)
ax.scatter(x, y, s=5)

m2 = np.random.random()
b2 = np.random.random()

x = np.linspace(0,1,100)
y = m2 * x + b2

ax.scatter(x, y, s=5)

x1 = np.linspace(0,1,100)
noise = (np.random.random(100) - 0.5) * 0.05
y1 = m1 * x + b1 + noise

ax.scatter(x1, y1, s=5)


x2 = np.linspace(0,1,100)
noise = (np.random.random(100) - 0.5) * 0.05
y2 = m2 * x + b2 + noise

ax.scatter(x2, y2, s=5)

all_x = np.concatenate((x1, x2), axis=0)
all_y = np.concatenate((y1, y2), axis=0)

ax = plt.subplot(2,2,2)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(True)

ax.scatter(all_x, all_y, s=5)
ml_x = all_x.reshape(-1,1)
reg = LinearRegression().fit(ml_x,all_y)

fit_m = reg.coef_[0]
fit_c = reg.intercept_

x = np.linspace(0,1,100)
ml_x = x.reshape(-1,1)
y = reg.predict(ml_x)

ax.scatter(ml_x, y, s=5)

all_x2 = all_x.reshape(-1,1)
all_y2 = all_y.reshape(-1,1)
#all_z = all_x2* fit_m - all_y2 + fit_c
#all_x2 = np.concatenate((all_x2, all_z), axis=1)

kmeans = KMeans(n_clusters=2, init="k-means++")
kmeans = kmeans.fit(all_x2, all_y)

colours = ["yellow", "purple"]
c = [colours[i] for i in kmeans.labels_]

ax1 = plt.subplot(2,2,3)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.grid(True)

#ax.scatter(all_x, all_y, s=5, c=c)

ax2 = plt.subplot(2,2,4)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.grid(True)

fit_lines(all_x, all_y, ax1, ax2)

plt.show()



