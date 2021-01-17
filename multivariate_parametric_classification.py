import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# I completed this HW mostly by editing the code from the labs. I hope this doesn't count as plagiarism.

np.random.seed(421)

# mean parameters
class_means = np.array([[0.0, 2.5], [-2.5, -2.0], [2.5, -2.0]])
# standard deviation parameters
class_covariances = np.array([[[3.2, 0.0], [0.0, 1.2]],
                              [[1.2, -0.8], [-0.8, 1.2]],
                              [[1.2, 0.8], [0.8, 1.2]]])
# sample sizes
class_sizes = np.array([120, 90, 90])

# generate random samples
points1 = np.random.multivariate_normal(class_means[0], class_covariances[0], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1], class_covariances[1], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2], class_covariances[2], class_sizes[2])

# generate X, y, K, N
X = np.concatenate((points1, points2, points3))
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))
K = np.max(y)
N = X.shape[0]

# plot the generated points
plt.figure(figsize=(5, 5))
plt.plot(points1.T[0], points1.T[1], "r.")
plt.plot(points2.T[0], points2.T[1], "g.")
plt.plot(points3.T[0], points3.T[1], "b.")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


sample_means = np.array([np.mean(X[y == c + 1], axis=0) for c in range(K)])

sample_covariances = np.array([np.cov(X[y == c + 1].T) for c in range(K)])

# My unsuccessful attempt attempt at computing SCM without np.cov().
# sample_covariances = [np.mean(np.matmul
#    ((X[y == c+1] - sample_means[c]), (X[y == c+1] - sample_means[c]).T), axis=0) for c in range(K)]

class_priors = [np.mean(y == c + 1, axis=0) for c in range(K)]

W = np.array([(-0.5 * np.linalg.inv(sample_covariances[c])) for c in range(K)])
w = np.array([(np.linalg.inv(sample_covariances[c]) @ sample_means[c]) for c in range(K)])
w0 = np.array([(-0.5 * (sample_means[c].T @ np.linalg.inv(sample_covariances[c]) @ sample_means[c])
               - 0.5 * np.log(np.linalg.det(sample_covariances[c])) + np.log(class_priors[c])) for c in range(K)])

Y_preds = []
for i in range(N):
    Y_preds.append([X[i].T @ W[c] @ X[i] + w[c].T @ X[i] + w0[c] for c in range(K)])

y_preds = np.argmax(Y_preds, axis=1)+1
confusion_matrix = pd.crosstab(y_preds, y, rownames=['y_predicted'], colnames=['y_truth'])
print(confusion_matrix)

# evaluate discriminant function on a grid
x1_interval = np.linspace(-6, +6, N)
x2_interval = np.linspace(-6, +6, N)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))

for c in range(K):
    discriminant_values[:, :, c] = W[c, 0, 0] * x1_grid**2 + W[c, 0, 1] * x1_grid * x2_grid +\
                                   W[c, 1, 0] * x1_grid * x2_grid + W[c, 1, 1] * x2_grid**2 +\
                                   w[c, 0] * x1_grid + w[c, 1] * x2_grid + w0[c]

lines = [np.array(discriminant_values[:, :, i]) for i in range(3)]
lines[0][(lines[0] < lines[1]) & (lines[0] < lines[2])] = np.nan
lines[1][(lines[1] < lines[0]) & (lines[1] < lines[2])] = np.nan
lines[2][(lines[2] < lines[0]) & (lines[2] < lines[1])] = np.nan

areas = [np.array(discriminant_values[:, :, i]) for i in range(3)]
areas[0][(areas[0] > areas[1]) & (areas[0] > areas[2])] = np.nan
areas[1][(areas[1] > areas[0]) & (areas[1] > areas[2])] = np.nan
areas[2][(areas[2] > areas[0]) & (areas[2] > areas[1])] = np.nan

plt.figure(figsize=(5, 5))
plt.plot(X[y == 1, 0], X[y == 1, 1], "r.", markersize=10)
plt.plot(X[y == 2, 0], X[y == 2, 1], "g.", markersize=10)
plt.plot(X[y == 3, 0], X[y == 3, 1], "b.", markersize=10)
plt.plot(X[y_preds != y, 0], X[y_preds != y, 1], "ko", markersize=12, fillstyle="none")

plt.contour(x1_grid, x2_grid, lines[0] - lines[1], levels=0, colors="k")
plt.contour(x1_grid, x2_grid, lines[0] - lines[2], levels=0, colors="k")
plt.contour(x1_grid, x2_grid, lines[1] - lines[2], levels=0, colors="k")

plt.contourf(x1_grid, x2_grid, areas[0] - areas[1], levels=0, colors="b", alpha=.25)
plt.contourf(x1_grid, x2_grid, areas[0] - areas[2], levels=0, colors="g", alpha=.25)
plt.contourf(x1_grid, x2_grid, areas[1] - areas[2], levels=0, colors="r", alpha=.25)

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
