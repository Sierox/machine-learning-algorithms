import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")  # Ignore insignificant warnings for cleaner output.

np.random.seed(421)  # Set random seed.

# (1)
class_means = np.array([[2.5, 2.5], [-2.5, 2.5], [-2.5, -2.5], [2.5, -2.5], [0.0, 0.0]])
class_covariances = np.array([[[0.8, -0.6], [-0.6, 0.8]],
                              [[0.8, 0.6], [0.6, 0.8]],
                              [[0.8, -0.6], [-0.6, 0.8]],
                              [[0.8, 0.6], [0.6, 0.8]],
                              [[1.6, 0.0], [0.0, 1.6]]])
class_sizes = np.array([50, 50, 50, 50, 100])

points1 = np.random.multivariate_normal(class_means[0], class_covariances[0], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1], class_covariances[1], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2], class_covariances[2], class_sizes[2])
points4 = np.random.multivariate_normal(class_means[3], class_covariances[3], class_sizes[3])
points5 = np.random.multivariate_normal(class_means[4], class_covariances[4], class_sizes[4])

X = np.concatenate((points1, points2, points3, points4, points5))
K = 5
N = X.shape[0]

plt.figure(figsize=(5, 5))
plt.plot(X.T[0], X.T[1], "ko")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# Methods for k-means clustering algorithm:
def k_means_clustering_init():
    centroids = X[np.random.choice(range(N), K), :]
    memberships = np.argmin(spa.distance_matrix(centroids, X), axis=0)
    return centroids, memberships


def k_means_clustering_iter(I, model):
    centroids, memberships = model
    for i in range(I):
        centroids = np.vstack([np.mean(X[memberships == k, :], axis=0) for k in range(K)])
        memberships = np.argmin(spa.distance_matrix(centroids, X), axis=0)
    return centroids, memberships


# Methods for EM clustering algorithm:
def em_clustering_init(from_model):
    centroids, memberships = from_model
    means = centroids
    covariances = np.array([np.sum([([x] - centroids[k]).T * ([x] - centroids[k]) for x in X[memberships == k]], axis=0)
                            / np.sum(memberships == k) for k in range(K)])
    priors = np.array([np.sum(memberships == k) / N for k in range(K)])
    return means, covariances, priors


def em_clustering_iter(I, model_g):
    means, covariances, priors = model_g
    for i in range(I):
        H = np.array([([stats.multivariate_normal(mean=means[k], cov=covariances[k]).pdf(x) * priors[k]
                        for k in range(K)]) for x in X])
        H = np.array([H[r] / np.sum(H[r]) for r in range(H.shape[0])])

        means = np.array([H[:, i].dot(X) / np.sum(H[:, i]) for i, m in enumerate(means)])
        covariances = np.array([np.sum(np.array(
            [H[j][i] * np.matmul(np.array([X[j] - means[i]]).T, np.array([X[j] - means[i]]))
             for j in range(N)]), axis=0) / np.sum(H[:, i])
                                for i, c in enumerate(covariances)])
        priors = np.array([np.sum(H[:, i] / N) for i, p in enumerate(priors)])
    return means, covariances, priors


def em_clustering_convert(model_g):
    means, covariances, priors = model_g
    H = np.array([([stats.multivariate_normal(mean=means[k], cov=covariances[k]).pdf(x) * priors[k]
                    for k in range(K)]) for x in X])
    H = np.array([H[r] / np.sum(H[r]) for r in range(H.shape[0])])
    centroids = means
    memberships = np.array([np.argmax(h) for h in H])
    return centroids, memberships


# Method for drawing the required plot.
def plot(model_gauss, plot_range=6.0, plot_density=20.0):
    plt.figure(figsize=(5, 5))
    plt.xlabel("x1")
    plt.ylabel("x2")

    centroids, memberships = em_clustering_convert(model_gauss)
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    for c in range(K):
        plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10,
                 color=cluster_colors[c])

    means, covariances, _ = model_gauss
    means = np.vstack((means, class_means))
    covariances = np.vstack((covariances, class_covariances))
    solids = np.hstack((np.zeros(K, dtype=bool), np.ones(K, dtype=bool)))
    for mean, covariance, solid in zip(means, covariances, solids):
        x1_interval = np.arange(-plot_range, plot_range, 1.0 / plot_density)
        x2_interval = np.arange(-plot_range, plot_range, 1.0 / plot_density)
        x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
        gaussian = stats.multivariate_normal(mean, covariance)
        gaussian = np.array([[(1 if gaussian.pdf([i, j]) >= 0.05 else 0) for j in x2_interval] for i in x1_interval])
        plt.contour(x1_grid, x2_grid, gaussian, colors='k', linestyles='solid' if solid else 'dashed', levels=0)
    plt.show()


# (2)
k_means_model = k_means_clustering_init()
k_means_model = k_means_clustering_iter(2, k_means_model)

# (3)
em_model_gauss = em_clustering_init(k_means_model)

# (4)
em_model_gauss = em_clustering_iter(100, em_model_gauss)
print(em_model_gauss[0])

# (5)
plot(em_model_gauss)
