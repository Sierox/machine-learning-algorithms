import matplotlib.pyplot as plt
import numpy as np

# (2)
data_set = np.genfromtxt("data/tree_data_set.csv", delimiter=",", skip_header=True)
N_train, N_test = (100, 33)
train_set = data_set[0:N_train, :]
test_set = data_set[N_train:N_train+N_test, :]


def tree_(train_set, P):
    """Trains and returns a decision tree over the given data set with the given pre-pruning size, P.

    :param train_set: Data set which the decision tree will be trained over.
    :param P: Pre-pruning size.
    :return: The trained decision tree.
    """
    node_indices = {}
    is_terminal = {}    # <- tree[0]
    need_split = {}
    node_means = {}     # <- tree[1]
    node_splits = {}    # <- tree[2]

    node_indices[1] = np.array(range(train_set.shape[0]))
    is_terminal[1] = False
    need_split[1] = True

    while True:
        split_nodes = [key for key, value in need_split.items() if value is True]
        if len(split_nodes) == 0:
            break
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            mean = np.mean(train_set[data_indices, 1])
            if len(train_set[data_indices, 0]) <= P:
                is_terminal[split_node] = True
                node_means[split_node] = mean
            else:
                is_terminal[split_node] = False
                unique_values = np.sort(np.unique(train_set[data_indices, 0]))
                split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2
                split_scores = np.repeat(0.0, len(split_positions))
                for s in range(len(split_positions)):
                    left_indices = data_indices[train_set[data_indices, 0] < split_positions[s]]
                    right_indices = data_indices[train_set[data_indices, 0] >= split_positions[s]]
                    split_scores[s] = (np.sum((train_set[left_indices, 1] - np.mean(train_set[left_indices, 1])) ** 2) +
                                       np.sum((train_set[right_indices, 1] - np.mean(train_set[right_indices, 1])) ** 2)
                                       ) / (len(left_indices) + len(right_indices))
                node_splits[split_node] = split_positions[np.argmin(split_scores)]

                left_indices = data_indices[train_set[data_indices, 0] < split_positions[np.argmin(split_scores)]]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True

                right_indices = data_indices[train_set[data_indices, 0] >= split_positions[np.argmin(split_scores)]]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True

    return is_terminal, node_means, node_splits


def predict_(tree, X):
    """Predicts the y values of the given x's in vector X using the given decision tree.

    :param tree: Decision tree to do the predictions.
    :param X: Vector containing x values whose y values are to be predicted.
    :return: Vector containing the predicted y values.
    """
    is_terminal, node_means, node_splits = tree
    y_pred = np.repeat(0, len(X))
    for n in range(len(X)):
        i = 1
        while True:
            if is_terminal[i] is True:
                y_pred[n] = node_means[i]
                break
            else:
                if X[n] <= node_splits[i]:
                    i = i * 2
                else:
                    i = i * 2 + 1
    return y_pred


def plot_(tree, grid_density=100):
    """Plots a given decision tree by predicting every value between 0 and 60 with 1/grid_density increments.

    :param tree: Decision tree to be plotted.
    :param grid_density: How much each integer unit is divided. Higher values lead to a more accurate plot of the tree.
    :return: void; displays the plot.
    """
    minimum_value, maximum_value = (0, 60)
    data_interval = np.linspace(minimum_value, maximum_value, (maximum_value - minimum_value) * grid_density)
    tree_plot = predict_(tree, data_interval)
    plt.figure(figsize=(10, 6))
    plt.plot(train_set[:, 0], train_set[:, 1], "c.", markersize=10, label="training")
    plt.plot(test_set[:, 0], test_set[:, 1], "r.", markersize=10, label="test")
    left_borders = np.arange(minimum_value, maximum_value, 1.0/grid_density)
    right_borders = np.arange(minimum_value + 1.0/grid_density, maximum_value + 1.0/grid_density, 1.0/grid_density)
    for b in range(len(left_borders)):
        plt.plot([left_borders[b], right_borders[b]], [tree_plot[b], tree_plot[b]], "k-")
    for b in range(len(left_borders) - 1):
        plt.plot([right_borders[b], right_borders[b]], [tree_plot[b], tree_plot[b + 1]], "k-")
    plt.title("P = 15")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.show()


def rmse_(tree, test_set):
    """Computes the RMSE of the given decision tree on the given data set.

    :param tree: Decision tree to be tested.
    :param test_set: Data set to test the tree over.
    :return: Computed RMSE (float).
    """
    return np.sqrt(np.sum((test_set[:, 1] - predict_(tree, test_set[:, 0])) ** 2) / test_set.shape[0])


# (3, 4)
tree15 = tree_(train_set, 15)
plot_(tree15)

# (5)
print("RMSE for Decision Tree (P = 15): %f" % rmse_(tree15, test_set))

# (6)
results = list()
for p in range(5, 55, 5):
    results.append((p, rmse_(tree_(train_set, p), test_set)))
results = np.array(results).T

plt.figure(figsize=(10, 6))
plt.plot(results[0], results[1], "k.-", markersize=10)
plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.show()
