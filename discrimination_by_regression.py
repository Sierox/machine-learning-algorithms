import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(421)

# Load data.
X = np.genfromtxt("data/letters_data_set_images.csv", delimiter=",")
y = np.genfromtxt("data/letters_data_set_labels.csv", delimiter=",", dtype='U')
N = X.shape[0]
K = np.unique(y).shape[0]
x_H, x_W = (20, 16)

# Convert labels to integers.
label_to_ints = {'"A"': 1, '"B"': 2, '"C"': 3, '"D"': 4, '"E"': 5}
y_num = np.array([label_to_ints.get(i) for i in y])

# Divide data into training and testing sets.
N_train, N_test = (25 * K, 14 * K)
N_c, N_train_c, N_test_c = (N / K, N_train / K, N_test / K)
X_train = np.array([(X[int(N_c * c):int((N_c * c) + N_train_c), :])
                    for c in range(K)]).reshape((N_train, x_H * x_W))
y_train = np.array([(y_num[int(N_c * c):int((N_c * c) + N_train_c)])
                    for c in range(K)]).reshape(N_train)
X_test = np.array([(X[int((N_c * c) + N_train_c):int((N_c * c) + N_train_c + N_test_c), :])
                   for c in range(K)]).reshape((N_test, x_H * x_W))
y_test = np.array([(y_num[int((N_c * c) + N_train_c):int((N_c * c) + N_train_c + N_test_c)])
                   for c in range(K)]).reshape(N_test)

# One-hot encode the labels.
y_train_oh = np.zeros((N_train, K)).astype(int)
y_train_oh[range(N_train), y_train - 1] = 1
y_test_oh = np.zeros((N_test, K)).astype(int)
y_test_oh[range(N_test), y_test - 1] = 1


# Function to plot X, X_train, or X_test for debugging.
def plot_data_set(X_plt):
    cols, rows = (X_plt.shape[0] / K, K)
    fig = plt.figure(figsize=(cols, rows))
    for i in range(int(cols * rows)):
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(X_plt[i].reshape((16, 20)).T, cmap='gray')
        plt.axis('off')
    plt.show()
# plot_data_set(X)
# plot_data_set(X_train)
# plot_data_set(X_test)


def sigmoid(X, W, w0):
    s = np.matmul(np.hstack((X, np.ones((X.shape[0], 1)))), np.vstack((W, w0)))
    return 1 / (1 + np.exp(-s))


def gradient_W(X, y, y_pred):
    return -np.asarray([np.sum(np.repeat(((y[:, c] - y_pred[:, c]) * y_pred[:, c] * (1 - y_pred[:, c]))[:, None],
                                         X.shape[1], axis=1) * X, axis=0) for c in range(K)]).T


def gradient_w0(y, y_pred):
    return -np.sum((y - y_pred) * y_pred * (1 - y_pred), axis=0)


# Learning hyperparameters.
eta = 0.01
epsilon = 1e-3

# Random initialization of W and w0.
W = np.random.uniform(low=-0.01, high=0.01, size=(X.shape[1], K))
w0 = np.random.uniform(low=-0.01, high=0.01, size=(1, K))

# Learning.
iteration = 1
objective_values = []
while 1:
    y_predicted_train = sigmoid(X_train, W, w0)
    objective_values = np.append(objective_values, np.sum(0.5 * ((y_train_oh - y_predicted_train)**2)))
    W_old = W
    w0_old = w0
    W = W - eta * gradient_W(X_train, y_train_oh, y_predicted_train)
    w0 = w0 - eta * gradient_w0(y_train_oh, y_predicted_train)
    if np.sqrt(np.sum((w0 - w0_old)) ** 2 + np.sum((W - W_old) ** 2)) < epsilon:
        break
    iteration += 1

# Plots and confusion matrices.
plt.figure(figsize=(10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

y_predicted_train = np.argmax(y_predicted_train, axis=1) + 1
confusion_matrix_train = pd.crosstab(y_predicted_train, y_train, rownames=['y_pred_train'], colnames=['y_truth'])
print(confusion_matrix_train)

# Apply trained model on testing set.
y_predicted_test = sigmoid(X_test, W, w0)
y_predicted_test = np.argmax(y_predicted_test, axis=1) + 1
confusion_matrix_test = pd.crosstab(y_predicted_test, y_test, rownames=['y_pred_test'], colnames=['y_truth'])
print(confusion_matrix_test)