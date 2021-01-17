import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (2)

# Load data.
X = np.genfromtxt("data/letters_data_set_images.csv", delimiter=",")
y = np.genfromtxt("data/letters_data_set_labels.csv", delimiter=",", dtype='U')
N = X.shape[0]
K = np.unique(y).shape[0]
x_H, x_W = (20, 16)

# Convert labels to integers.
label_to_ints = {'"A"': 1, '"B"': 2, '"C"': 3, '"D"': 4, '"E"': 5}
y_num = np.array([label_to_ints.get(i) for i in y])

# (3)

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

# (4)

# Estimate pcd's and prior probabilities for each class.
pcd = np.array([np.mean(X_train[y_train == (c + 1)], axis=0) for c in range(K)])
prior = [np.mean(y_train == c + 1, axis=0) for c in range(K)]

# (5)

# Function to plot data points as images.
def plot_data_set(X_plt):
    rows, cols = (X_plt.shape[0] / K, K)
    fig = plt.figure(figsize=(cols, rows))
    for i in range(int(cols * rows)):
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(X_plt[i].reshape((x_W, x_H)).T, cmap='Greys')
        plt.axis('off')
    plt.show()


# Plot the pcd's.
plot_data_set(pcd)

# (6, 7)

# Safe log.
def slog(x):
    return np.log(x + 1e-100)


# Naive Bayes' Classifier function for class prediction.
def predict_class(X):
    prob = np.empty((0, K), float)
    for x in X:
        prob = np.vstack((prob, ([np.sum((x * slog(pcd[c])) + ((1 - x) * slog(1 - pcd[c])))
                          + slog(prior[c]) for c in range(K)])))
    return np.argmax(prob, axis=1)+1


# Predictions and confusion matrices.
confusion_matrix_train = pd.crosstab(predict_class(X_train), y_train, rownames=['y_predicted'], colnames=['y_truth'])
confusion_matrix_test = pd.crosstab(predict_class(X_test), y_test, rownames=['y_predicted'], colnames=['y_truth'])
print(confusion_matrix_train)
print(confusion_matrix_test)
