import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")  # Ignore insignificant warnings for cleaner output.

# (2)

# Load data.
data_set = np.genfromtxt("hw04_data_set.csv", delimiter=",", skip_header=True)
N_train, N_test = (100, 33)
train_set = data_set[0:N_train, :]
test_set = data_set[N_train:N_train+N_test, :]

# (3)

# Regressogram model:
minimum_value = 0
maximum_value = 60
bin_width = 3
grid_density = 1/bin_width
left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)
reg = np.asarray(
    [np.sum(((left_borders[b] < train_set[:, 0]) & (train_set[:, 0] <= right_borders[b])) * train_set[:, 1]) /
     np.sum((left_borders[b] < train_set[:, 0]) & (train_set[:, 0] <= right_borders[b]))
     for b in range(len(left_borders))])

# Plotting the Regressogram.
plt.figure(figsize=(10, 6))
plt.plot(train_set[:, 0], train_set[:, 1], "c.", markersize=10)
plt.plot(test_set[:, 0], test_set[:, 1], "r.", markersize=10)
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [reg[b], reg[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [reg[b], reg[b + 1]], "k-")
plt.show()

# (4)

# Calculating RMSE for Regressogram.
RMSE_reg = np.sqrt(np.sum((test_set[:, 1] - [reg[int(np.floor(x * grid_density))] for x in test_set[:, 0]])**2)/N_test)
print('RMSE for Regressogram (h=%d): %f' % (bin_width, RMSE_reg))

# (5)

# Running Mean Smoother model:
minimum_value = 0
maximum_value = 60
bin_width = 3
grid_density = 100
data_interval = np.linspace(minimum_value, maximum_value, (maximum_value-minimum_value)*grid_density + 1)
rms = np.asarray(
    [np.sum((np.abs(((x - train_set[:, 0])/bin_width)) <= 0.5) * train_set[:, 1]) /
     np.sum((np.abs(((x - train_set[:, 0])/bin_width)) <= 0.5))
     for x in data_interval])

# Plotting the Running Mean Smoother.
plt.figure(figsize=(10, 6))
plt.plot(train_set[:, 0], train_set[:, 1], "c.", markersize=10)
plt.plot(test_set[:, 0], test_set[:, 1], "r.", markersize=10)
plt.plot(data_interval, rms, "k-")
plt.show()

# (6)

# Calculating RMSE for Running Mean Smoother.
RMSE_rms = np.sqrt(np.sum((test_set[:, 1] - [rms[int(np.round(x * grid_density))] for x in test_set[:, 0]])**2)/N_test)
print('RMSE for Running Mean Smoother (h=%d): %f' % (bin_width, RMSE_rms))

# (7)

# Kernel Smoother model:
minimum_value = 0
maximum_value = 60
bin_width = 1
grid_density = 100
data_interval = np.linspace(minimum_value, maximum_value, (maximum_value-minimum_value)*grid_density + 1)
ker = np.asarray(
    [np.sum((1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - train_set[:, 0])/bin_width)**2) * train_set[:, 1]) /
     np.sum((1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - train_set[:, 0])/bin_width)**2))
     for x in data_interval])

# Plotting the Kernel Smoother.
plt.figure(figsize=(10, 6))
plt.plot(train_set[:, 0], train_set[:, 1], "c.", markersize=10)
plt.plot(test_set[:, 0], test_set[:, 1], "r.", markersize=10)
plt.plot(data_interval, ker, "k-")
plt.show()

# (8)

# Calculating RMSE for Kernel Smoother.
RMSE_ker = np.sqrt(np.sum((test_set[:, 1] - [ker[int(np.round(x * grid_density))] for x in test_set[:, 0]])**2)/N_test)
print('RMSE for Kernel Smoother (h=%d): %f' % (bin_width, RMSE_ker))
