import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import time
from cycler import cycler

from worms import Worms

X, y = load_iris(return_X_y=True)

np.random.seed(1)

net = Worms(k=3)
net.load_data(X)

start = time.time()
net.learn(iterations=100, epochs=100, lam=1.3, mu=1.1, lr=1e-1)
end = time.time()

print(f"Elapsed time : {end - start}")

num_features = X.shape[1]

fig, axs = plt.subplots(num_features - 1, num_features - 1, figsize=(8, 8))

custom_cycler = (cycler(color=['c', 'm', 'y', 'k']) +
                 cycler(lw=[4, 4, 4, 4]))

for i in range(1, num_features):
    for j in range(num_features - 1):
        if i <= j:
            axs[i - 1, j].axis('off')
        else:
            for target in set(y):
                axs[i - 1, j].scatter(X[y == target, i], X[y == target, j], label=f'Class {target}')

            axs[i-1, j].set_prop_cycle(custom_cycler)
            for k in range(net.out_dim):
                axs[i - 1, j].plot(net.clusters[k][:, i], net.clusters[k][:, j], linewidth=5)

plt.tight_layout()

plt.show()

acc = 0
for i in range(10):
    net.learn(iterations=100, epochs=100, lam=1.3, mu=1.1, lr=1e-1)
    guess = net.predict(X)

    L = np.empty(3)

    for k in range(3):
        unique_values, counts = np.unique(guess[y == k], return_counts=True)
        max_count_index = np.argmax(counts)
        most_frequent_value = unique_values[max_count_index]
        L[k] = most_frequent_value

    acc += np.mean(guess == L[y]) / 10

print(f"Accuracy : {acc}")
