import numpy as np
import matplotlib.pyplot as plt
import time
import umap
from datasets import load_dataset

# Load the MNIST dataset
image_dataset = load_dataset('mnist')

# Prepare the data
X = np.array(image_dataset['test']['image']).reshape(-1, 28*28)
y = np.array(image_dataset['test']['label'])

# Randomly select 5000 samples
ridx = np.random.randint(X.shape[0], size=5000)
X = X[ridx]
y = y[ridx]

# Reduce dimensionality to 2 components using UMAP
X = umap.UMAP(n_components=2).fit_transform(X)

print("done")

# Initialize the Worms network
np.random.seed()
from worms import Worms  # Ensure that the 'worms' package is installed and available
net = Worms(k=10)
net.data = X

# Train the network
start = time.time()
net.learn(iterations=100, epochs=200, lam=2.5, mu=2.2, lr=1e-1)
end = time.time()

# Plot the results in 2D
for target in np.unique(y):
    plt.scatter(X[(y == target), 0], X[(y == target), 1], label=f'Class {target}')
    cluster = np.array(net.clusters[target])
    plt.plot(cluster[:, 0], cluster[:, 1], linewidth=5, color='black')

plt.legend()
plt.show()

print(f"Elapsed time: {end - start}")
