import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import time

X, y = make_moons(n_samples=2048, noise=0.1, random_state=42)

np.random.seed()

from worms import Worms

net = Worms(k=2)
net.data = X

start = time.time()
net.learn(iterations=100, epochs=100, lam=.2, mu=.15, lr=5e-2, alpha1=.75, alpha2=.25)
end = time.time()

print(f"Elapsed time : {end - start}")


plt.scatter(X[:, 0], X[:, 1], color="gray")

for i in range(2):
    plt.plot(net.clusters[i][:, 0], net.clusters[i][:, 1], linewidth=5)

plt.show()
