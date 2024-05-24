import numpy as np
from sklearn.cluster import KMeans

class Worms:
    def __init__(self, k):
        self.out_dim = k

    def load_data(self, data):
        self.data = data

    def __growth_death(self, worm_idx, direction, alpha1=.75, alpha2=.5):
        init_cost = self.loss(self.clusters)

        snapshot = self.clusters.copy()

        new = np.random.multivariate_normal(np.zeros(self.data.shape[1]), self.__var, 1)
        snapshot[worm_idx] = np.vstack((self.clusters[worm_idx][0] + new, snapshot[worm_idx])) if direction == 0 else np.vstack((snapshot[worm_idx], self.clusters[worm_idx][-1] + new))

        new_cost = self.loss(snapshot)

        if new_cost / init_cost < alpha1:
            self.clusters = snapshot

        if self.clusters[worm_idx].shape[0] < 2:
            return

        snapshot = self.clusters.copy()

        snapshot[worm_idx] = snapshot[worm_idx][1:] if direction == 0 else snapshot[worm_idx][:-1]

        new_cost = self.loss(snapshot)

        if new_cost / init_cost < alpha2 * .5:
            self.clusters = snapshot

    def learn(self, iterations, epochs, lam, mu, lr):
        self.lam = lam
        self.mu = mu
        self.__var = 1e-4 * np.eye(self.data.shape[1])

        kmeans = KMeans(self.out_dim).fit(self.data)
        self.clusters = [center.reshape(1, -1) for center in kmeans.cluster_centers_]

        for _ in range(iterations):
            for _ in range(epochs):
                x = self.data[np.random.randint(self.data.shape[0])]

                diff = [x - worm for worm in self.clusters]
                dist = [np.einsum('ij,ij->i', df, df) for df in diff]

                winner_worm_idx = np.argmin([np.min(dist) for dist in dist])
                winner_idx = np.argmin(dist[winner_worm_idx])

                segments = self.clusters[winner_worm_idx][1:] - self.clusters[winner_worm_idx][:-1]

                segments *= (self.lam + self.mu / 2) * lr
                self.clusters[winner_worm_idx][:-1] += segments
                self.clusters[winner_worm_idx][1:] -= segments

                segments /= (1 + 2 * self.lam / self.mu)
                self.clusters[winner_worm_idx][:-2] -= segments[1:]
                self.clusters[winner_worm_idx][2:] += segments[:-1]

                self.clusters[winner_worm_idx][winner_idx] += lr * diff[winner_worm_idx][winner_idx]

            [self.__growth_death(k, 0) for k in range(self.out_dim)]
            [self.__growth_death(k, 1) for k in range(self.out_dim)]

    def loss(self, clusters):
        x = self.data[np.random.randint(self.data.shape[0])]

        diff = [x - worm for worm in self.clusters]
        dist = [np.einsum('ij,ij->i', df, df) for df in diff]

        winner_worm_idx = np.argmin([np.min(dist) for dist in dist])
        winner_idx = np.argmin(dist[winner_worm_idx])

        mse = dist[winner_worm_idx][winner_idx]

        segments = clusters[winner_worm_idx][1:] - clusters[winner_worm_idx][:-1]

        closeness_error = self.lam * np.sum(np.einsum('ij,ij->i', segments, segments))

        smoothness_error = -self.mu * np.sum(np.einsum('ij,ij->i', segments[1:], segments[:-1]))

        return mse + closeness_error + smoothness_error

    def predict(self, x):
        diffs = [x - worm[:, np.newaxis] for worm in self.clusters]
        dists = [np.einsum('ijk,ijk->ij', df, df) for df in diffs]

        min_dists = np.array([np.min(dt, axis=0) for dt in dists])
        return np.argmin(min_dists, axis=0)
