import numpy as np
from sklearn.cluster import KMeans
from bisect import bisect

class Worms:
    def __init__(self, k):
        self.out_dim = k

    def load_data(self, data):
        self.data = data

    def __growth_death(self, worm_idx, direction, alpha1, alpha2):
        init_cost = self.loss()

        snapshot = self.clusters.copy()

        worm_length = self.__worm_lengths[worm_idx]
        worm_start = self.__worm_cutoffs[worm_idx]
        worm_end = worm_start + worm_length

        new = self.__var * np.random.uniform(low=-1, high=1, size=(1, self.data.shape[1]))

        if direction == 0:
            self.clusters = np.vstack((self.clusters[:worm_start], self.clusters[worm_start] + new, self.clusters[worm_start:]))

        elif direction == 1:
            self.clusters = np.vstack((self.clusters[:worm_end], self.clusters[worm_end - 1] + new, self.clusters[worm_end:]))

        self.__worm_lengths[worm_idx] += 1
        self.__worm_cutoffs[worm_idx + 1:] += 1

        new_cost = self.loss()

        if new_cost / init_cost > alpha1:
            self.clusters = snapshot
            self.__worm_lengths[worm_idx] -= 1
            self.__worm_cutoffs[worm_idx + 1:] -= 1

        if worm_length < 2:
            return

        snapshot = self.clusters.copy()
        init_cost = new_cost

        if direction == 0:
            self.clusters = np.vstack((self.clusters[:worm_start], self.clusters[worm_start + 1:]))

        elif direction == 1:
            self.clusters = np.vstack((self.clusters[:worm_end - 1], self.clusters[worm_end:]))

        self.__worm_lengths[worm_idx] -= 1
        self.__worm_cutoffs[worm_idx + 1:] -= 1

        new_cost = self.loss()

        if new_cost / init_cost > alpha2:
            self.clusters = snapshot
            self.__worm_lengths[worm_idx] += 1
            self.__worm_cutoffs[worm_idx + 1:] += 1

    def learn(self, iterations, epochs, lam, mu, lr, alpha1=.75, alpha2=.25):
        self.lam = lam
        self.mu = mu
        self.__var = 1e-4
        self.__worm_cutoffs = np.arange(self.out_dim)
        self.__worm_lengths = np.ones(self.out_dim, dtype=np.int32)

        kmeans = KMeans(self.out_dim).fit(self.data)

        self.clusters = kmeans.cluster_centers_

        for _ in range(iterations):
            for _ in range(epochs):
                x = self.data[np.random.randint(self.data.shape[0])]

                diff = x - self.clusters
                dist = np.einsum('ij,ij->i', diff, diff)

                winner_idx = np.argmin(dist)

                winner_worm_idx = bisect(self.__worm_cutoffs, winner_idx) - 1
                winner_worm_length = self.__worm_lengths[winner_worm_idx]
                winner_worm_start = self.__worm_cutoffs[winner_worm_idx]
                winner_worm_end = winner_worm_start + winner_worm_length

                segments = self.clusters[winner_worm_start + 1 : winner_worm_end] - self.clusters[winner_worm_start : winner_worm_end - 1]

                self.clusters[winner_idx] += lr * diff[winner_idx]

                if winner_worm_length < 2:
                    continue

                segments *= (self.lam + self.mu / 2) * lr
                self.clusters[winner_worm_start : winner_worm_end - 1] += segments
                self.clusters[winner_worm_start + 1 : winner_worm_end] -= segments

                if winner_worm_length < 3:
                    continue

                segments /= (1 + 2 * self.lam / self.mu)
                self.clusters[winner_worm_start : winner_worm_end - 2] -= segments[1:]
                self.clusters[winner_worm_start + 2 : winner_worm_end] += segments[:-1]

            for k in range(self.out_dim):
                self.__growth_death(k, 0, alpha1, alpha2)
                self.__growth_death(k, 1, alpha1, alpha2)

        self.clusters = [self.clusters[self.__worm_cutoffs[k] : self.__worm_cutoffs[k] + self.__worm_lengths[k]] for k in range(self.out_dim)]

    def loss(self):
        x = self.data[np.random.randint(self.data.shape[0])]

        diff = x - self.clusters
        dist = np.einsum('ij,ij->i', diff, diff)

        winner_idx = np.argmin(dist)

        winner_worm_idx = bisect(self.__worm_cutoffs, winner_idx) - 1

        winner_worm_length = self.__worm_lengths[winner_worm_idx]
        winner_worm_start = self.__worm_cutoffs[winner_worm_idx]
        winner_worm_end = winner_worm_start + winner_worm_length

        mse = dist[winner_idx]

        segments = self.clusters[winner_worm_start + 1 : winner_worm_end] - self.clusters[winner_worm_start : winner_worm_end - 1]

        closeness_error = self.lam * np.sum(np.einsum('ij,ij->i', segments, segments))

        smoothness_error = -self.mu * np.sum(np.einsum('ij,ij->i', segments[1:], segments[:-1]))

        return mse + closeness_error + smoothness_error

    def predict(self, x):
        diffs = [x - worm[:, np.newaxis] for worm in self.clusters]
        dists = [np.einsum('ijk,ijk->ij', df, df) for df in diffs]

        min_dists = np.array([np.min(dt, axis=0) for dt in dists])
        return np.argmin(min_dists, axis=0)
