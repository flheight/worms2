import numpy as np
from sklearn.cluster import KMeans
from bisect import bisect

class Worms:
    def __init__(self, k):
        self.out_dim = k

    def load_data(self, data):
        self.data = data

    def __grow(self, worm_idx, direction):
        worm_length = self.__worm_lengths[worm_idx]
        worm_start = self.__worm_cutoffs[worm_idx]
        worm_end = worm_start + worm_length

        new = self.__var * np.random.uniform(low=-1, high=1, size=(1, self.data.shape[1]))

        if direction == 0:
            self.clusters = np.insert(self.clusters, worm_start, self.clusters[worm_start] + new, axis=0)
        elif direction == 1:
            self.clusters = np.insert(self.clusters, worm_end, self.clusters[worm_end - 1] + new, axis=0)

        self.__worm_lengths[worm_idx] += 1
        self.__worm_cutoffs[worm_idx + 1:] += 1

    def __revert(self, worm_idx, direction):
        worm_length = self.__worm_lengths[worm_idx]
        worm_start = self.__worm_cutoffs[worm_idx]
        worm_end = worm_start + worm_length

        self.clusters = np.delete(self.clusters, worm_start if direction == 0 else worm_end - 1, axis=0)
        self.__worm_lengths[worm_idx] -= 1
        self.__worm_cutoffs[worm_idx + 1:] -= 1

    def learn(self, iterations, epochs, lam, link, lr):
        self.lam = lam
        self.link = link
        self.__magic = link**2 / 4
        self.__var = 1e-3
        self.__worm_cutoffs = np.arange(self.out_dim, dtype=np.int32)
        self.__worm_lengths = np.ones(self.out_dim, dtype=np.int32)

        kmeans = KMeans(self.out_dim).fit(self.data)

        self.clusters = kmeans.cluster_centers_
        prev_mean_error = kmeans.inertia_ / self.data.shape[0]

        for i in range(iterations):
            worm_idx = i % self.out_dim
            direction = self.__worm_lengths[worm_idx] % 2

            self.__grow(worm_idx, direction)

            mean_error = 0

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

                segment_dists = np.einsum('ij,ij->i', segments, segments).reshape(-1, 1)
                segment_lengths = np.sqrt(segment_dists).reshape(-1, 1)

                mse = dist[winner_idx]
                linkage_error = self.lam * (np.sum(segment_dists - self.link * segment_lengths) + self. __magic * winner_worm_length)
                mean_error += (mse + linkage_error) / epochs

                segments *= lr * self.lam * (2 - self.link / segment_lengths)
                self.clusters[winner_worm_start : winner_worm_end - 1] += segments
                self.clusters[winner_worm_start + 1 : winner_worm_end] -= segments

            if mean_error > prev_mean_error:
                self.__revert(worm_idx, direction)
            else:
                prev_mean_error = mean_error

        self.clusters = [self.clusters[self.__worm_cutoffs[k] : self.__worm_cutoffs[k] + self.__worm_lengths[k]] for k in range(self.out_dim)]

    def predict(self, x):
        diffs = [x - worm[:, np.newaxis] for worm in self.clusters]
        dists = [np.einsum('ijk,ijk->ij', df, df) for df in diffs]

        min_dists = np.array([np.min(dt, axis=0) for dt in dists])
        return np.argmin(min_dists, axis=0)
