import numpy as np


class SyntheticData:
    """"""

    def __init__(self, means_pos=((10, 7), ), covs_pos=(((1.2, 0), (0, 1.3)), ),
                 means_neg=((0, 1), ), covs_neg=(((1, 0), (0, 1.2)), ),
                 num_per_pos_label=(500, ), num_per_neg_label=(500, ),
                 pos_lbl=1, neg_lbl=0):
        """

        Args:
            means_pos list(tuple):
            covs_pos list(tuple):
            means_neg:
            covs_neg:
            num_per_label:
            pos_lbl:
            neg_lbl:
        """

        self.mean_pos = [np.asarray(m) for m in means_pos]
        self.mean_neg = [np.asarray(m) for m in means_neg]

        self.cov_pos = [np.asarray(c) for c in covs_pos]
        self.cov_neg = [np.asarray(c) for c in covs_neg]

        self.pos_lbl = pos_lbl
        self.neg_lbl = neg_lbl

        self.num_per_pos_label = num_per_pos_label
        self.num_per_neg_label = num_per_neg_label

    def sample_initial_data(self):
        """Return sample of the initial data."""

        pos_samples = []
        for i, mean in enumerate(self.mean_pos):
            s = np.random.multivariate_normal(mean,
                                              self.cov_pos[i],
                                              int(self.num_per_pos_label[i]))
            pos_samples.append(s)
        pos_samples = np.vstack(pos_samples)

        neg_samples = []
        for i, mean in enumerate(self.mean_neg):
            s = np.random.multivariate_normal(mean,
                                              self.cov_neg[i],
                                              int(self.num_per_neg_label[i]))
            neg_samples.append(s)
        neg_samples = np.vstack(neg_samples)

        samples = np.vstack((pos_samples, neg_samples))
        labels = np.hstack((np.ones(int(np.sum(self.num_per_pos_label))) * self.pos_lbl,
                            np.ones(int(np.sum(self.num_per_neg_label))) * self.neg_lbl))

        # TODO: shuffle

        return samples, labels

    def predict(self, X):  # noqa
        """"""
        return self.get_labels(X)

    def get_labels(self, X):  # noqa
        """Return labels for X."""

        if len(X.shape) < 2:
            X = np.array([X])

        diffs_pos = np.array([np.linalg.norm(X - m, axis=1)
                             for m in self.mean_pos])
        diffs_neg = np.array([np.linalg.norm(X - m, axis=1)
                             for m in self.mean_neg])

        pos_mask = np.asarray(np.min(diffs_pos, axis=0) <
                              np.min(diffs_neg, axis=0))
        y = np.ones_like(pos_mask) * self.neg_lbl
        y[pos_mask] = self.pos_lbl

        return y
