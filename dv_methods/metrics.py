"""Measures difference between two decision boundaries."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


class TrainingPerformance:
    """Training performance when removing points with high and low data values."""

    def __init__(self, X_train, y_train, X_test, y_test, data_values):
        """"""

    def run(self, model):
        """Run the evaluation for the model."""


def remove_high_low(data_values, eval_model, x_train, y_train,
                    x_valid, y_valid, x_test, y_test, plot=True):
    """Evaluates performance after removing a portion of high/low valued samples.
      CODE from: https://github.com/google-research/google-research/blob/master/dvrl/dvrl_metrics.py

      Args:
        data_values: data values
        eval_model: evaluation model (object)
        x_train: training features
        y_train: training labels
        x_valid: validation features
        y_valid: validation labels
        x_test: testing features
        y_test: testing labels
        plot: print plot or not

      Returns:
        output_perf: Prediction performances after removing a portion of high
                     or low valued samples.
      """

    x_train = np.asarray(x_train)
    y_train = np.reshape(np.asarray(y_train), [len(y_train), ])
    x_valid = np.asarray(x_valid)
    y_valid = np.reshape(np.asarray(y_valid), [len(y_valid), ])
    x_test = np.asarray(x_test)
    y_test = np.reshape(np.asarray(y_test), [len(y_test), ])

    # Sorts samples by data values
    num_bins = 20  # Per 100/20 percentile
    sort_idx = np.argsort(data_values)
    n_sort_idx = np.argsort(-data_values)

    # Output Initialization
    temp_output = np.zeros([2 * num_bins, 2])

    # For each percentile bin
    for itt in range(num_bins):

        # 1. Remove least valuable samples first
        new_x_train = x_train[sort_idx[int(itt * len(x_train[:, 0]) / num_bins):], :]
        new_y_train = y_train[sort_idx[int(itt * len(x_train[:, 0]) / num_bins):]]

        if len(np.unique(new_y_train)) > 1:

            eval_model.fit(new_x_train, new_y_train)
            y_valid_hat = eval_model.predict_proba(x_valid)
            y_test_hat = eval_model.predict_proba(x_test)

            temp_output[itt, 0] = metrics.accuracy_score(y_valid,
                                                         np.argmax(y_valid_hat, axis=1))

            temp_output[itt, 1] = metrics.accuracy_score(y_test,
                                                         np.argmax(y_test_hat, axis=1))

        # 2. Remove most valuable samples first
        new_x_train = x_train[n_sort_idx[int(itt * len(x_train[:, 0]) / num_bins):], :]
        new_y_train = y_train[n_sort_idx[int(itt * len(x_train[:, 0]) / num_bins):]]

        if len(np.unique(new_y_train)) > 1:

            eval_model.fit(new_x_train, new_y_train)

            y_valid_hat = eval_model.predict_proba(x_valid)
            y_test_hat = eval_model.predict_proba(x_test)

            temp_output[num_bins + itt, 0] = metrics.accuracy_score(y_valid, np.argmax(y_valid_hat, axis=1))
            temp_output[num_bins + itt, 1] = metrics.accuracy_score(y_test, np.argmax(y_test_hat, axis=1))

    # Plot graphs
    if plot:
        # Defines x-axis
        num_x = int(num_bins / 2 + 1)
        x = [a * (1.0 / num_bins) for a in range(num_x)]

        # Prediction performances after removing high or low values
        plt.figure(figsize=(6, 7.5))
        plt.plot(x, temp_output[:num_x, 1], 'o-')
        plt.plot(x, temp_output[num_bins:(num_bins + num_x), 1], 'x-')

        plt.xlabel('Fraction of Removed Samples', size=16)
        plt.ylabel('Accuracy', size=16)
        plt.legend(['Removing low value data', 'Removing high value data'], prop={'size': 16})
        plt.title('Remove High/Low Valued Samples', size=16)

        plt.show()

    return temp_output


class ParamDifference:
    """Measure value as difference between parameters."""

    def __init__(self, base_model):
        """"""
        self.base_model = base_model
        self.base_params = self._concat(base_model.coefs_)

    def _concat(self, coefs):
        """Concatenate list of arrays 'coeafs'."""
        coeafs_flat = [c.flatten() for c in coefs]
        return  np.concatenate(coeafs_flat)

    def compute_difference(self, model):
        """"""
        params = self._concat(model.coefs_)
        return np.mean(params - self.base_params)

class AccuracyDifference:
    """"""

    def __init__(self, X_test, y_test, base_model):
        """"""
        self.X_test = X_test
        self.y_test = y_test
        self.base_model = base_model
        self.acc_base = self.base_model.score(self.X_test, self.y_test)

    def compute_difference(self, model):
        """"""
        try:
            pred = model.predixt(self.X_test)
        except:
            pred = model(self.X_test)

        true_pred = pred == self.y_test
        acc = np.mean(true_pred)
        return abs(acc - self.acc_base)


class DecisionBoundaryDifference:
    """"""

    def __init__(self, x_lim, y_lim, baseline_model, mesh_size=500):
        """

        Args:
            x_lim (int, int): Lower and upper bound for x-axis in plot.
            y_lim (int, int): Lower and upper bound for y-axis in plot.
            baseline_model (function): The baseline (ground truth model), should be calleble e.g. 'model.predict'.
            mesh_size (int): The size of the mesh grid.
        """

        self.x_lim = x_lim
        self.y_lim = y_lim
        self.baseline_model = baseline_model
        self.mesh_size = mesh_size

        x_min, x_max = self.x_lim
        y_min, y_max = self.y_lim

        x1_step = (x_max - x_min) / self.mesh_size
        x2_step = (y_max - y_min) / self.mesh_size

        xx, yy = np.meshgrid(np.arange(x_min, x_max, x1_step),
                             np.arange(y_min, y_max, x2_step))

        self.mesh = np.c_[xx.ravel(), yy.ravel()]
        self.Z_true = self.baseline_model(self.mesh)

    def compute_difference(self, learned_dec_bound_func):
        """

        Args:
            learned_dec_bound_func (Callable): The models decision boundary
                (i.e. the decision function or 'model.predict()') or a model implementing '.predict()'.
        """
        try:
            Z_model = learned_dec_bound_func.predict(self.mesh)

        except:
            Z_model = learned_dec_bound_func(self.mesh)

        return np.mean(np.abs(self.Z_true - Z_model))


if __name__ == "__main__":
    from synthetic_data import SyntheticData
    from decision_boundary import Scatter2D
    from sklearn.neural_network import MLPClassifier as MLP

    data, labels = SyntheticData().sample_initial_data()

    model1 = MLP(hidden_layer_sizes=(100), activation='relu', max_iter=1000)
    model1.fit(data, labels)

    model2 = MLP(hidden_layer_sizes=(50), activation='relu', max_iter=1000)
    model2.fit(data, labels)

    sct = Scatter2D(data, labels)
    sct.add_boundary(model1.predict)
    sct.add_boundary(model2.predict)
    sct.show()

    db_diff = DecisionBoundaryDifference(x_lim=(-10, 20), y_lim=(-10, 20))
    print(db_diff.compute_difference(model1.predict, model2.predict))
