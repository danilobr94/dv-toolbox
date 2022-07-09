"""Measures difference between two decision boundaries."""
import numpy as np


class ParamDifference:
    """Measure value as difference between parameters."""

    def __init__(self, base_model):
        """"""
        self.base_model = base_model
        self.base_params = self._concat(base_model.coefs_)

    @staticmethod
    def _concat(coefs):  # noqa
        """Concatenate list of arrays 'coefs'."""  # noqa
        coefs_flat = [c.flatten() for c in coefs]
        return np.concatenate(coefs_flat)

    def compute_difference(self, model):
        """"""
        params = self._concat(model.coefs_)
        return np.mean(params - self.base_params)


class AccuracyDifference:
    """"""

    def __init__(self, X_test, y_test, base_model):  # noqa
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
    """Measure value of a data points as area between curves of a model with and without the POI."""

    def __init__(self, x_lim, y_lim, baseline_model, mesh_size=500):
        """

        Args:
            x_lim (int, int): Lower and upper bound for x-axis in plot.
            y_lim (int, int): Lower and upper bound for y-axis in plot.
            baseline_model (function): The baseline (ground truth model), should be callable e.g. 'model.predict'.
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
            Z_model = learned_dec_bound_func.predict(self.mesh)  # noqa

        except:
            Z_model = learned_dec_bound_func(self.mesh)  # noqa

        return np.mean(np.abs(self.Z_true - Z_model))
