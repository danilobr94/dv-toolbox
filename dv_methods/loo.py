"""Leave-one-out data valuation."""
import numpy as np
import sklearn

from .base import DVMethodBase
from ui.model_selection_components import model_selector
from dv_methods.metrics import DecisionBoundaryDifference
from ui.data_selection_components import X_BOUNDS, Y_BOUNDS
from utils import StreamlitProgressBar


class LOODV(DVMethodBase):  # noqa
    """Leave-one-out data valuation."""

    NAME = "Leave One Out"
    URL = "TODO"

    def __init__(self, X_base=None, y_base=None):  # noqa
        """"""
        super().__init__(X_base, y_base)

        self.model_class, self.model = model_selector()
        self.base_model = sklearn.base.clone(self.model)
        self.base_model.fit(X_base, y_base)

        self.metric = DecisionBoundaryDifference(x_lim=X_BOUNDS,
                                                 y_lim=Y_BOUNDS,
                                                 baseline_model=self.base_model.predict,
                                                 mesh_size=500).compute_difference

    def predict_dv(self, X, y, inv_diff=False):  # noqa
        """"""

        db_diff = []
        for i in StreamlitProgressBar(range(X.shape[0])):
            if self.X_base is not None:
                X_train_new = np.vstack([self.X_base, X[i]])  # noqa
                y_train_new = np.hstack([self.y_base, y[i]])
            else:
                X_train_new = np.delete(X, i, 0)  # noqa
                y_train_new = np.delete(y, i)

            self.model.fit(X_train_new, y_train_new)

            if inv_diff:
                db_diff.append(1 - self.metric(self.model))
            else:
                db_diff.append(self.metric(self.model))

        return np.asarray(db_diff), self.base_model
