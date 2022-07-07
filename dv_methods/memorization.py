"""Data valuation with memorization."""
import numpy as np
import sklearn
import streamlit as st

from utils import StreamlitProgressBar
from ui.model_selection_components import num_models_selector
from models.nn import NN
from .base import DVMethodBase


class MemorizationDV(DVMethodBase):
    """"""
    NAME = "Memorization"
    URL = "TODO"

    def __init__(self, X_base=None, y_base=None):  # noqa
        """"""
        super().__init__(X_base, y_base)

        container = st.sidebar.expander("Configure the neural network", True)
        with container:
            self.model = NN.param_selector()

        self.num_models = num_models_selector(container)

    def predict_dv(self, X, y, *args):  # noqa
        """"""

        # Make predictions with the baseline model
        base_predictions = []
        for _ in range(self.num_models):
            base_model = sklearn.base.clone(self.model)
            if self.X_base is not None:
                base_model.fit(self.X_base, self.y_base)
            else:
                base_model.fit(X, y)

            base_predictions.append(base_model.predict(X))

        base_predictions = np.array(base_predictions).T

        # Make predictions with models trained without each x_i
        predictions = []
        for i in StreamlitProgressBar(range(X.shape[0])):

            if self.X_base is not None:
                X_train_new = np.vstack([self.X_base, X[i]])
                y_train_new = np.hstack([self.y_base, y[i]])
            else:
                X_train_new = np.delete(X, i, 0)  # noqa
                y_train_new = np.delete(y, i)

            pred = []
            for _ in range(self.num_models):
                self.model.fit(X_train_new, y_train_new)
                pred.append(self.model.predict(X)[i])

            predictions.append(np.array(pred))

        predictions = np.array(predictions)

        # Compare predictions
        diff = np.abs(predictions - base_predictions)
        return np.mean(diff, axis=1), base_model
