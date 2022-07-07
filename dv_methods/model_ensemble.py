""""""
import numpy as np
import sklearn

from models.nn import NN
from ui.model_selection_components import num_models_selector
from .base import DVMethodBase
import streamlit as st
from utils import StreamlitProgressBar


class EnsembleOOD(DVMethodBase):
    NAME = "Ensemble"
    URL = "TODO"

    def __init__(self, X_base=None, y_base=None):  # noqa
        """"""
        super().__init__(X_base, y_base)

        container = st.sidebar.expander("Configure the neural network", True)

        with container:
            self.model = NN.param_selector()

        self.num_models = num_models_selector(container)
        self.models = [sklearn.base.clone(self.model).fit(X_base, y_base) for _ in range(self.num_models)]

    def predict_dv(self, X, _y):  # noqa
        predictions = np.array([model.predict(X) for model in StreamlitProgressBar(self.models)])
        std = np.std(predictions, axis=0)
        return std, self

    def predict(self, X):  # noqa
        """Compute predictions with majority voting for the ensemble."""
        predictions = np.array([model.predict(X) for model in self.models])
        mean = np.mean(predictions, axis=0)
        return np.round(mean)
