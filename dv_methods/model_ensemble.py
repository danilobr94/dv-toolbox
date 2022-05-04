""""""
import numpy as np
import sklearn
from stqdm import stqdm
from models.NeuralNetwork import NN
from ui.sidebar_components import num_models_selector
from .base import DVMethodBase
import streamlit as st


class EnsembleOOD(DVMethodBase):
    NAME = "Ensemble"
    URL = "TODO"

    def __init__(self, X_base=None, y_base=None):
        """"""
        print("E1")
        self.X_base = X_base
        self.y_base = y_base

        print("E2")
        container = st.sidebar.expander("Configure the neural network", True)

        with container:
            self.model = NN.param_selector()

        self.num_models = num_models_selector(container)

        print("E3")
        self.models = [sklearn.base.clone(self.model).fit(
            X_base, y_base) for _ in stqdm(range(self.num_models))]
        print("Done!")

    def predict_dv(self, X, _y):
        predictions = np.array([model.predict(X) for model in self.models])
        std = np.std(predictions, axis=0)
        return std, self

    @st.cache(suppress_st_warning=True)
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        mean = np.mean(predictions, axis=0)
        return np.round(mean)
