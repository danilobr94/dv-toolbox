"""Data values with catastrophic forgetting and latest-learned samples.

Catastrophic forgetting counts how often a point was forgotten again during training after being correctly classified
at least once.

Latest learned uses the first epoch a point was classified correctly for the first time as its value.

"""
from abc import ABC
import sklearn.base
import sklearn as sk
import streamlit as st
import numpy as np

from models.nn import NN
from utils import StreamlitProgressBar
from .base import DVMethodBase


class _ForgettingBase(DVMethodBase, ABC):
    """Base class for the two forgetting methods."""

    def __init__(self, X_base=None, y_base=None):  # noqa
        """The init function is the same for both catastrophic forgetting and latest learned."""
        super().__init__(X_base, y_base)

        # self.hidden_layer_sizes = size_hidden_layer_selector(container)
        container = st.sidebar.expander("Configure the neural network", True)

        with container:
            self.base_model = NN.param_selector()

        # Clone the base model, activate 'warm_start' and
        # set 'max_iter=1' so that the model can be evaluated each epoch
        self.model = sklearn.base.clone(self.base_model)
        self.model.max_iter = 1
        self.model.warm_start = True

        self.base_model.fit(X_base, y_base)
        self.num_epochs = 100


class ForgettingDV(_ForgettingBase):
    """Estimate data value with forgetting events."""

    NAME = "Catastrophic Forgetting"
    URL = "TBD"

    def predict_dv(self, X, y):  # noqa
        """"""
        forgetting_stats = np.zeros_like(y)
        for i in StreamlitProgressBar(range(X.shape[0])):

            X_train_new = np.vstack([self.X_base, X[i]])  # noqa
            y_train_new = np.hstack([self.y_base, y[i]])

            memorized = False
            model = sk.base.clone(self.model)

            for epoch in range(self.num_epochs):
                # model initialized with warm start and fits for one epoch only
                model.fit(X_train_new, y_train_new)
                y_pred = model.predict(X[i].reshape(1, -1))

                if y[i] == y_pred and not memorized:
                    memorized = True

                if y[i] != y_pred and memorized:
                    forgetting_stats[i] += 1

        forgetting_stats = forgetting_stats / self.num_epochs
        return forgetting_stats, self.base_model


class LastLearnedDV(_ForgettingBase):
    """Estimate data value as the epoch a point was correctly classified in for the first time."""

    NAME = "Latest Learned"
    URL = "TODO"

    def predict_dv(self, X, y):  # noqa
        """"""
        forgetting_stats = np.zeros_like(y)
        for i in StreamlitProgressBar(range(X.shape[0])):
            X_train_new = np.vstack([self.X_base, X[i]])  # noqa
            y_train_new = np.hstack([self.y_base, y[i]])

            memorized = False
            model = sk.base.clone(self.model)

            for epoch in range(self.num_epochs):
                # model initialized with warm start and fits for one epoch only
                model.fit(X_train_new, y_train_new)

                y_pred = model.predict(X[i].reshape(1, -1))

                if y[i] == y_pred and not memorized:
                    forgetting_stats[i] = epoch
                    memorized = True

        forgetting_stats = forgetting_stats / self.num_epochs
        return forgetting_stats, self.base_model
