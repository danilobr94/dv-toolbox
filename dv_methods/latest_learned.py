"""Last-learned samples"""
import streamlit as st
import numpy as np
from .base import DVMethodBase
from ui.sidebar_components import num_models_selector, size_hidden_layer_selector
from sklearn.neural_network import MLPClassifier as MLP
from stqdm import stqdm
import sklearn as sk


class LastLearnedDV(DVMethodBase):
    """Estimate data value with forgetting events."""

    NAME = "Latest Learned"
    URL = "TODO"

    def __init__(self, X_base=None, y_base=None):
        """

        Args:
            X_base:
            y_base:
        """

        self.X_base = X_base
        self.y_base = y_base

        container = st.sidebar.expander("Configure the neural network", True)
        self.hidden_layer_sizes = size_hidden_layer_selector(container)
        # Init model with warm start
        self.model = MLP(hidden_layer_sizes=self.hidden_layer_sizes,
                         activation='relu', max_iter=1, warm_start=True)

        self.base_model = MLP(
            hidden_layer_sizes=self.hidden_layer_sizes, activation='relu')
        self.base_model.fit(X_base, y_base)

        self.num_epochs = 100

    def predict_dv(self, X, y):
        """"""
        forgetting_stats = np.zeros_like(y)
        for i in stqdm(range(X.shape[0])):
            X_train_new = np.vstack([self.X_base, X[i]])
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
