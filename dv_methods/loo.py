"""Leave-one-out data valuation"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sklearn
from data.decision_boundary import Scatter2D
from stqdm import stqdm
from .base import DVMethodBase
from ui.sidebar_components import model_selector
from dv_methods.metrics import DecisionBoundaryDifference


class LOODV(DVMethodBase):
    """"""

    NAME = "Leave One Out"
    URL = "TODO"

    def __init__(self, X_base=None, y_base=None):
        """

        Args:
            model:
            metric:
            X_base:
            y_base:
        """
        _, self.model = model_selector()
        # TODO: need to set the limits somewhere central ...
        self.base_model = sklearn.base.clone(self.model)
        self.base_model.fit(X_base, y_base)

        self.metric = DecisionBoundaryDifference(x_lim=(-10, 20),
                                                 y_lim=(-10, 20),
                                                 baseline_model=self.base_model.predict,
                                                 mesh_size=500).compute_difference

        self.X_base = X_base
        self.y_base = y_base


    def predict_dv(self, X, y, inv_diff=False):
        """"""

        db_diff = []
        for i in stqdm(range(X.shape[0])):
            if self.X_base is not None:
                X_train_new = np.vstack([self.X_base, X[i]])
                y_train_new = np.hstack([self.y_base, y[i]])
            else:
                X_train_new = np.delete(X, i, 0)
                y_train_new = np.delete(y, i)

            self.model.fit(X_train_new, y_train_new)

            if inv_diff:
                db_diff.append(1 - self.metric(self.model))
            else:
                db_diff.append(self.metric(self.model))

        return np.asarray(db_diff), self.base_model
