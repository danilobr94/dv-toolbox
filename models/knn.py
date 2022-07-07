import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from .base import ModelBase


class KNN(ModelBase):
    NAME = 'KNN'
    URL = 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html'

    @staticmethod
    def param_selector():

        n_neighbors = st.number_input("n_neighbors", 5, 20, 5, 1)
        metric = st.selectbox(
            "metric", ("minkowski", "euclidean", "manhattan", "chebyshev", "mahalanobis")
        )

        params = {"n_neighbors": n_neighbors, "metric": metric}

        model = KNeighborsClassifier(**params)
        return model
