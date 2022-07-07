import streamlit as st
from sklearn.svm import SVC
from .base import ModelBase


class SVC_(ModelBase):
    NAME = 'SVC'
    URL = 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html'

    @staticmethod
    def param_selector():
        C = st.number_input("C", 0.01, 2.0, 1.0, 0.01)
        kernel = st.selectbox("kernel", ("rbf", "linear", "poly", "sigmoid"))
        params = {"C": C, "kernel": kernel}
        model = SVC(**params)
        return model