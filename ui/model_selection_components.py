"""Sidebar components related to the model selection."""
import streamlit as st

from models.NaiveBayes import NaiveBayes
from models.NeuralNetwork import NN
from models.RandomForest import RF
from models.DecisionTree import DT
from models.LogisticRegression import LogReg
from models.KNearesNeighbors import KNN
from models.SVC import SVC_
from models.GradientBoosting import GB

MODELS = (NN, NaiveBayes, RF, DT, LogReg, KNN, SVC_, GB)


def model_selector(container=None):
    """"""
    model_training_container = st.sidebar.expander(
        "Model selection", True) if container is None else container

    with model_training_container:
        model_type = st.selectbox("Choose a model", [m.NAME for m in MODELS], )

        # TODO: geht eleganter...
        for model_class in MODELS:
            if model_class.NAME == model_type:
                model = model_class.param_selector()
                return model_class, model

    return None


def num_models_selector(container=None):
    """"""
    container = st.sidebar.expander(
        "Number of models", True) if container is None else container

    with container:
        num_steps = st.number_input(
            "Set number of models:", min_value=1, value=5, step=25)

    return num_steps


def num_hidden_layers_selector(container=None):
    """"""
    container = st.sidebar.expander(
        "Number of hidden layers", True) if container is None else container

    with container:
        number_hidden_layers = st.number_input(
            "Select number of hidden layers", 1, 5, 1)

    return number_hidden_layers


def size_hidden_layer_selector(container=None):
    """"""
    container = st.sidebar.expander(
        "Size of hidden layer", True) if container is None else container

    with container:
        number_hidden_layers = st.number_input(
            "Select size of hidden layer(s)", 1, 1000, 50, 25)

    return number_hidden_layers
