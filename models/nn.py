import streamlit as st
from sklearn.neural_network import MLPClassifier
from .base import ModelBase


class NN(ModelBase):
    NAME = 'Neural Network'
    URL = 'https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html'

    @staticmethod
    def param_selector():

        batch_size = st.number_input("Batch size:", 1, 1000, 200, 10)
        max_iter = st.number_input("Maximum number of iterations", 2, 2000, 500, 25)
        activation = st.selectbox("Activation function", ("relu", "identity", "logistic", "tanh"), 0)
        solver = st.selectbox("Solver", ("lbfgs", "sgd", "adam"), 2)

        number_hidden_layers = st.number_input("Number of hidden units", 1, 5, 1)
        hidden_layer_sizes = []

        for i in range(number_hidden_layers):
            n_neurons = st.number_input(f"Number of neurons at layer {i + 1}", 2, 1000, 500, 25)
            hidden_layer_sizes.append(n_neurons)

        hidden_layer_sizes = tuple(hidden_layer_sizes)
        params = {"hidden_layer_sizes": hidden_layer_sizes,
                  "batch_size": batch_size,
                  "max_iter": max_iter,
                  "activation": activation,
                  "solver": solver}

        model = MLPClassifier(**params)
        return model
