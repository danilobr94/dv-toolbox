import numpy as np
import tensorflow as tf
import streamlit as st

from utils import StreamlitProgressBar
from .base import DVMethodBase
from ui.sidebar_components import num_hidden_layers_selector, num_models_selector, size_hidden_layer_selector


class MCDV(DVMethodBase):
    """Data valuation using MC drop with in keras."""

    NAME = "MC Dropout"
    URL = "TBD"

    def __init__(self, X_base, y_base):
        """"""
        container = st.sidebar.expander(
            "Configure the neural network ensemble", True)
        num_hidden_layers = num_hidden_layers_selector(container)
        size_hidden_layers = size_hidden_layer_selector(container)
        self.num_mc_steps = num_models_selector(container)

        num_epochs = 100
        batch_size = 64

        # Build model
        def _get_model(mc=True):
            input_shape = X_base.shape[1:]
            inp = tf.keras.Input(input_shape)

            z = tf.keras.layers.Dense(
                size_hidden_layers, activation='relu')(inp)

            for _ in range(num_hidden_layers - 1):
                z = tf.keras.layers.Dense(
                    size_hidden_layers, activation='relu')(z)

            z = tf.keras.layers.Dropout(0.3)(z, training=mc)
            out = tf.keras.layers.Dense(2, activation='softmax')(z)

            model = tf.keras.models.Model(inputs=inp, outputs=out)
            model.compile(
                optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model

        self.model = _get_model()
        self.base_model = _get_model(mc=False)

        # Fit model
        for _ in range(num_epochs):
            self.model.fit(X_base, [y_base], batch_size=batch_size, epochs=1, verbose=0)
            self.base_model.fit(X_base, [y_base], batch_size=batch_size, epochs=1, verbose=0)

    def evaluate_mc(self, X_test, y_test, num_steps=100):
        """"""
        acc = [self.model.evaluate(X_test, y_test, verbose=0)
               for _ in range(num_steps)]
        print("Mean accuarcy:", np.mean(acc), "std:", np.std(acc))
        print("Min:", np.min(acc), "max:", np.max(acc))

    def predict(self, X):
        """"""
        return np.argmax(self.base_model.predict(X), axis=1)

    def predict_dv(self, X, _y):
        """"""
        predictions = np.array(
            [np.argmax(self.model.predict(X, verbose=0), axis=1)
             for _ in StreamlitProgressBar(range(self.num_mc_steps))]
        )
        std = np.std(predictions, axis=0)
        return std, self
