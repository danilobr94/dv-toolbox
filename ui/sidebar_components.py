import numpy as np
import streamlit as st
from typing import Tuple
from models.NaiveBayes import NaiveBayes
from models.NeuralNetwork import NN
from models.RandomForest import RF
from models.DecisionTree import DT
from models.LogisticRegression import LogReg
from models.KNearesNeighbors import KNN
from models.SVC import SVC_
from models.GradientBoosting import GB
from dv_methods.base import DVMethodBase
from data.synthetic_data import SyntheticData
from sklearn.model_selection import train_test_split

MODELS = (NN, NaiveBayes, RF, DT, LogReg, KNN, SVC_, GB)
LABEL_AUTO = "auto"
LABEL_POS = "1"
LABEL_NEG = "0"
LABEL_OPTIONS = (LABEL_AUTO, LABEL_POS, LABEL_NEG)
X_BOUNDS = (-10.0, 20.0)
Y_BOUNDS = (-10.0, 20.0)


def dataset_selector(container=None):
    dataset_container = st.sidebar.expander(
        "Dataset", False) if container is None else container

    with dataset_container:
        num_blobs = st.slider(
            "Set number of blobs",
            min_value=2,
            max_value=10,
            step=1,
            value=2,
        )

        st.markdown("""---""", unsafe_allow_html=True)

        def blob_selector(i, default_x=10.0, default_y=7.0, lbl=None):

            x = st.slider("x-value for blob " + str(i), X_BOUNDS[0], X_BOUNDS[1], default_x, 0.5)
            y = st.slider("y-value for blob " + str(i), Y_BOUNDS[0], Y_BOUNDS[1], default_y, 0.5)

            cov = st.slider("Covariance for blob " + str(i), -1, 5, 1)
            num_samples = st.number_input("Number of samples for blob " + str(i), 1, 250, 50)

            if lbl is None:
                lbl = st.selectbox("Label for blob " + str(i), (LABEL_POS, LABEL_NEG))
            else:
                lbl = st.selectbox("Label for blob " + str(i), (lbl, ), disabled=True)

            st.markdown("""---""", unsafe_allow_html=True)
            return x, y, lbl, np.eye(2)*cov, num_samples

        x1, y1, lbl1, cov1, num1 = blob_selector(1, lbl=LABEL_POS)
        x2, y2, lbl2, cov2, num2 = blob_selector(2, .0, .1,  lbl=LABEL_NEG)

        means_pos = [np.array((x1, y1))]
        cov_pos = [cov1]
        num_per_pos_label = [num1]

        means_neg = [np.array((x2, y2))]
        cov_neg = [cov2]
        num_per_neg_label = [num2]

        # Maybe add boxes for the other blobs
        for i in range(3, num_blobs+1):
            x_, y_, lbl_, cov_, num_ = blob_selector(i)
            if lbl_ == LABEL_POS:
                means_pos.append((x_, y_))
                cov_pos.append(cov_)
                num_per_pos_label.append(num_)

            else:
                means_neg.append((x_, y_))
                cov_neg.append(cov_)
                num_per_neg_label.append(num_)

        syn = SyntheticData(means_pos, cov_pos,
                            means_neg, cov_neg,
                            num_per_pos_label, num_per_neg_label)

        data, labels = syn.sample_initial_data()

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test, syn


def model_selector(container=None):
    model_training_container = st.sidebar.expander(
        "Model selection", True) if container is None else container

    with model_training_container:
        model_type = st.selectbox("Choose a model", [m.NAME for m in MODELS], )

        for model_class in MODELS:
            if model_class.NAME == model_type:
                model = model_class.param_selector()
                return model_class, model

    return None


def num_repetitions_selector(container=None):
    container = st.sidebar.expander(
        "Number of Repetitions", True) if container is None else container

    with container:
        num_repetitions = st.number_input("Set number of repetitions:",
                                          min_value=1,
                                          max_value=15,
                                          step=1,
                                          value=1, key="3")
    return num_repetitions


def label_selector(container=None):
    container = st.sidebar.expander(
        "Label Selection", True) if container is None else container

    with container:
        label = st.selectbox("Choose label", LABEL_OPTIONS)

    return label


def point_slider(container=None):
    slider_container = st.sidebar.expander(
        "Point Position Selection", True) if container is None else container

    with slider_container:
        label = st.selectbox("Choose label", LABEL_OPTIONS)
        x1 = st.slider("Slider for x position", X_BOUNDS[0], X_BOUNDS[1], -8.0, 0.5)
        x2 = st.slider("Slider for y position", X_BOUNDS[0], X_BOUNDS[1], -8.0, 0.5)

    return np.array([x1, x2]), label


def step_selector(container=None):
    container = st.sidebar.expander(
        "Number of steps", True) if container is None else container

    with container:
        num_steps = st.number_input(
            "set number of steps:", min_value=1, value=1, step=1)

    return num_steps


def step_size_selector(container=None):
    container = st.sidebar.expander(
        "Step size", True) if container is None else container

    with container:
        num_steps = st.number_input(
            "Set step size:", min_value=1, value=1, step=1)

    return num_steps


def num_models_selector(container=None):
    container = st.sidebar.expander(
        "Number of models", True) if container is None else container

    with container:
        num_steps = st.number_input(
            "Set number of models:", min_value=1, value=5, step=25)

    return num_steps


def num_hidden_layers_selector(container=None):
    container = st.sidebar.expander(
        "Number of hidden layers", True) if container is None else container

    with container:
        number_hidden_layers = st.number_input(
            "Select number of hidden layers", 1, 5, 1)

    return number_hidden_layers


def size_hidden_layer_selector(container=None):
    container = st.sidebar.expander(
        "Size of hidden layer", True) if container is None else container

    with container:
        number_hidden_layers = st.number_input(
            "Select size of hidden layer(s)", 1, 1000, 50, 25)

    return number_hidden_layers


def dv_fn_selector(dv_options: Tuple[DVMethodBase], container=None):
    dv_container = st.sidebar.expander(
        "Data Valuation Method", True) if container is None else container

    with dv_container:
        dv_functions = st.selectbox(
            "Choose a method", options=dv_options, format_func=lambda option: option.NAME)

    return dv_functions


def show_info():

    st.sidebar.markdown(
        """
        [<img src="https://github.com/danilobr94/dv-toolbox/blob/main/images/ipa_logo.png" \
        width=25 height=25>](https://www.ipa.fraunhofer.de/)
        """,
        unsafe_allow_html=True,
    )
