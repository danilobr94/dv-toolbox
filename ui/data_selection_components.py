"""Sidebar components for data set selection."""
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split

from data.synthetic_data import SyntheticData


LABEL_AUTO = "auto"
LABEL_POS = "1"
LABEL_NEG = "0"
LABEL_OPTIONS = (LABEL_AUTO, LABEL_POS, LABEL_NEG)
X_BOUNDS = (-10.0, 20.0)
Y_BOUNDS = (-10.0, 20.0)


def label_selector(container=None):
    """"""
    container = st.sidebar.expander(
        "Label Selection", True) if container is None else container

    with container:
        label = st.selectbox("Choose label", LABEL_OPTIONS)

    return label


def point_slider(container=None):
    """"""
    slider_container = st.sidebar.expander(
        "Point Position Selection", True) if container is None else container

    with slider_container:
        x1 = st.slider("Slider for x position", X_BOUNDS[0], X_BOUNDS[1], -8.0, 0.5)
        x2 = st.slider("Slider for y position", X_BOUNDS[0], X_BOUNDS[1], -8.0, 0.5)

    return np.array([x1, x2])


@st.cache
def _get_data(means_pos, cov_pos, means_neg, cov_neg, num_per_pos_label, num_per_neg_label):
    """"""
    syn = SyntheticData(means_pos, cov_pos,
                        means_neg, cov_neg,
                        num_per_pos_label, num_per_neg_label)

    data, labels = syn.sample_initial_data()
    return data, labels, syn


def dataset_selector(container=None):
    """"""
    dataset_container = st.sidebar.expander(
        "Dataset", False) if container is None else container

    with dataset_container:
        num_blobs = st.slider(
            "Set number of blobs", min_value=2, max_value=10, step=1, value=2, )

        st.markdown("""---""", unsafe_allow_html=True)

        def blob_selector(k, default_x=10.0, default_y=7.0, lbl=None):
            """Helper function to set values for a single blob."""
            x = st.slider("x-value for blob " + str(k), X_BOUNDS[0], X_BOUNDS[1], default_x, 0.5)
            y = st.slider("y-value for blob " + str(k), Y_BOUNDS[0], Y_BOUNDS[1], default_y, 0.5)

            cov = st.slider("Covariance for blob " + str(k), -1, 5, 1)
            num_samples = st.number_input("Number of samples for blob " + str(k), 25, 500, 250)

            if lbl is None:
                lbl = st.selectbox("Label for blob " + str(k), (LABEL_POS, LABEL_NEG))
            else:
                lbl = st.selectbox("Label for blob " + str(k), (lbl,), disabled=True)

            st.markdown("""---""", unsafe_allow_html=True)
            return x, y, lbl, np.eye(2) * cov, num_samples

        x1, y1, lbl1, cov1, num1 = blob_selector(1, lbl=LABEL_POS)
        x2, y2, lbl2, cov2, num2 = blob_selector(2, .0, .1, lbl=LABEL_NEG)

        means_pos = [np.array((x1, y1))]
        cov_pos = [cov1]
        num_per_pos_label = [num1]

        means_neg = [np.array((x2, y2))]
        cov_neg = [cov2]
        num_per_neg_label = [num2]

        # Maybe add boxes for the other blobs
        for i in range(3, num_blobs + 1):
            x_, y_, lbl_, cov_, num_ = blob_selector(i)

            if lbl_ == LABEL_POS:
                means_pos.append((x_, y_))
                cov_pos.append(cov_)
                num_per_pos_label.append(num_)

            else:
                means_neg.append((x_, y_))
                cov_neg.append(cov_)
                num_per_neg_label.append(num_)

        data, labels, syn = _get_data(means_pos, cov_pos, means_neg, cov_neg,
                                      num_per_pos_label, num_per_neg_label)

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)  # noqa

    return X_train, X_test, y_train, y_test, syn
