"""All side bar components not related to the data set or model selection."""
import streamlit as st
from typing import Tuple

from dv_methods.base import DVMethodBase


def project_links(container=None):
    """Links to related apps."""
    container = st.sidebar.expander(
        "Related Apps", True) if container is None else container

    with container:
        st.write("[Create a gif](https://share.streamlit.io/danilobr94/dv-toolbox/main/app_autorun.py)")
        st.write("[Play with the decision boundary](https://share.streamlit.io/danilobr94/dv-toolbox/main/app_home.py)")
        st.write("[Create heatmap](https://share.streamlit.io/danilobr94/dv-toolbox/main/app_heatmap.py)")


def show_info():
    """"""
    st.sidebar.markdown("[<img src=https://danilobr94.github.io/dv-toolbox/images/ipa_logo.png width=70% height=70%>]"
                        "(https://www.ipa.fraunhofer.de/)",
                        unsafe_allow_html=True,
                        )


def num_repetitions_selector(container=None):
    """"""
    container = st.sidebar.expander(
        "Number of Repetitions", True) if container is None else container

    with container:
        num_repetitions = st.number_input("Set number of repetitions:",
                                          min_value=1,
                                          max_value=15,
                                          step=1,
                                          value=1, key="3")
    return num_repetitions


def step_selector(container=None):
    """"""
    container = st.sidebar.expander(
        "Number of steps", True) if container is None else container

    with container:
        num_steps = st.number_input(
            "set number of steps:", min_value=1, value=1, step=1)

    return num_steps


def step_size_selector(container=None):
    """"""
    container = st.sidebar.expander(
        "Step size", True) if container is None else container

    with container:
        num_steps = st.number_input(
            "Set step size:", min_value=1, value=3, step=1)

    return num_steps


def dv_fn_selector(dv_options: Tuple[DVMethodBase], container=None):
    """"""
    dv_container = st.sidebar.expander("Data Valuation Method", True) if container is None else container

    with dv_container:
        dv_function = st.selectbox("Choose a method", [option.NAME for option in dv_options])

        # TODO: geht eleganter...
        for dv_method in dv_options:
            if dv_method.NAME == dv_function:
                return dv_method

    return dv_function
