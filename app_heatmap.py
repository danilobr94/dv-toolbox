"""Page for plotting heatmap of data values with different methods."""
import time
import numpy as np
import streamlit as st
import plotly.graph_objs as go

from ui.data_selection_components import dataset_selector, X_BOUNDS, Y_BOUNDS
from ui.sidebar_components import dv_fn_selector, show_info, step_size_selector, project_links
from dv_methods.catastrophic_forgetting import ForgettingDV, LastLearnedDV
from dv_methods.memorization import MemorizationDV
from dv_methods.model_ensemble import EnsembleOOD
from dv_methods.loo import LOODV
from dv_methods.mc_dropout import MCDV
from utils import get_mesh, get_scatter_trace, get_heatmap_trace, get_contour_trace_from_model


DV_METHODS = (EnsembleOOD, ForgettingDV, MemorizationDV,
              LOODV, LastLearnedDV, MCDV)

NUM_POINTS_MESH = 500  # Number of points for decision boundary mesh grid

INSTRUCTION_TEXT = "Select a data valuation method from the sidebar and plot the resulting heatmap of data values." \
                   " \n" \
                   "In the background a mesh-grid is spanned over the range of the plot and each point on the mesh" \
                   "is valued according to the selected method."


def app():
    """Build the streamlit app."""

    # ## Sidebar ##
    project_links()
    X_train, _X_test, y_train, _y_test, syn = dataset_selector()  # noqa
    step_size = step_size_selector()

    dv_function = dv_fn_selector(DV_METHODS)(X_train, y_train)
    show_info()

    # Compute data values
    num_points_heatmap = int(abs(X_BOUNDS[1] - X_BOUNDS[0]) / step_size)
    xx, yy = get_mesh(x_lim=X_BOUNDS, y_lim=Y_BOUNDS, num_points=num_points_heatmap)
    X_mesh = np.c_[xx.ravel(), yy.ravel()]  # noqa
    y_mesh = syn.get_labels(X_mesh)

    dv, baseline_model = dv_function.predict_dv(X_mesh, y_mesh)

    base_model_func = baseline_model.predict if baseline_model is not None else None

    # Create the plot
    fig = go.Figure()

    fig.add_trace(get_scatter_trace(X_train, y_train))
    fig.add_trace(get_heatmap_trace(dv, X_BOUNDS, Y_BOUNDS, num_points_heatmap, "Data Values"))

    fig.add_trace(get_contour_trace_from_model(syn.get_labels, X_BOUNDS, Y_BOUNDS,
                                               NUM_POINTS_MESH, 'True decision boundary',
                                               colorscale='rdbu'))

    fig.add_trace(get_contour_trace_from_model(base_model_func, X_BOUNDS, Y_BOUNDS,
                                               NUM_POINTS_MESH, 'Models decision boundary'))

    model_name = "" if dv_function.NAME != LOODV.NAME else ' with ' + dv_function.model_class.NAME
    fig.update_layout(title={
        'text': f'Data value heatmap for {dv_function.NAME}{model_name}',
        'y': 0.85,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

    # ## Body ##
    with st.expander("See details"):
        st.write(INSTRUCTION_TEXT)
        st.write("Selected valuation method: ", dv_function.NAME)
        st.write("Further Links: ", dv_function.URL)

    st.spinner("Heatmap in Progress")
    st.plotly_chart(fig)

    # Seems to prevent a freeze
    time.sleep(5)


if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(page_title="Create Heatmap", page_icon="images/fhg_logo.png", layout="centered")
    app()
