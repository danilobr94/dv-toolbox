"""Page for plotting heatmap of data values with different methods."""
import streamlit as st
from ui.sidebar_components import dataset_selector, dv_fn_selector, show_info, step_size_selector, project_links
from dv_methods.catastrophic_forgetting import ForgettingDV
from dv_methods.memorization import MemorizationDV
from dv_methods.model_ensemble import EnsembleOOD
from dv_methods.loo import LOODV
from dv_methods.latest_learned import LastLearnedDV
from dv_methods.mc_dropout import MCDV
from utils import *
from ui.sidebar_components import X_BOUNDS, Y_BOUNDS

DV_METHODS = (EnsembleOOD, ForgettingDV, MemorizationDV,
              LOODV, LastLearnedDV, MCDV)

NUM_POINTS_MESH = 500  # Number of points for decision boundary mesh grid


def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Sidebar
    project_links()
    X_train, X_test, y_train, y_test, syn = dataset_selector()  # noqa
    step_size = step_size_selector()
    dv_function = dv_fn_selector(DV_METHODS)(X_train, y_train)
    show_info()

    # Body
    col1, col2, col3 = st.columns((1, 4, 1))
    with col1:
        st.markdown('')

    with col2:
        plot1_placeholder = st.empty()

    with col3:
        st.markdown('')

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

    st.spinner("Heatmap in Progress")
    plot1_placeholder.plotly_chart(fig)


if __name__ == "__main__":
    app()
