"""Page for plotting heatmap of data values with different methods."""
from ui.data_selection_components import dataset_selector, X_BOUNDS, Y_BOUNDS
from ui.sidebar_components import dv_fn_selector, show_info, step_size_selector, project_links
from dv_methods.catastrophic_forgetting import ForgettingDV, LastLearnedDV
from dv_methods.memorization import MemorizationDV
from dv_methods.model_ensemble import EnsembleOOD
from dv_methods.loo import LOODV
from dv_methods.mc_dropout import MCDV
from utils import *


DV_METHODS = (EnsembleOOD, ForgettingDV, MemorizationDV,
              LOODV, LastLearnedDV, MCDV)

NUM_POINTS_MESH = 500  # Number of points for decision boundary mesh grid


def app():

    # Sidebar
    project_links()
    X_train, X_test, y_train, y_test, syn = dataset_selector()  # noqa
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

    # Add the figure to the body
    st.spinner("Heatmap in Progress")
    st.plotly_chart(fig)

    # seems to prevent a freeze
    time.sleep(5)


if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    app()
