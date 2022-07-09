"""Entrance point for data value computation."""
import sklearn.base

from dv_methods.metrics import DecisionBoundaryDifference
from ui.data_selection_components import X_BOUNDS, Y_BOUNDS, dataset_selector, label_selector, point_slider
from ui.sidebar_components import show_info, project_links
from ui.model_selection_components import model_selector

from utils import *

NUM_POINTS_MESH = 500


def app():

    # Sidebar
    project_links()

    x_train, x_test, y_train, y_test, syn = dataset_selector()

    use_gt_db = st.sidebar.expander('Reference decision boundary').\
        checkbox('Use ground-truth decision boundary as reference', value=False)

    model_class, model = model_selector()
    label_new = label_selector()

    show_info()

    base_model = sklearn.base.clone(model)
    base_model.fit(x_train, y_train)

    # Body
    c = st.container()
    with c:
        plot1_placeholder = st.empty()
        x_new = point_slider(c)

    # Computations
    x_train_new, y_train_new = add_point(x_train, y_train, x_new, label_new, syn)
    model.fit(x_train_new, y_train_new)

    baseline_db_diff = DecisionBoundaryDifference(x_lim=X_BOUNDS,
                                                  y_lim=Y_BOUNDS,
                                                  baseline_model=base_model.predict,
                                                  mesh_size=500)

    db_diff = baseline_db_diff.compute_difference(model.predict)

    # Plot
    fig = go.Figure()

    fig.add_trace(get_scatter_trace(x_train_new, y_train_new))

    m = syn.get_labels if use_gt_db else base_model.predict
    fig.add_trace(get_contour_trace_from_model(m, X_BOUNDS, Y_BOUNDS,
                                               NUM_POINTS_MESH, 'True decision boundary',
                                               colorscale='rdbu'))

    fig.add_trace(get_contour_trace_from_model(model.predict, X_BOUNDS, Y_BOUNDS,
                                               NUM_POINTS_MESH, 'Models decision boundary'))

    color = "orange" if y_train_new[-1] == 1 else "blue"
    fig.add_trace(go.Scatter(x=[x_train_new[-1, 0]], y=[x_train_new[-1, 1]],
                             marker=dict(size=20, color=color, symbol='diamond'), showlegend=False))

    fig.update_layout(showlegend=False,
                      title={
                          'text': f'Data value of new point with {model_class.NAME}: {db_diff}',
                          'y': 0.85,
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'},
                      )

    plot1_placeholder.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(page_title="Experiment with decision boundary",
                       page_icon="images/dallemini_2022-7-9_15-49-15.png",
                       layout="centered")
    app()
