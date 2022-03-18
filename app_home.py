"""Entrance point for data value computation."""
import sklearn.base
import sklearn.base
from dv_methods.metrics import DecisionBoundaryDifference
from ui.sidebar_components import *
from utils import *

NUM_POINTS_MESH = 500


def _introduction():
    st.title("**Decision Boundary PlayGround**")


def app():

    # sidebar
    project_links()

    x_train, x_test, y_train, y_test, syn = dataset_selector()
    model_class, model = model_selector()
    x_new, label_new = point_slider()
    show_info()

    base_model = sklearn.base.clone(model)
    base_model.fit(x_train, y_train)

    # body
    _introduction()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    col1, col2 = st.columns((1, 1))

    with col1:
        plot1_placeholder = st.empty()

    model_url_placeholder = st.empty()

    # Computations
    x_train_new, y_train_new = add_point(x_train, y_train, x_new, label_new, syn)
    model.fit(x_train_new, y_train_new)

    baseline_db_diff = DecisionBoundaryDifference(x_lim=X_BOUNDS,
                                                  y_lim=Y_BOUNDS,
                                                  baseline_model=base_model.predict,
                                                  mesh_size=500)

    # TODO report the data value
    db_diff = baseline_db_diff.compute_difference(model.predict)

    # Plot
    fig = go.Figure()

    fig.add_trace(get_scatter_trace(x_train_new, y_train_new))
    fig.add_trace(get_contour_trace_from_model(syn.get_labels, X_BOUNDS, Y_BOUNDS,
                                               NUM_POINTS_MESH, 'True decision boundary',
                                               colorscale='rdbu'))

    # TODO add option to compare to true db
    fig.add_trace(get_contour_trace_from_model(base_model.predict, X_BOUNDS, Y_BOUNDS,
                                               NUM_POINTS_MESH, 'Models decision boundary'))

    color = "orange" if y_train_new[-1] == 1 else "blue"
    fig.add_trace(go.Scatter(x=[x_train_new[-1, 0]], y=[x_train_new[-1, 1]],
                             marker=dict(size=20, color=color, symbol='diamond'), showlegend=False))

    plot1_placeholder.plotly_chart(fig)

    # TODO: print the accuracy
    base_acc_test = base_model.score(x_test, y_test)
    new_model_acc_test = model.score(x_test, y_test)

    model_url_placeholder.markdown(model_class.URL)


if __name__ == "__main__":
    app()
