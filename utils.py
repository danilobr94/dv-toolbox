"""Utility functions"""
import plotly.graph_objs as go
import numpy as np
from ui.sidebar_components import LABEL_AUTO


def add_point(X_train, y_train, x_new, label_new, syn):  # noqa
    """Appends 'x_new' to 'X_train' and 'label_new' to 'y_train'."""
    X_train_new = np.vstack([X_train, x_new])  # noqa
    label_new = syn.get_labels(x_new) if label_new == LABEL_AUTO else int(label_new)
    y_train_new = np.hstack([y_train, int(label_new)])
    return X_train_new, y_train_new


def get_mesh(x_lim, y_lim, num_points):
    """"""
    x1_min, x1_max = x_lim
    x2_min, x2_max = y_lim

    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, num_points),
                         np.linspace(x2_min, x2_max, num_points))

    return xx, yy


def get_scatter_trace(X, y):  # noqa
    """Returns plotly trace for scattering."""
    return go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y))


def get_contour_trace_from_model(model, x_lim, y_lim, num_points, name, colorscale='greens'):
    """"""
    xx, yy, = get_mesh(x_lim, y_lim, num_points)

    X_mesh = np.c_[xx.ravel(), yy.ravel()]  # noqa
    Z = model(X_mesh)  # noqa
    Z = Z.reshape(xx.shape)  # noqa
    y_ = np.linspace(y_lim[0], y_lim[1], num_points)

    return go.Contour(z=Z, x=xx[0], y=y_,
                      contours_coloring='lines',
                      line_width=2,
                      showscale=False,
                      ncontours=2,
                      colorscale=colorscale,
                      hoverinfo='skip',
                      name=name)


def get_heatmap_trace_from_model(model, x_lim, y_lim, num_points, name):
    """"""
    xx, yy, = get_mesh(x_lim, y_lim, num_points)

    X_mesh = np.c_[xx.ravel(), yy.ravel()]  # noqa
    Z = model(X_mesh)  # noqa
    Z = Z.reshape(xx.shape)  # noqa

    y_ = np.linspace(y_lim[0], y_lim[1], num_points)

    return go.Contour(z=Z, x=xx[0], y=y_, showscale=False,
                      line_smoothing=0.85, name=name)


def get_contour_trace(Z, x_lim, y_lim, num_points, name=""):  # noqa
    """"""
    x_ = np.linspace(x_lim[0], x_lim[1], num_points)
    y_ = np.linspace(y_lim[0], y_lim[1], num_points)
    Z = Z.reshape((x_.shape[0], x_.shape[0]))  # noqa

    return go.Contour(z=Z, x=x_, y=y_,
                      contours_coloring='lines',
                      line_width=2,
                      ncontours=2,
                      showscale=False,
                      hoverinfo='none',
                      name=name)


def get_heatmap_trace(Z, x_lim, y_lim, num_points, name=""):  # noqa
    """"""
    x_ = np.linspace(x_lim[0], x_lim[1], num_points)
    y_ = np.linspace(y_lim[0], y_lim[1], num_points)
    Z = Z.reshape((x_.shape[0], x_.shape[0]))  # noqa

    return go.Contour(z=Z, x=x_, y=y_, colorscale='RdBu', zmid=1,
                      showscale=False, name=name,
                      line_smoothing=0.3)
