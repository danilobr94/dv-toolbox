"""Utility functions."""
import plotly.graph_objs as go
import numpy as np
import streamlit as st
import time

from ui.data_selection_components import LABEL_AUTO


class StreamlitProgressBar:
    """Custom sidebar with text.

    This was necessary, because 'stqdm' sometimes caused the app to freeze.
    """

    def __init__(self, iterable):
        """"""
        self.iterable = iterable

        self.current_item = 0
        self.num_items = len(iterable)

        self.text = st.container().empty()

        self.prev_time = time.time()
        self.times = []

        self.p_bar = st.progress(0.0)

    def _format_text(self):
        """"""
        avg = np.mean(self.times)
        eta = avg * (self.num_items - self.current_item)
        total = np.sum(self.times)

        return f"{self.current_item}/{self.num_items} [{round(total, 2)} < {round(eta, 2)}, {round(avg, 2)}s/it]"

    def __next__(self):
        """"""
        current_time = time.time()

        if self.current_item >= self.num_items:
            self.p_bar.empty()
            self.text.empty()
            raise StopIteration

        self.times.append(current_time - self.prev_time)
        self.prev_time = current_time

        item = self.iterable[self.current_item]
        self.current_item += 1

        self.text.empty()
        self.text.write(self._format_text())
        self.p_bar.progress(self.current_item / self.num_items)

        return item

    def __iter__(self):
        """"""
        return self


def add_point(X_train, y_train, x_new, label_new, syn):  # noqa
    """Appends 'x_new' to 'X_train' and 'label_new' to 'y_train'."""
    print(X_train.shape, x_new.shape)
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
