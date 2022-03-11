import base64
from itertools import count
import os
from pathlib import Path

import imageio
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import sklearn.base
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score

from dv_methods.metrics import DecisionBoundaryDifference
from data.decisionboundary import Scatter2D
from data.synthetic_data import SyntheticData


def local_css(file_name):
    """[load the css file from style.css with different css properties]

    Args:
        file_name : define the path where css file consist of.

    Returns:
        css file and its properties will be loaded upon calling of local_css
        method in main (app.py)
    """
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def add_new_point(X_train, y_train, x_new, label_new, syn):
    """[function to add new point in the training set af x & y]

    Args:
        X_train : previous X training data
        y_train : previous Y training data
        x_new : point to be added in X train
        label_new : assigning labels for y train data
        syn : synthetic data generator

    Returns:
        returns new X_train and y_train with data point
    """

    X_train_new = np.vstack([X_train, x_new])
    label_new = syn.get_labels(x_new) if label_new == "auto" else int(label_new)
    y_train_new = np.hstack([y_train, int(label_new)])

    return X_train_new, y_train_new


count = 1


def point_added_graph(model, base_model, X_train_new, y_train_new, x_test, y_test, save_gif=None,
                      title='Decision boundary change'):
    """[Shows the Plot with boundaries of with and without decision boundaries
    based upon color regions ]

    Args:
        model : assign the model to be choosen for decion boundary showcase
        base_model : model which is replica of 'model' to be used as clone to
        see with and wothout plots
        X_train_new : Point added X train data
        y_train_new : Point added y train data(label)
        x_test, y_test : previous test data
        title: output title with data value showcase of data point
        save_gif (int): If 'None' no plot is created, otherwise the value is used as counter.

    Returns:
        returns scattered plot of with and without data point and with, without
        decision boundary with data value of data point
    """
    global count
    cmap = matplotlib.colors.ListedColormap(['tab:blue', 'tab:red'])
    cmap_base = matplotlib.colors.ListedColormap(['tab:blue', 'tab:green'])

    sct = Scatter2D(X_train_new, y_train_new,
                    x_test, y_test,
                    x_lim=(-10, 20), y_lim=(-10, 20), )

    model.fit(X_train_new, y_train_new)
    sct.add_boundary(model.predict)
    sct.add_boundary(model.predict, cmap=cmap)

    sct.add_boundary(base_model.predict)
    sct.add_boundary(base_model.predict, cmap=cmap_base)

    color = "tab:orange" if y_train_new[-1] == 1 else "tab:blue"
    plt.scatter(X_train_new[-1, 0], X_train_new[-1, 1],
                marker="D", color=color, s=100)

    if save_gif is not None:
        plt.savefig("out/dv_" + str(count) + ".png",
                    dpi=150, bbox_inches='tight')
    count += 1

    return sct.show(title=title)


def plot_multiple_iterations(X_train_new, y_train_new, X_test, y_test,
                             num_iter, model, train_on_x_new=False,
                             title='Decision boundary stability'):
    """[Shows the Plot with number of iterations for boundaries of with and
    without decision boundaries based upon color regions ]

    Args:
        X_train_new : Point added X train data
        y_train_new : Point added y train data(label)
        x_test, y_test : previous test data
        num_iter : number of iterations to be seen in the plot w.r.t to DB
        model : assign the model to be choosen for decion boundary showcase
        train_on_x_new : retraining multiple times on new data point based
        plot dependend upon num_iter
        title: output title with data value showcase of data point

    Returns:
        returns scattered plot of with and without data point and with, without
        decision boundary with data value of data point
    """

    sct = Scatter2D(X_train_new, y_train_new,
                    X_test, y_test,
                    x_lim=(-10, 20), y_lim=(-10, 20), )

    color = "grey" if not train_on_x_new else "tab:orange" if y_train_new[-1] == 1 else "tab:blue"
    plt.scatter(X_train_new[-1, 0], X_train_new[-1, 1],
                marker="D", color=color, s=100)

    for _ in range(num_iter):
        new_model = sklearn.base.clone(model)

        if train_on_x_new:
            new_model.fit(X_train_new, y_train_new)
        else:
            new_model.fit(X_train_new[:-1, :], y_train_new[:-1])

        sct.add_boundary(new_model.predict)

    return sct.show(title=title)


def display_graph():
    """[function to display Synthetic data on a scattered plot graph]

    Returns:
        [scatter]: [Plotting scattered X, y of Synthetic data]
    """
    syn = SyntheticData(num_per_pos_label=(500,), num_per_neg_label=(500,))
    data, labels = syn.sample_initial_data()

    sct = Scatter2D(
        data, labels, x_lim=(-10, 20), y_lim=(-10, 20))

    return sct.show(), syn


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def show_accuracy(metrics):
    """[shows plotly gauges which display the values of accuracy of with
    and without data points]

    Args:
        metrics ([dictionary]): [key: value pairs of method related to accuracy
        scores of train, test data]

    Returns:
        [figure: plotly(1x2)]: [Plotly gauge being returned carry the value of
        Accuracy of training, test data w.r.t with & without data point]
    """

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[
            {"type": "indicator"}, {"type": "indicator"}]],

    )

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=np.float(metrics["train_accuracy_bmodel"]),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Acccuracy: without new point"},
        gauge={"axis": {"range": [0, 1]}}), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=np.float(metrics["train_accuracy_model"]),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Acccuracy with new point"},
        gauge={"axis": {"range": [0, 1]}}), row=1, col=2)

    fig.update_layout(
        height=250,
    )

    return fig


def comp_diff(baseline_model, model, inv_diff=False):
    """[decision boundary difference between model and base_model based
    on selection]

    Args:
        baseline_model ([type]): [clone of model for distance of
        decision boundary]
        model : [model for which boundary difference needed to be computed]
        inv_diff : [calculation of data value - 1]. Defaults to False.

    Returns:
        [int]: [decision boundary difference being calculated]
    """
    # TODO: need to set the limits somewhere central ...
    baseline_db_diff = DecisionBoundaryDifference(x_lim=(-10, 20),
                                                  y_lim=(-10, 20),
                                                  baseline_model=baseline_model.predict,
                                                  mesh_size=500)

    compute_db_diff = baseline_db_diff.compute_difference

    db_diff = []

    if inv_diff:
        db_diff.append(1 - compute_db_diff(model))
    else:
        db_diff.append(compute_db_diff(model))

    return db_diff


def accuracy_model(model, base_model, X_train_new, y_train_new, X_train,
                   y_train, x_test, y_test):
    """[Calculate accuracy with and without data points]

    Args:
        model : [model to be used for calculating accuracy]
        base_model : [base model accuracy]
        X_train_new : [X train with data point]
        y_train_new : [y train with data point label]
        X_train : [X train without data point]
        y_train : [y train without data point]
        x_test : [x test without point]
        y_test : [y test without data point label]

    Returns:
        [numpy float]: [returns different model accuracy with and
        without point]
    """
    base_model.fit(X_train, y_train)

    y_train_pred_base = base_model.predict(X_train)
    y_test_pred_base = base_model.predict(x_test)
    train_accuracy_bmodel = np.round(
        accuracy_score(y_train, y_train_pred_base), 3)

    test_accuracy_bmodel = np.round(
        accuracy_score(y_test, y_test_pred_base), 3)

    model.fit(X_train_new, y_train_new)
    y_train_pred = model.predict(X_train_new)

    train_accuracy_model = np.round(
        accuracy_score(y_train_new, y_train_pred), 3)

    return train_accuracy_bmodel, test_accuracy_bmodel, train_accuracy_model


def xy_mesh(x_lim=(-10, 25,), y_lim=(-10, 25,), step_size=1):
    """[Calculate x-y mesh for a specified x, y limits with step size]

    Args:
        x_lim : [x limit points]
        y_lim : [y limit points]
        step_size : [step_size of points in plot]

    Returns:
        [numpy float]: [returns X, y mesh grid values]
    """
    xx, yy = np.meshgrid(np.arange(x_lim[0], x_lim[1], step_size),
                         np.arange(y_lim[0], y_lim[1], step_size))
    syn = SyntheticData()

    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    y_mesh = syn.get_labels(X_mesh)

    return X_mesh, y_mesh, xx, yy


def baseline_df(baseline_model, syn):
    """[Calculate baseline difference between 2 models w.r.t diff. points]

    Args:
        baseline_model : [baseline model]
        syn : [Scatter object for initial data, label]

    Returns:
        [float]: [returns baseline difference value for model1, 2 , colormaps for the models]
    """
    cmap = matplotlib.colors.ListedColormap(['tab:blue', 'tab:red'])
    cmap_base = matplotlib.colors.ListedColormap(['tab:blue', 'tab:green'])
    baseline_db_diff = DecisionBoundaryDifference(x_lim=(-10, 25),
                                                  y_lim=(-10, 25),
                                                  baseline_model=baseline_model.predict,
                                                  mesh_size=500)

    gt_db_diff = DecisionBoundaryDifference(x_lim=(-10, 25),
                                            y_lim=(-10, 25),
                                            baseline_model=syn.get_labels,
                                            mesh_size=100)
    return baseline_db_diff, gt_db_diff, cmap, cmap_base


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def show_heatmap(X_train, y_train, xx, yy, X_test, y_test, dv, syn_func, model_func,):
    """[Show the heatmap for data value methods]

    Args:
        X_train : [Training data]
        y_train : [Training data(labels)]
        xx, yy : [returned grid value]
        X_test : [Test data]
        y_test : [Test data(labels)]
        dv: [data value object]
        syn_func : [Scatter object function]
        model_func : [baseline model function]

    Returns:
        [scatter]: [returns Scatter Plot heatmap]
    """
    """
    baseline_model = MLP(hidden_layer_sizes=1000,
                         activation='relu', max_iter=1000)

    """
    st.spinner('In Progress')
    dv = np.asarray(dv).reshape(xx.shape)
    im = plt.contourf(xx, yy, dv, cmap="RdBu", alpha=.5)

    sct = Scatter2D(X_train, y_train,
                    X_test, y_test,
                    x_lim=(-10, 20), y_lim=(-10, 20))

    sct.scatter(X_train, y_train)

    plt.colorbar(im)
    sct.add_boundary(syn_func)

    if model_func is not None:
        sct.add_boundary(model_func)

    return sct.show(scatter=False, title="Data Values")


def gif_gen():
    """[generate and store images generated in a dir. and return gif after storing them in a collection of png]

    Returns:
        [.gif]: [gif from png collection]
    """
    png_dir = 'out'
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave('out/dv_gif.gif', images, fps=2, format='.gif')
    test = os.listdir(png_dir)
    for images in test:
        if images.endswith(".png"):
            os.remove(os.path.join(png_dir, images))


def get_binary_file_downloader_html(bin_file="out/dv_gif.gif", file_label='GIF'):
    """[Show the heatmap for data value methods]

    Args:
        bin_file : [Selected gif(path) file to be downloaded]
        file_label : [label of type of file: GIF]

    Returns:
        [http]: [Link reference to download the gif]
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}"> {file_label}</a>'
    return href
