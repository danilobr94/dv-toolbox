""""""
import sklearn.base
import sklearn.base
import matplotlib.pyplot as plt
from data.decision_boundary import Scatter2D
from ui.sidebar_components import *
from utils import *


def plot_multiple_iterations(X_train_new, y_train_new, X_test, y_test, # noqa
                             num_iter, model, train_on_x_new=False,
                             title='Decision boundary stability'):
    """"""

    sct = Scatter2D(X_train_new[:-1, :], y_train_new[:-1],  # hide the new point here
                    X_test, y_test,
                    x_lim=(-10, 20), y_lim=(-10, 20), )

    for _ in range(num_iter):
        new_model = sklearn.base.clone(model)

        if train_on_x_new:
            new_model.fit(X_train_new, y_train_new)
        else:
            new_model.fit(X_train_new[:-1, :], y_train_new[:-1])

        sct.add_boundary(new_model.predict)

    # add diamond for the new point
    color = "grey" if not train_on_x_new else "tab:orange" if y_train_new[-1] == 1 else "tab:blue"
    plt.scatter(X_train_new[-1, 0], X_train_new[-1, 1], marker="D", color=color, s=100)

    return sct.show(title=title)


def app():

    # sidebar
    project_links()

    x_train, x_test, y_train, y_test, syn = dataset_selector()
    model_class, model = model_selector()
    num_repetitions = num_repetitions_selector()
    label_new = label_selector()
    x_new = point_slider()
    show_info()

    # body
    st.set_option('deprecation.showPyplotGlobalUse', False)
    col1, col2 = st.columns((1, 1))

    with col1:
        plot1_placeholder = st.empty()

    with col2:
        plot2_placeholder = st.empty()

    model_url_placeholder = st.empty()

    # Computations
    base_model = sklearn.base.clone(model)
    base_model.fit(x_train, y_train)

    x_train_new, y_train_new = add_point(x_train, y_train, x_new, label_new, syn)
    model.fit(x_train_new, y_train_new)

    sct1 = plot_multiple_iterations(X_train_new=x_train_new,
                                    y_train_new=y_train_new,
                                    X_test=x_test,
                                    y_test=y_test,
                                    num_iter=num_repetitions,
                                    model=model,
                                    train_on_x_new=False,
                                    title="Decision boundaries on original data")
    plot1_placeholder.pyplot(sct1)

    sct2 = plot_multiple_iterations(X_train_new=x_train_new,
                                    y_train_new=y_train_new,
                                    X_test=x_test,
                                    y_test=y_test,
                                    num_iter=num_repetitions,
                                    model=model,
                                    train_on_x_new=True,
                                    title="Decision boundaries with new point")

    plot2_placeholder.pyplot(sct2)
    model_url_placeholder.markdown(model_class.URL)


if __name__ == "__main__":
    app()
