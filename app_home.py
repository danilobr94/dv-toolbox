"""Entrance point for data value computation."""
import sklearn.base
import sklearn.base
from ui.functions import *
from ui.sidebar_components import *


def _introduction():
    st.title("**Decision Boundary PlayGround**")
    st.subheader(
        """
        A simple tool to investigate the effect of data points on the decision boundary!
        """
    )

    # st.markdown(
    #     """
    # - 🗂️ Choose a dataset
    # - ⚙️ Pick a model and set its hyper-parameters
    # - 📉 Train it and check its performance metrics and decision boundary on train and test data
    # - 🩺 Diagnose possible Decision Boundary changes based on new points in the Plot
    # -----
    # """
    # )


def app():
    # sidebar
    x_train, x_test, y_train, y_test, syn = dataset_selector()
    model_class, model = model_selector()
    num_repetitions = num_repetitions_selector()
    x_new, label_new = point_slider()
    show_info()

    base_model = sklearn.base.clone(model)
    base_model.fit(x_train, y_train)

    # body
    _introduction()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    col1, col2 = st.columns((1, 1))
    # duration_placeholder = st.empty()

    with col1:
        plot1_placeholder = st.empty()

    with col2:
        plot2_placeholder = st.empty()

    plot3_placeholder = st.empty()

    model_url_placeholder = st.empty()

    # Computations
    x_train_new, y_train_new = add_new_point(
        x_train, y_train, x_new, label_new, syn)
    model.fit(x_train_new, y_train_new)

    db_diff = comp_diff(base_model, model, inv_diff=False)

    if num_repetitions == 1:
        sct1 = point_added_graph(model, base_model,
                                 x_train_new, y_train_new,
                                 x_test, y_test, save_gif=None,
                                 title=("Decision boundary change (DV):" + str(db_diff[-1])))
        plot1_placeholder.pyplot(sct1)

        sct2 = point_added_graph(model, syn,
                                 x_train_new, y_train_new,
                                 x_test, y_test, save_gif=None,
                                 title=(
                                         "Decision boundary change w.r.t. gt model (DV):" + str(db_diff[-1])))
        plot2_placeholder.pyplot(sct2)

    else:
        sct1 = plot_multiple_iterations(X_train_new=x_train_new,
                                        y_train_new=y_train_new,
                                        X_test=x_test,
                                        y_test=y_test,
                                        num_iter=num_repetitions,
                                        model=model,
                                        train_on_x_new=False,
                                        title=("Decision boundary change on original data " + str(db_diff[-1])))
        plot1_placeholder.pyplot(sct1)

        sct2 = plot_multiple_iterations(X_train_new=x_train_new,
                                        y_train_new=y_train_new,
                                        X_test=x_test,
                                        y_test=y_test,
                                        num_iter=num_repetitions,
                                        model=model,
                                        train_on_x_new=True,
                                        title=("Decision boundary change with new point " + str(db_diff[-1])))
        plot2_placeholder.pyplot(sct2)

    train_accuracy_bmodel, test_accuracy_bmodel, train_accuracy_model = accuracy_model(
        model, base_model, x_train_new, y_train_new,
        x_train, y_train, x_test, y_test
    )

    metric = {
        "train_accuracy_bmodel": train_accuracy_bmodel,
        "test_accuracy_bmodel": test_accuracy_bmodel,
        "train_accuracy_model": train_accuracy_model,
    }

    fig3 = show_accuracy(metric)
    plot3_placeholder.plotly_chart(fig3, use_container_width=True)
    # duration_placeholder.warning(f"Training took {duration:.3f} seconds")
    model_url_placeholder.markdown(model_class.URL)


if __name__ == "__main__":
    app()