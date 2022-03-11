"""Entrance point for data value computation."""
import sklearn.base
import sklearn.base
from ui.functions import *
from ui.sidebar_components import *


def _introduction():
    st.title("**Decision Boundary PlayGround**")
    st.subheader("""---TEST APP---""")


def app():
    # sidebar
    x_train, x_test, y_train, y_test, syn = dataset_selector()
    model_class, model = model_selector()
    x_new, label_new = point_slider()
    show_info()

    base_model = sklearn.base.clone(model)
    base_model.fit(x_train, y_train)

    # body
    _introduction()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plot1_placeholder = st.empty()

    # Computations
    x_train_new, y_train_new = add_new_point(x_train, y_train, x_new, label_new, syn)
    model.fit(x_train_new, y_train_new)

    db_diff = comp_diff(base_model, model, inv_diff=False)

    sct1 = point_added_graph(model, base_model,
                             x_train_new, y_train_new,
                             x_test, y_test, save_gif=None,
                             title=("Decision boundary change (DV):" + str(db_diff[-1])))
    plot1_placeholder.pyplot(sct1)


if __name__ == "__main__":
    app()