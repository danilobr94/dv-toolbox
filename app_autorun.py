"""Automatically running data value computation on the mesh grid and creating a gif as output."""
import sklearn.base
from stqdm import stqdm
from utils import add_point
from ui.sidebar_components import *
from dv_methods.metrics import DecisionBoundaryDifference
from data.decision_boundary import Scatter2D
import matplotlib.pyplot as plt
import matplotlib


def app():
    # Sidebar
    project_links()

    x_train, x_test, y_train, y_test, syn = dataset_selector()
    model_class, model = model_selector()
    step = step_size_selector()

    base_model = sklearn.base.clone(model)
    base_model.fit(x_train, y_train)
    label_new = label_selector()
    show_info()

    # Body
    st.set_option('deprecation.showPyplotGlobalUse', False)
    col1, col2, col3 = st.columns((1, 4, 1))
    with col1:
        st.markdown('')

    with col2:
        plot1_placeholder = st.empty()

    with col3:
        st.markdown('')

    model_url_placeholder = st.empty()

    baseline_db_diff = DecisionBoundaryDifference(x_lim=X_BOUNDS,
                                                  y_lim=Y_BOUNDS,
                                                  baseline_model=base_model.predict,
                                                  mesh_size=500)

    cmap = matplotlib.colors.ListedColormap(['tab:blue', 'tab:red'])
    cmap_base = matplotlib.colors.ListedColormap(['tab:blue', 'tab:green'])
    sct = Scatter2D(x_train, y_train, x_test, y_test, x_lim=X_BOUNDS, y_lim=Y_BOUNDS)

    x = np.arange(-10, 21, step)
    crdnts = np.fliplr(np.array(np.meshgrid(x, x)).T.reshape(-1, 2))

    for crdnt in stqdm(crdnts):
        st.spinner(text="In progress...")
        x_train_new, y_train_new = add_point(x_train, y_train, crdnt, label_new, syn)

        model.fit(x_train_new, y_train_new)
        db_diff = baseline_db_diff.compute_difference(model.predict)

        sct.add_boundary(model.predict)
        sct.add_boundary(model.predict, cmap=cmap)

        sct.add_boundary(base_model.predict)
        sct.add_boundary(base_model.predict, cmap=cmap_base)

        color = "tab:orange" if y_train_new[-1] == 1 else "tab:blue"
        plt.scatter(x_train_new[-1, 0], x_train_new[-1, 1], marker="D", color=color, s=100)
        plt.title("Data Value: " + str(db_diff))

        plot1_placeholder.pyplot(sct.show())

    # gif_gen()
    # st.markdown(get_binary_file_downloader_html('out/dv_gif.gif', 'Export GIF'), unsafe_allow_html=True)
    model_url_placeholder.markdown(model_class.URL)


if __name__ == "__main__":
    app()
