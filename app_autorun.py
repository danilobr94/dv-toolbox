"""Automatically running data value computation on the mesh grid and creating a gif as output."""
import sklearn.base
from stqdm import stqdm
from ui.functions import *
from ui.sidebar_components import *


def app():
    # Sidebar
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

    x = np.arange(-10, 21, step)
    crdnts = np.fliplr(np.array(np.meshgrid(x, x)).T.reshape(-1, 2))

    for crdnt in stqdm(crdnts):
        st.spinner(text="In progress...")
        # fname = 'dv_loo{}.png'.format(crdnt)
        x_train_new, y_train_new = add_new_point(x_train,
                                                 y_train,
                                                 crdnt,
                                                 label_new,
                                                 syn)

        model.fit(x_train_new, y_train_new)

        db_diff = comp_diff(base_model, model, inv_diff=False)

        sct = point_added_graph(model, base_model,
                                x_train_new, y_train_new,
                                x_test, y_test, save_gif=False,
                                title=("Decision boundary changev(DV):" + str(db_diff[-1])))
        plot1_placeholder.pyplot(sct)

    gif_gen()
    st.markdown(get_binary_file_downloader_html(
        'out/dv_gif.gif', 'Export GIF'), unsafe_allow_html=True)
    # duration_placeholder.warning(f"Training took {duration:.3f} seconds")
    model_url_placeholder.markdown(model_class.URL)


if __name__ == "__main__":
    app()