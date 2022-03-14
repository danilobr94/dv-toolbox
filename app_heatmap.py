"""Page for plotting heatmap of data values with different methods."""
from ui.functions import *
from ui.sidebar_components import dataset_selector, dv_fn_selector, show_info, step_size_selector
from dv_methods.catastrophic_forgetting import ForgettingDV
from dv_methods.memorization import MemorizationDV
from dv_methods.model_ensemble import EnsembleOOD
from dv_methods.loo import LOODV
from dv_methods.latest_learned import LastLearnedDV
from dv_methods.mc_dropout import MCDV

DV_METHODS = (EnsembleOOD, ForgettingDV, MemorizationDV,
              LOODV, LastLearnedDV, MCDV)


def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # side bar
    X_train, X_test, y_train, y_test, syn = dataset_selector()
    step_size = step_size_selector()
    dv_function = dv_fn_selector(DV_METHODS)(X_train, y_train)
    show_info()

    # body
    col1, col2, col3 = st.columns((1, 4, 1))
    with col1:
        st.markdown('')

    with col2:
        plot1_placeholder = st.empty()

    with col3:
        st.markdown('')

    X_mesh, y_mesh, xx, yy = xy_mesh(step_size=step_size)
    dv, baseline_model = dv_function.predict_dv(X_mesh, y_mesh)
    b_model_func = baseline_model.predict if baseline_model is not None else None

    sct_csf = show_heatmap(X_train, y_train, xx, yy,
                           X_test, y_test, dv, syn.get_labels, b_model_func,)
    st.spinner("Heatmap in Progress")
    plot1_placeholder.pyplot(sct_csf)
    plt.pause(10)


if __name__ == "__main__":
    app()