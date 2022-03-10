import streamlit as st
from apps import app_home, app_autorun, app_heatmap
from ui.functions import local_css  # import your app modules here
st.set_page_config(page_title="Decision Boundary Playground", page_icon="./images/flask.png")

local_css("./css/style.css")


class MultiApp:
    """Framework for combining multiple streamlit applications.

    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()

    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()

    TODO:
        * add source of code
    """

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        app = st.sidebar.radio(
            'Go To',
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()


app = MultiApp()

# Add all your application here
app.add_app("Home", app_home.app)
app.add_app("DV Methods", app_heatmap.app)
app.add_app("Auto Run", app_autorun.app)

if __name__ == "__main__":
    app.run()
