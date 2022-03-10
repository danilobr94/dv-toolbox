# Data Value Toolbox(v0.1.0)

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/streamlit/demo-face-gan)

[![Generic badge](https://img.shields.io/badge/<IPA>-<DATA_Value_Toolbox>-<COLOR>.svg)](https://shields.io/)

# Structure Conventions for Dava Value Toolbox

> Folder structure options and naming conventions for data value toolbox project

### A typical top-level directory layout

    ├── apps                  # All Application Pages comes under it
    ├── css                   # css files for streamlit designing
    ├── images                # All icons, thumbnails images
    ├── models                # All ML models with parameters tunning
    ├── out                   # New Generated Gif from Page 3(DB Auto) comes under it
    ├── utils                 # all temp. generated functions files
    └── dv_methods            # It contains all data values predictable methods
    └── Plottting             # decision boundary plotting and synthetic data generator of 2D type comes under it
    └── README.md

### Source files

The actual source files of data value project are stored inside the`apps` folder
which consist of `home.py`: start point of app, our first page of application,
`dv_methods.py`: it consist of various data value methods along with there
respective model selection and the plotted results, `app_home.py`: It is our
3rd page of the application , which runs our Data value graph with automate
process depends upon the size of the steps size given by user.

### dv_methods

The dv_methods dir. mainly contails all of the Predict data values functions

### Plotting

Along with it `Scatter2D, decision boundary and metrics` like files, which
helps to generate 2d Scatter plots, adding decision boundaries and metrics
to check the accuracy.

### ui

It also contains `ui.py` file which contains all ui based features, functions used in the toolbox.
`functions.py` which contains the required functions to be used
by 3 apps pages in our application.

### out

Generated GIF from Auto DB Page will showcase DV from starting till end (Depends upon step size)

along with it, `requirements.txt` to pip install the project, Please use (Python: 3.7.9)

## Installation

```bash
pip install requirements.txt
```

## Run : Toolbox?

```bash
Streamlit run app.py
```

## Example

Start Page of the Application

![App Home Page](images\App_Home.PNG "App")

![App Home Page](out\dv_gif.gif "App")

## Streamlit Cloud

[Streamlit Cloud](https://streamlit.io/cloud) is our deployment solution for managing, sharing, and collaborating on your Streamlit apps.

- The Teams and Enterprise tiers provide secure single-click deploy, authentication, web editing, versioning, and much more for your Streamlit apps. You can sign-up [here](https://share.streamlit.io/signup).
- The Community tier (formerly Streamlit sharing) is the perfect solution if your app is hosted in a public GitHub repo and you’d like anyone in the world to be able to access it. It's completely free to use and you can sign-up [here](https://share.streamlit.io).

## License

Streamlit is completely free and open-source and licensed under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.
