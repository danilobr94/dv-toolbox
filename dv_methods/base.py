from abc import ABC, abstractmethod

import numpy as np


class DVMethodBase(ABC):
    """Base class for methods to compute data values.

    The '__init__()' function should set the necessary parameters via streamlint components.
    """

    NAME: str
    URL: str

    @abstractmethod
    def __init__(self, X_base=None, y_base=None):
        """"""

    @abstractmethod
    def predict_dv(self, X, y) -> (np.ndarray, callable):
        ...