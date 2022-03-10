from abc import ABC, abstractmethod


class ModelBase(ABC):
    """Base class for sklearn models."""

    NAME: str
    URL: str

    @staticmethod
    @abstractmethod
    def param_selector():
        ...