from sklearn.naive_bayes import GaussianNB
from .base import ModelBase


class NaiveBayes(ModelBase):
    NAME = 'Naive Bayes'
    URL = 'https://scikit-learn.org/stable/modules/naive_bayes.html'

    @staticmethod
    def param_selector():
        model = GaussianNB()
        return model
