from interpret_extension.glassbox import GaussianNB
import numpy as np
from models.basemodel import BaseModel

class GaussianNaiveBayesModel(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)
        self.model = GaussianNB(**params)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-9, 1e-7, step=1e-8),
        }
        return params

    @classmethod
    def get_random_parameters(cls, seed):
        rs = np.random.RandomState(seed)
        params = {
            "var_smoothing": rs.uniform(1e-9, 1e-7),
        }
        return params

    @classmethod
    def default_parameters(cls):
        params = {
            "var_smoothing": 1e-8,
        }
        return params