from interpret_extension.glassbox import LogisticRegression
import numpy as np
from models.basemodel import BaseModel

class LinearModelInterpret(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)
        self.model = LogisticRegression(**params)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
        }
        return params

    @classmethod
    def get_random_parameters(cls, seed):
        rs = np.random.RandomState(seed)
        params = {
            "C": rs.uniform(1e-4, 1e4),
            "penalty": rs.choice(["l1", "l2"]),
            "solver": rs.choice(["liblinear", "saga"]),
        }
        return params

    @classmethod
    def default_parameters(cls):
        params = {
            "C": 1.0,
            "penalty": "l2",
            "solver": "liblinear",
        }
        return params