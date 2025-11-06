from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
import numpy as np
from models.basemodel import BaseModel

"""
    Define EBM models implemented by the InterpretML library:
"""

class EBMModel(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = ExplainableBoostingRegressor(n_jobs=-1, **params)
        elif args.objective == "classification":
            self.model = ExplainableBoostingClassifier(n_jobs=-1, **params)
        elif args.objective == "binary":
            self.model = ExplainableBoostingClassifier(n_jobs=-1, **params)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "max_bins": trial.suggest_categorical("max_bins", [128, 256, 512, 1024]),
            "interactions": trial.suggest_categorical("interactions", [0, 0.1, 0.5, 0.9]),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, step=0.005),
            "max_rounds": trial.suggest_categorical("max_rounds", [1000, 5000, 10000]),
        }
        return params

    @classmethod
    def get_random_parameters(cls, seed):
        rs = np.random.RandomState(seed)
        params = {
            "max_bins": rs.choice([128, 256, 512, 1024]),
            "interactions": rs.choice([0, 0.1, 0.5, 0.9]),
            "learning_rate": rs.uniform(0.005, 0.05),
            "max_rounds": rs.choice([1000, 5000, 10000]),
        }
        return params

    @classmethod
    def default_parameters(cls):
        params = {
            "max_bins": 256,
            "interactions": 0.1,
            "learning_rate": 0.015,
            "max_rounds": 5000,
        }
        return params

    def get_classes(self):
        return self.model.classes_