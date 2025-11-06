from interpret_extension.glassbox import LinearDiscriminantAnalysisClassifier
import numpy as np
from models.basemodel import BaseModel

class LDAModel(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)
        self.model = LinearDiscriminantAnalysisClassifier(**params)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "solver": trial.suggest_categorical("solver", ["svd", "lsqr", "eigen"]),
        }
        return params

    @classmethod
    def get_random_parameters(cls, seed):
        rs = np.random.RandomState(seed)
        params = {
            "solver": rs.choice(["svd", "lsqr", "eigen"]),
        }
        return params

    @classmethod
    def default_parameters(cls):
        params = {
            "solver": "svd",
            "shrinkage": None,
        }
        return params
    
    def get_classes(self):
        return self.model.classes_